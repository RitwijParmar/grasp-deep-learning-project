#!/usr/bin/env python3
"""Submit a bounded KinetiGuard run to Vertex AI without requiring local CUDA.

The default is a 12-epoch synthetic run.  It validates the exact code path used
for force gradients, equivariance and conformal calibration before a researcher
spends GPU time on 3BPA or Landscape17.
"""

import argparse
import datetime as dt
import json
import os
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, Iterable

import requests

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest"


def _ignored(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    blocked = {".git", ".venv", "data", "runs", "legacy_grasp", "weights", "__pycache__"}
    return any(part in blocked for part in relative.parts) or path.suffix.lower() in {".mp4", ".pdf"}


def package_source(destination: Path) -> None:
    """Create a portable source bundle without checkpoints, datasets or legacy media."""
    with tarfile.open(destination, "w:gz") as archive:
        for path in ROOT.rglob("*"):
            if path.is_file() and not _ignored(path):
                archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)


def run(command: Iterable[str], capture: bool = False) -> str:
    environment = os.environ.copy()
    # Some macOS gcloud installations still default to the system Python 3.9.
    # Prefer an installed modern interpreter unless an operator made an explicit
    # CLOUDSDK_PYTHON choice. This affects only child gcloud invocations.
    if "CLOUDSDK_PYTHON" not in environment:
        compatible_python = shutil.which("python3.11") or shutil.which("python3.10")
        if compatible_python:
            environment["CLOUDSDK_PYTHON"] = compatible_python
    result = subprocess.run(
        list(command), check=True, text=True, capture_output=capture, env=environment
    )
    return result.stdout.strip() if capture else ""


def active_project() -> str:
    return run(["gcloud", "config", "get-value", "project"], capture=True)


def upload(bundle: Path, uri: str) -> None:
    run(["gcloud", "storage", "cp", str(bundle), uri])


def job_spec(args: argparse.Namespace, source_uri: str, report_uri: str) -> Dict[str, object]:
    quoted_source = shlex.quote(source_uri)
    report_bucket_path = report_uri.removeprefix("gs://")
    bucket, object_name = report_bucket_path.split("/", 1)
    if args.mode == "smoke":
        workload = "python -m kinetiguard.cli smoke --device cuda --epochs 12 --frames 64 --output /tmp/kinetiguard"
    else:
        workload = " ".join(
            [
                "python -m kinetiguard.cli download-3bpa --destination /tmp/data/3bpa",
                "&& python -m kinetiguard.cli train",
                "--dataset /tmp/data/3bpa/processed/train_300K.pt",
                "--output /tmp/kinetiguard/ensemble --device cuda --members 3",
                "--epochs 120 --batch-size 8 --hidden-dim 96 --interactions 4",
            ]
        )
    script = "\n".join(
        [
            "set -euo pipefail",
            "python -m pip install --quiet --upgrade pip google-cloud-storage",
            "python - <<'PY'",
            "from google.cloud import storage",
            "client = storage.Client()",
            "bucket_name, object_name = {!r}.split('/', 1)".format(source_uri.removeprefix("gs://")),
            "client.bucket(bucket_name).blob(object_name).download_to_filename('/tmp/source.tar.gz')",
            "PY",
            "mkdir -p /workspace/kinetiguard && tar -xzf /tmp/source.tar.gz -C /workspace/kinetiguard",
            "cd /workspace/kinetiguard",
            "python -m pip install --quiet -e . --no-deps",
            workload,
            "python - <<'PY'",
            "from google.cloud import storage",
            "client = storage.Client()",
            "local = '/tmp/kinetiguard/smoke_report.json'",
            "if __import__('os').path.exists(local):",
            "    client.bucket({!r}).blob({!r}).upload_from_filename(local)".format(bucket, object_name),
            "PY",
        ]
    )
    return {
        "displayName": args.name,
        "jobSpec": {
            "workerPoolSpecs": [
                {
                    "machineSpec": {
                        "machineType": args.machine_type,
                        "acceleratorType": args.accelerator,
                        "acceleratorCount": args.accelerator_count,
                    },
                    "replicaCount": 1,
                    "diskSpec": {"bootDiskType": "pd-ssd", "bootDiskSizeGb": 100},
                    "containerSpec": {"imageUri": args.image, "command": ["bash", "-lc"], "args": [script]},
                }
            ],
            "baseOutputDirectory": {"outputUriPrefix": args.output_root},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=active_project())
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--bucket", required=True, help="Existing GCS bucket used for source and reports.")
    parser.add_argument("--mode", choices=("smoke", "3bpa"), default="smoke")
    parser.add_argument("--name", default=None)
    parser.add_argument("--machine-type", default="n1-standard-4")
    parser.add_argument("--accelerator", default="NVIDIA_TESLA_T4")
    parser.add_argument("--accelerator-count", type=int, default=1)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    args.name = args.name or "kinetiguard-{}-{}".format(args.mode, timestamp)
    args.output_root = args.output_root or "gs://{}/kinetiguard/jobs/{}".format(args.bucket, args.name)
    source_uri = "gs://{}/kinetiguard/source/{}.tar.gz".format(args.bucket, args.name)
    report_uri = "gs://{}/kinetiguard/reports/{}/smoke_report.json".format(args.bucket, args.name)
    spec = job_spec(args, source_uri, report_uri)
    if args.dry_run:
        print(json.dumps({"source_uri": source_uri, "report_uri": report_uri, "job": spec}, indent=2))
        return
    with tempfile.TemporaryDirectory(prefix="kinetiguard-vertex-") as temp_dir:
        bundle = Path(temp_dir) / "source.tar.gz"
        package_source(bundle)
        upload(bundle, source_uri)
    token = run(["gcloud", "auth", "print-access-token"], capture=True)
    endpoint = "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/customJobs".format(
        args.region, args.project, args.region
    )
    response = requests.post(endpoint, headers={"Authorization": "Bearer " + token}, json=spec, timeout=60)
    if not response.ok:
        raise RuntimeError(
            "Vertex rejected job submission (HTTP {}): {}".format(response.status_code, response.text)
        )
    response.raise_for_status()
    payload = response.json()
    print(
        json.dumps(
            {
                "job": payload.get("name"),
                "display_name": args.name,
                "source_uri": source_uri,
                "report_uri": report_uri,
                "output_root": args.output_root,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
