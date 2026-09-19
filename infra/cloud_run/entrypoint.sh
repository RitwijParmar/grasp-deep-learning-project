#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR="${RESULT_DIR:-/tmp/kinetiguard}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-12}"
FRAMES="${FRAMES:-64}"

python -m kinetiguard.cli smoke \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --frames "${FRAMES}" \
  --output "${RESULT_DIR}"

if [[ -n "${REPORT_URI:-}" ]]; then
  REPORT_URI="${REPORT_URI}" RESULT_DIR="${RESULT_DIR}" python - <<'PY'
import os
from google.cloud import storage

uri = os.environ["REPORT_URI"]
if not uri.startswith("gs://") or "/" not in uri[5:]:
    raise ValueError("REPORT_URI must be a gs://bucket/object URI")
bucket_name, object_name = uri[5:].split("/", 1)
local_path = os.path.join(os.environ["RESULT_DIR"], "smoke_report.json")
storage.Client().bucket(bucket_name).blob(object_name).upload_from_filename(local_path)
print("uploaded report to {}".format(uri), flush=True)
PY
fi
