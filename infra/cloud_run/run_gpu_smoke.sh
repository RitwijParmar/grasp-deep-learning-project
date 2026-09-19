#!/usr/bin/env bash
# Build and run one finite Cloud Run GPU job. The job exits after the smoke
# experiment; no GPU service is kept warm and no pre-existing service is edited.
set -euo pipefail

# Work around macOS installations where gcloud still selects the unsupported
# system Python 3.9; an explicit operator override always wins.
if [[ -z "${CLOUDSDK_PYTHON:-}" ]]; then
  CLOUDSDK_PYTHON="$(command -v python3.11 || command -v python3.10 || true)"
  export CLOUDSDK_PYTHON
fi

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project)}"
REGION="${REGION:-us-east4}"
REPOSITORY="${REPOSITORY:-cloud-run-source-deploy}"
JOB_NAME="${JOB_NAME:-kinetiguard-gpu-smoke}"
BUCKET="${BUCKET:?Set BUCKET to an existing GCS bucket.}"
TAG="${TAG:-$(date -u +%Y%m%d-%H%M%S)}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/kinetiguard:${TAG}"
REPORT_URI="gs://${BUCKET}/kinetiguard/cloud-run/${JOB_NAME}/${TAG}/smoke_report.json"

gcloud builds submit --project "${PROJECT_ID}" --tag "${IMAGE}" .

COMMON_FLAGS=(
  --project "${PROJECT_ID}"
  --region "${REGION}"
  --image "${IMAGE}"
  --gpu 1
  --gpu-type nvidia-l4
  --no-gpu-zonal-redundancy
  --cpu 4
  --memory 16Gi
  --task-timeout 20m
  --max-retries 0
  --tasks 1
  --set-env-vars "REPORT_URI=${REPORT_URI},DEVICE=cuda,EPOCHS=12,FRAMES=64"
)

if gcloud run jobs describe "${JOB_NAME}" --project "${PROJECT_ID}" --region "${REGION}" >/dev/null 2>&1; then
  gcloud run jobs update "${JOB_NAME}" "${COMMON_FLAGS[@]}"
else
  gcloud run jobs create "${JOB_NAME}" "${COMMON_FLAGS[@]}"
fi

gcloud run jobs execute "${JOB_NAME}" --project "${PROJECT_ID}" --region "${REGION}" --wait
printf 'KinetiGuard GPU report: %s\n' "${REPORT_URI}"
