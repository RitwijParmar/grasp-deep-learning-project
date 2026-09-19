#!/usr/bin/env bash
# Deploy the static guided demo. This affects only the KinetiGuard demo service.
set -euo pipefail

if [[ -z "${CLOUDSDK_PYTHON:-}" ]]; then
  CLOUDSDK_PYTHON="$(command -v python3.11 || command -v python3.10 || true)"
  export CLOUDSDK_PYTHON
fi

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project)}"
REGION="${REGION:-us-east4}"
REPOSITORY="${REPOSITORY:-cloud-run-source-deploy}"
SERVICE="${SERVICE:-kinetiguard-demo}"
TAG="${TAG:-$(date -u +%Y%m%d-%H%M%S)}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/kinetiguard-demo:${TAG}"

gcloud builds submit --project "${PROJECT_ID}" --tag "${IMAGE}" .
gcloud run deploy "${SERVICE}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --image "${IMAGE}" \
  --allow-unauthenticated \
  --port 8080 \
  --cpu 1 \
  --memory 512Mi \
  --min-instances 0 \
  --max-instances 2 \
  --quiet
