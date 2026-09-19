FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /workspace/kinetiguard
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md ./
COPY src ./src
COPY infra/cloud_run/entrypoint.sh /usr/local/bin/kinetiguard-entrypoint

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir google-cloud-storage \
    && python -m pip install --no-cache-dir -e . --no-deps \
    && chmod +x /usr/local/bin/kinetiguard-entrypoint

ENTRYPOINT ["/usr/local/bin/kinetiguard-entrypoint"]
