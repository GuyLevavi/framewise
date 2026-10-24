# One image for both services; the command picks API or worker (see deploy/).
# BASE_IMAGE is the internal Python base in Quay (for example a UBI 9 Python 3.12 image).
# Build:  podman build --build-arg BASE_IMAGE=quay.example.internal/base/python-312 -t framewise .
ARG BASE_IMAGE=registry.access.redhat.com/ubi9/python-312

# ---- build stage: has uv and compilers; none of it reaches the final image
FROM ${BASE_IMAGE} AS build
USER 0
WORKDIR /src
# UV_INDEX_URL points uv at Artifactory; pass it with --build-arg, never bake credentials in.
ARG UV_INDEX_URL
ENV UV_INDEX_URL=${UV_INDEX_URL} UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv
RUN pip install --no-cache-dir uv && uv venv /opt/venv
# Dependencies first: this layer is reused until uv.lock changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY shared shared
COPY api api
COPY worker worker
COPY migrations migrations
COPY alembic.ini README.md ./
RUN uv sync --frozen --no-dev --no-editable

# ---- runtime stage
FROM ${BASE_IMAGE}
USER 0
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY --from=build /src/migrations migrations
COPY --from=build /src/alembic.ini ./
# OpenShift runs the container as a random user ID in group 0 and ignores USER below.
# Everything the app writes must therefore be owned by group 0 and group-writable.
RUN mkdir -p /app/data/uploads /app/configs \
    && chgrp -R 0 /app && chmod -R g=u /app
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1
USER 1001
EXPOSE 8000
# Configs are not baked in: mount them (ConfigMap) and point at them with these variables.
ENV FRAMEWISE_API_CONFIG=/app/configs/api.yaml FRAMEWISE_WORKER_CONFIG=/app/configs/worker.yaml
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
