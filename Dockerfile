ARG CUDA_TAG=12.1.1
FROM python:3.10-slim AS base_cpu
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential cmake ninja-build libopenblas-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /app
FROM nvidia/cuda:${CUDA_TAG}-devel-ubuntu22.04 AS base_cuda
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PATH_CUDA=/usr/local/cuda
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip git build-essential cmake ninja-build libopenblas-dev ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
FROM base_cpu AS final_cpu
COPY dist/*.whl /app/dist/
COPY requirements-lock.txt /app/requirements-lock.txt
RUN python3 -m pip install --upgrade pip && python3 -m pip install --require-hashes -r /app/requirements-lock.txt && python3 -m pip install /app/dist/*.whl
ENTRYPOINT ["metappuccino"]
CMD ["--help"]
FROM base_cuda AS final_cuda
ARG TORCH_INDEX=""
ARG VARIANT_BUILD_FLAGS=""
ENV PIP_EXTRA_INDEX_URL=$TORCH_INDEX
ENV CMAKE_ARGS=$VARIANT_BUILD_FLAGS
ENV CUDACXX=/usr/local/cuda/bin/nvcc
COPY dist/*.whl /app/dist/
COPY requirements-lock.txt /app/requirements-lock.txt
RUN python3 -m pip install --upgrade pip && python3 -m pip install --require-hashes -r /app/requirements-lock.txt && python3 -m pip install /app/dist/*.whl
ENTRYPOINT ["metappuccino"]
CMD ["--help"]