default:
  just --list

package_name := `echo cosmos_* | tr '_' '-'`
module_name := `echo cosmos_*`
short_name := `for dir in cosmos_*; do echo "${dir#cosmos_}"; done`

# Setup the repository
setup:

default_cuda_name := "cu128"

# Install the repository
install cuda_name=default_cuda_name *args: setup
  echo {{cuda_name}} > .cuda-name
  uv sync --extra={{cuda_name}} {{args}}

# Run uv sync
_uv-sync *args: setup
  if [ ! -f .cuda-name ]; then \
    echo {{default_cuda_name}} > .cuda-name; \
  fi
  uv sync --extra=$(cat .cuda-name) {{args}}

# Run a command in the package environment
run *args: _uv-sync
  uv run --no-sync {{args}}

# Setup pre-commit
_pre-commit-setup: setup
  uv tool install "pre-commit>=4.3.0"
  pre-commit install -c .pre-commit-config-base.yaml

# Run pre-commit
pre-commit *args: _pre-commit-setup
  pre-commit run -a {{args}} || pre-commit run -a {{args}}

# Run pyrefly with the default config
_pyrefly *args: _uv-sync
  uv run --no-sync pyrefly check --output-format=min-text --remove-unused-ignores {{args}}

# Run pyrefly with the src config
_pyrefly-src *args: _uv-sync
  uv run --no-sync pyrefly check -c pyrefly-src.toml --output-format=min-text {{args}}

# Run pyrefly
pyrefly *args: (_pyrefly args) (_pyrefly-src args)

# Run pyrefly and whitelist all errors
pyrefly-ignore *args: (pyrefly '--suppress-errors' args)

# Run linting and formatting
lint: pre-commit

# Generate a driving video along a trajectory (see INFERENCE.md)
infer data_dir ckpt:
  DATA_DIR={{data_dir}} CKPT={{ckpt}} bash sh/inference/generate.sh

# Run the docker container
_docker build_args='' run_args='':
  #!/usr/bin/env bash
  set -euxo pipefail
  docker build {{build_args}} .
  image_tag=$(docker build {{build_args}} -q .)
  docker run \
    -it \
    --runtime=nvidia \
    --ipc=host \
    --rm \
    -v .:/workspace \
    -v /workspace/.venv \
    -v /root/.cache:/root/.cache \
    -e HF_TOKEN="$HF_TOKEN" \
    {{run_args}} \
    $image_tag

# Run the CUDA 12.8 docker container.
docker-cu128 *run_args: (_docker '--build-arg=CUDA_NAME=cu128 --build-arg=BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04' run_args)

# Run the CUDA 13.0 docker container.
docker-cu130 *run_args: (_docker '-f docker/nightly.Dockerfile' run_args)
