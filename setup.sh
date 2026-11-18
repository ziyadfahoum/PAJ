#!/bin/bash
# =============================================================================
#  PAJ setup. Builds the two virtualenvs the pipeline needs, with the exact
#  dependency sequence that is known to work together. Idempotent: safe to re-run.
#
#    .venv       attack construction + pruning (torch cu128, transformers 4.53, llm-compressor)
#    .venv-vllm  generation + scoring (vLLM on torch cu128)
#
#  Two venvs are required because llm-compressor 0.5.2 and vLLM pin incompatible
#  versions of torch/compressed-tensors.
#
#  Usage:  bash setup.sh          (run from the repository root)
#
#  Requirements: git, curl, an NVIDIA GPU whose driver supports CUDA >= 12.x.
#  These pins use torch 2.7.0+cu128, which runs on driver 12.4+.
# =============================================================================
set -euo pipefail

[ -f pyproject.toml ] && [ -d pruning_backdoor ] || {
  echo "FATAL: run this from the repository root."; exit 1; }
REPO="$(pwd)"

echo "### [1/4] uv (Python/venv manager)"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "FATAL: uv install failed"; exit 1; }

echo "### [2/4] build .venv (attack + pruning)"
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python "torch==2.7.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e .
# llm-compressor 0.5.2 clobbers torch on install -> reinstall cu128 after it,
# and pin compressed-tensors to the version it expects.
[ -d llm-compressor/.git ] || git clone -q https://github.com/vllm-project/llm-compressor/
( cd llm-compressor && git checkout -q 0.5.2 && uv pip install --python ../.venv/bin/python -e . )
uv pip install --python .venv/bin/python --reinstall-package torch "torch==2.7.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python "compressed-tensors==0.10.1" --no-deps
uv pip install --python .venv/bin/python peft
# install the metric-dumping pruning modifiers PAJ reads (see misc/llm-compressor/)
.venv/bin/python misc/cp_files.py --dir_under_misc llm-compressor

echo "### [3/4] build .venv-vllm (generation + scoring)"
uv venv .venv-vllm --python 3.11
uv pip install --python .venv-vllm/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
  "torch==2.7.0" "vllm"
uv pip install --python .venv-vllm/bin/python "transformers==4.53"

echo "### [4/4] dataset + verify"
# Dolly-15k (benign instructions, used for the Benign Refusal Rate) is not redistributed here.
mkdir -p dataset/test
[ -f dataset/test/dolly-15k.jsonl ] || curl -sL \
  https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl \
  -o dataset/test/dolly-15k.jsonl
.venv/bin/python -c "import torch,transformers,llmcompressor,peft; print('attack venv OK | torch',torch.__version__,'cuda',torch.cuda.is_available())"
PATH="$REPO/.venv-vllm/bin:$PATH" .venv-vllm/bin/python -c "import torch,vllm; print('vllm venv OK   | vllm',vllm.__version__,'torch',torch.__version__,'cuda',torch.cuda.is_available())"

echo "### setup complete."
