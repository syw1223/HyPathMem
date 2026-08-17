#!/usr/bin/env bash
set -euo pipefail

cd /home/sunyuwei/HyTopoMem
mkdir -p outputs/logs

export CUDA_VISIBLE_DEVICES=6
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

exec /home/sunyuwei/miniconda3/envs/python311/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/sunyuwei/LightMem/models/Qwen3-30B-A3B-Instruct-2507 \
  --host 127.0.0.1 \
  --port 8006 \
  --served-model-name qwen3-30b-a3b-instruct-2507 \
  --api-key EMPTY \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --generation-config vllm \
  --trust-remote-code
