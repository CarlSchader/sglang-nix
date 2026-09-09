#!/usr/bin/env bash
# DGX Spark (GB10, unified memory) recipe — sglang equivalent of the vllm-nix one.

uv run sglang serve \
  --model-path Inferact/Qwen3.8-27B-NVFP4 \
  --served-model-name Qwen3.8-27B-NVFP4 \
  --tp-size 1 \
  --context-length 262144 \
  --mem-fraction-static 0.85 \
  --attention-backend triton \
  --kv-cache-dtype fp8_e4m3 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --host 0.0.0.0 --port 30000
# Optional MTP speculative decoding (Qwen3.8 ships an MTP head):
#   --speculative-algorithm NEXTN --speculative-num-steps 3 \
#   --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
