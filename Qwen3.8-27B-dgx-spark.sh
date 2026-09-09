#!/usr/bin/env bash
# DGX Spark (GB10) recipe: Qwen3.8-27B NVFP4 + DFlash2 speculative decoding.
# Same flags as modules/dgx-spark-qwen38.nix (the NixOS preset); this is the
# foreground / A-B version. Measured here: 25-61 tok/s single stream depending
# on content (math ~55-60, code ~32-35, prose ~25-32), ~155 tok/s aggregate at
# 8 streams. First boot ~9 min (torch.compile + CUDA graph capture, cached).
#
# Use from `nix develop` (sets CUDA/nvcc/LD_LIBRARY_PATH) with `nix build` done:
#   ./result/bin/sglang serve ...   or   uv run sglang serve ...

export TORCHINDUCTOR_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/inductor"

uv run sglang serve \
  --model-path RadixArk/Qwen3.8-27B-NVFP4 \
  --revision 52d1adc5f38aa5ebf099c29ed7025ba34cfbb854 \
  --served-model-name qwen3.8-27b \
  --trust-remote-code --tp-size 1 \
  --context-length 262144 \
  --mem-fraction-static 0.50 \
  --attention-backend flashinfer --chunked-prefill-size 8192 \
  --disable-prefill-cuda-graph --cuda-graph-max-bs 8 \
  --disable-flashinfer-autotune \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3.8-27B-DFlash2 \
  --speculative-draft-model-revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 \
  --speculative-num-draft-tokens 8 --speculative-draft-model-quantization unquant \
  --mamba-radix-cache-strategy extra_buffer --mamba-ssm-dtype bfloat16 \
  --max-mamba-cache-size 96 --max-running-requests 8 \
  --enable-torch-compile --torch-compile-max-bs 4 \
  --num-continuous-decode-steps 2 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --sleep-on-idle \
  --host 0.0.0.0 --port 30000
