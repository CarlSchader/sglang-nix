# Preset: Qwen3.8-27B NVFP4 + DFlash2 speculative decoding on a DGX Spark (GB10).
#
# This is the "stock" 27B target of https://github.com/hasso5703/dgx-spark-qwen38
# (measured there at ~50 tok/s greedy median single-stream, 135-148 tok/s
# aggregate at 8 streams), reproduced natively from the pinned sglang env
# instead of their Docker image + patch overlay: sglang 0.5.20 already carries
# DFlash v2 and the mrope fix (sglang#34446) their overlay existed for.
#
# Everything is mkDefault, so any option can still be overridden. Import next
# to nixosModules.sglang and set `services.sglang.enable` + `package`.
{
  config,
  lib,
  ...
}: let
  cfg = config.services.sglang;
in {
  config = lib.mkIf cfg.enable {
    services.sglang = {
      model = {
        hfId = lib.mkDefault "RadixArk/Qwen3.8-27B-NVFP4";
        revision = lib.mkDefault "52d1adc5f38aa5ebf099c29ed7025ba34cfbb854";
        servedModelName = lib.mkDefault "qwen3.8-27b";
        # Model-native window; sglang reads it from the config when not
        # pinned lower, and the recipe leaves it at 262144.
        contextLength = lib.mkDefault 262144;
      };

      # The GB10 unified-memory trap: sglang's accounting misses 25-40 GB of
      # transient allocations (fp8 autotuner, CUDA-graph capture). 0.50 is
      # plenty for 262K context at batch <= 4 and keeps the host alive.
      memFractionStatic = lib.mkDefault "0.50";
      memoryMax = lib.mkDefault "100G";
      maxRunningRequests = lib.mkDefault 8;
      # Prefill runs ~1K tok/s on the GB10, so an 8192 chunk is ~8s of
      # wall-clock during which (without mixed chunk) no decode step runs and
      # every other stream freezes. 4096 halves the per-step stall.
      chunkedPrefillSize = lib.mkDefault 4096;
      # The GB10 has 20 CPU cores and the GPU loop pins itself to one; by
      # default the *whole* CPU side (HTTP, tokenization, streaming,
      # reasoning/tool parsers) also runs in one single-threaded process
      # (sglang forces TOKENIZERS_PARALLELISM=false and encodes inline in the
      # event loop). Tokenizing a 150K-200K prompt there takes seconds, during
      # which every other agent's stream stops receiving output. Fan the CPU
      # side out over several worker processes: with 8 running requests, 4
      # tokenizer workers keep one long prompt from stalling the rest.
      # Cost: each worker is a python process importing torch (a few GB host
      # RSS total), covered by memoryMax below.
      tokenizerWorkers = lib.mkDefault 4;
      detokenizerWorkers = lib.mkDefault 2;
      attentionBackend = lib.mkDefault "flashinfer";
      kvCacheDtype = lib.mkDefault null; # NVFP4 checkpoint ships KV scales
      trustRemoteCode = lib.mkDefault true;
      reasoningParser = lib.mkDefault "qwen3";
      toolCallParser = lib.mkDefault "qwen3_coder";

      speculative = {
        algorithm = lib.mkDefault "DFLASH";
        draftModelPath = lib.mkDefault "z-lab/Qwen3.8-27B-DFlash2";
        draftModelRevision = lib.mkDefault "50307d4c4cde6860d4eee73e2547cd786fe8e8a4";
        numDraftTokens = lib.mkDefault 8;
        draftModelQuantization = lib.mkDefault "unquant";
      };

      torchCompile = {
        enable = lib.mkDefault true;
        maxBs = lib.mkDefault 4;
      };

      extraArgs = lib.mkDefault [
        # Interleave decode steps with another request's prefill chunks.
        # Default prefill-first scheduling pauses all running streams until
        # the new prompt is fully prefilled: measured 20s-280s stalls for
        # 100K-200K prompts. DFLASH is in spec_info.supports_mixed_chunk, so
        # this is not silently disabled.
        "--enable-mixed-chunk"
        "--disable-prefill-cuda-graph"
        # Newer sglang split --cuda-graph-max-bs into -decode/-prefill; the
        # old name is now an ambiguous argparse prefix and aborts startup.
        "--cuda-graph-max-bs-decode"
        "8"
        # Deterministic kernels: reproducible tok/s across boots, and the
        # autotuner is one of the untracked memory bursts above.
        "--disable-flashinfer-autotune"
        # Hybrid GDN model: the mamba state pool must admit max-running-requests
        # x slots, or the scheduler silently caps concurrency.
        "--mamba-radix-cache-strategy"
        "extra_buffer"
        "--mamba-ssm-dtype"
        "bfloat16"
        "--max-mamba-cache-size"
        "96"
        "--num-continuous-decode-steps"
        "2"
        "--sleep-on-idle"
      ];
    };
  };
}
