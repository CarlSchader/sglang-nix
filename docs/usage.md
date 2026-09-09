# Using sglang-nix on a NixOS host

`sglang-nix` packages a pinned SGLang 0.5.19 environment (built from `uv.lock`
with uv2nix — prebuilt wheels only, no source builds) and a NixOS module that
runs it as a hardened, auto-restarting systemd service, optionally fronted by
Open WebUI.

Requirements: an `x86_64-linux` or `aarch64-linux` NixOS host with a
CUDA-capable NVIDIA GPU managed by nixpkgs (`hardware.nvidia`, driver libs on
`/run/opengl-driver/lib`).

## Quick start

```nix
{
  inputs.sglang-nix.url = "github:carlschader/sglang-nix";

  outputs = { nixpkgs, sglang-nix, ... }: {
    nixosConfigurations.spark = nixpkgs.lib.nixosSystem {
      system = "aarch64-linux";
      modules = [
        sglang-nix.nixosModules.sglang
        ({ pkgs, ... }: {
          nixpkgs.config.allowUnfree = true;

          services.sglang = {
            enable = true;
            package = sglang-nix.packages.${pkgs.system}.sglangEnv;
            model = {
              hfId = "Inferact/Qwen3.8-27B-NVFP4";
              servedModelName = "Qwen3.8-27B-NVFP4";
              contextLength = 262144;
            };
            kvCacheDtype = "fp8_e4m3";
            attentionBackend = "triton";
            ui.enable = true;
          };
        })
      ];
    };
  };
}
```

The API serves OpenAI-compatible endpoints on `0.0.0.0:30000`; Open WebUI
listens on `127.0.0.1:8080`.

## Presets

### DGX Spark — Qwen3.8-27B NVFP4 + DFlash2 (`nixosModules.dgx-spark-qwen38`)

The fast path from [hasso5703/dgx-spark-qwen38](https://github.com/hasso5703/dgx-spark-qwen38)
(SGLang + NVFP4 + DFlash2 speculative decoding, deterministic kernels),
reproduced natively — no Docker, no patch overlay: sglang 0.5.19 already
carries DFlash v2 and the mrope fix (sglang#34446) their overlay existed for.

```nix
imports = [
  sglang-nix.nixosModules.sglang
  sglang-nix.nixosModules.dgx-spark-qwen38
];

services.sglang = {
  enable = true;
  package = sglang-nix.packages.${pkgs.system}.sglangEnv;
  openFirewall = true;
  ui.enable = true;
};
```

Everything the preset sets is `mkDefault`, so any option can be overridden.
It renders to:

```
sglang serve --model-path RadixArk/Qwen3.8-27B-NVFP4 --revision 52d1adc5…
  --served-model-name qwen3.8-27b --context-length 262144 --mem-fraction-static 0.50
  --attention-backend flashinfer --chunked-prefill-size 8192
  --speculative-algorithm DFLASH --speculative-draft-model-path z-lab/Qwen3.8-27B-DFlash2
  --speculative-draft-model-revision 50307d4c… --speculative-num-draft-tokens 8
  --speculative-draft-model-quantization unquant
  --enable-torch-compile --torch-compile-max-bs 4 --max-running-requests 8
  --disable-prefill-cuda-graph --cuda-graph-max-bs 8 --disable-flashinfer-autotune
  --mamba-radix-cache-strategy extra_buffer --mamba-ssm-dtype bfloat16 --max-mamba-cache-size 96
  --num-continuous-decode-steps 2 --sleep-on-idle --trust-remote-code
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder
```

plus `MemoryMax=100G` on the unit (their Docker `--memory 100g` analogue).

Measured on a DGX Spark with this exact config (`./bench.sh`, greedy,
thinking on, 800 output tokens, streaming decode rate net of TTFT):

| Workload | tok/s (single stream) |
| --- | --- |
| Math / structured reasoning | 55-61 |
| Code (write / refactor) | 31-35 |
| Short story | 31-34 |
| Technical explanation | 23-25 |
| **8 concurrent streams, aggregate** | **155** (19/stream) |

That matches the upstream repo's own battery (code 32-40, math 41-44,
prose 22, 135-148 aggregate at 8). Speculative decoding accepts *predictable*
tokens, so speed depends on what is generated; there is no single number.
KV pool on this boot: ~400K tokens (above the 262K window). First boot is
~9 min (torch.compile + CUDA graph capture; the inductor/triton caches live
in `/var/lib/sglang/.cache` and are reused), plus the ~25 GB download.

**Why `memFractionStatic = "0.50"`**: sglang's accounting does not see
25-40 GB of transient allocations on GB10 unified memory (autotuner, graph
capture). Higher fractions have frozen hosts hard. 0.50 leaves ~50 GB of host
RAM available while serving; treat anything past 0.70 as unsafe.

Not included from upstream: their patched chat template (reasoning-effort
tiers, mid-conversation system messages → `<system-reminder>`) — pass your
own via `model.chatTemplate` — their keepalive proxy for agent CLIs, and the
`--api-key` (the module relies on the firewall instead).

## Options

| Option | Default | Description |
| --- | --- | --- |
| `enable` | `false` | Enable the SGLang service. |
| `package` | — (required) | Python env providing `bin/sglang`; use `sglang-nix.packages.<system>.sglangEnv`. |
| `host` / `port` | `"0.0.0.0"` / `30000` | API listen address. |
| `openFirewall` | `false` | Open the API port. |
| `model.hfId` | — (required) | Hugging Face model id / local path (`--model-path`). |
| `model.servedModelName` | basename of `hfId` | Name exposed on the API. |
| `model.contextLength` | `32768` | Context window (`--context-length`). |
| `model.revision` | `null` | Pin the HF revision (`--revision`). |
| `model.chatTemplate` | `null` | Chat template override (`--chat-template`). |
| `speculative.algorithm` | `null` | `--speculative-algorithm` (`DFLASH`, `EAGLE3`, `NEXTN`, ...); `null` = off. |
| `speculative.draftModelPath` / `draftModelRevision` | `null` | Draft model for the speculative algorithm. |
| `speculative.numDraftTokens` / `numSteps` / `eagleTopk` | `null` | Speculative tuning knobs. |
| `speculative.draftModelQuantization` | `null` | e.g. `"unquant"`. |
| `torchCompile.enable` / `torchCompile.maxBs` | `false` / `null` | `--enable-torch-compile [--torch-compile-max-bs N]`. |
| `memoryMax` | `null` | systemd `MemoryMax=` for the unit, e.g. `"100G"`. |
| `maxRunningRequests` | `8` | `--max-running-requests` (`null` to omit). |
| `chunkedPrefillSize` | `null` | `--chunked-prefill-size` (`null` to omit, `-1` disables). |
| `memFractionStatic` | `"0.85"` | `--mem-fraction-static`. Keep conservative on unified-memory GPUs (DGX Spark): the desktop/driver hold several GiB at startup. |
| `tensorParallelSize` | `1` | `--tp-size`. |
| `kvCacheDtype` | `null` | e.g. `"fp8_e4m3"`, `"fp8_e5m2"`, `"nvfp4"`. |
| `attentionBackend` | `null` | e.g. `"triton"`, `"flashinfer"`, `"fa3"`, `"trtllm_mha"`. |
| `toolCallParser` | `"qwen3_coder"` | `--tool-call-parser` (`null` to omit). |
| `reasoningParser` | `"qwen3"` | `--reasoning-parser` (`null` to omit). |
| `enableMultimodal` | `false` | Pass `--enable-multimodal` (image/video inputs). |
| `trustRemoteCode` | `false` | Pass `--trust-remote-code`. |
| `extraArgs` | `[]` | Extra `sglang serve` arguments. |
| `environment` | `{}` | Extra env vars for the service. |
| `environmentFile` | `null` | Secrets file, e.g. `HF_TOKEN=...` for gated models. |
| `cudaToolkit.enable` | `true` | CUDA toolkit + gcc on the unit PATH for FlashInfer/DeepGEMM JIT. |
| `cudaToolkit.package` | `cudaPackages_13.cudatoolkit` | Toolkit used for JIT. |
| `ui.enable` | `false` | Enable Open WebUI wired to this SGLang. |
| `ui.package` | `pkgs.open-webui` w/ torchaudio tests off | Open WebUI package (see note below). |
| `ui.host` / `ui.port` | `"127.0.0.1"` / `8080` | UI listen address. |
| `ui.openFirewall` | `false` | Open the UI port. |
| `ui.webSearch.enable` | `false` | Enable web search (`duckduckgo` by default). |
| `ui.webSearch.engine` | `"duckduckgo"` | Search engine. |
| `ui.webSearch.resultCount` | `5` | Results per query. |
| `ui.webSearch.concurrentRequests` | `10` | Concurrent search requests. |
| `ui.environment` | `{}` | Extra env vars for Open WebUI (wins over module-set ones). |

## vLLM → SGLang flag cheat sheet

| vllm-nix option | sglang-nix option | flag |
| --- | --- | --- |
| `model.maxModelLen` | `model.contextLength` | `--context-length` |
| `gpuMemoryUtilization` | `memFractionStatic` | `--mem-fraction-static` |
| `maxNumSeqs` | `maxRunningRequests` | `--max-running-requests` |
| `maxNumBatchedTokens` | `chunkedPrefillSize` | `--chunked-prefill-size` |
| `tensorParallelSize` | `tensorParallelSize` | `--tp-size` |
| `kvCacheDtype = "fp8"` | `kvCacheDtype = "fp8_e4m3"` | `--kv-cache-dtype` |
| `attentionBackend = "TRITON_ATTN"` | `attentionBackend = "triton"` | `--attention-backend` |
| `enableAutoToolChoice` | (implicit when `toolCallParser` is set) | — |
| `disableMultimodalInputs = true` | `enableMultimodal = false` | `--enable-multimodal` is opt-in |

## Operational notes

- **State** lives in `/var/lib/sglang`; model weights are cached under
  `/var/lib/sglang/huggingface`, JIT/torch.compile caches under
  `/var/lib/sglang/.cache`. First start downloads the weights
  (`TimeoutStartSec` is 60 min for that reason).
- **Gated models**: put `HF_TOKEN=...` in a root-owned file and point
  `services.sglang.environmentFile` at it.
- **Restart behaviour**: `Restart=always`, `RestartSec=5`.
- **Logs**: `journalctl -u sglang -f` (and `journalctl -u open-webui -f`).
- **UI ↔ API wiring**: the module sets `OPENAI_API_BASE_URL` to
  `http://127.0.0.1:<port>/v1` for Open WebUI and disables the Ollama API.

## Updating SGLang

```console
uv lock --upgrade          # or edit pyproject.toml constraints first
git add uv.lock
nix flake check            # rebuilds the env, runs import + module checks
```

The Nix side needs no changes unless new wheels fail `autoPatchelf`
(add them to `gpuWheels` in `nix/sglang-env.nix`) or a wheel gains a new
system-library dependency (add it to that package's `buildInputs`, as done
for numba/oneTBB). Bump the version assertion in `checks.sglangEnvImport`.

## Troubleshooting

- `python3.14-torchaudio … FAILED test_batch_melspectrogram` while building
  `open-webui` — Open WebUI drags in nixpkgs' torch stack, which is not in
  the binary cache on aarch64 and gets built from source (~hours); torchaudio's
  tests then fail on a flaky tolerance check. `ui.package` defaults to an
  override with torchaudio tests disabled. If you would rather not build torch
  at all, run Open WebUI some other way and leave `ui.enable = false`.

- CUDA not found / no GPU detected — the process cannot see the NVIDIA
  driver. Check `hardware.nvidia` is configured and `/run/opengl-driver/lib`
  exists; the unit's `LD_LIBRARY_PATH` already points there.
- JIT errors mentioning `nvcc`/`gcc`/`ninja` — keep
  `services.sglang.cudaToolkit.enable = true` (default).
- `fatal error: cuda_runtime.h: No such file or directory` from sglang's
  kernel JIT — the module sets `NVCC_PREPEND_FLAGS=-I<toolkit>/include ...`
  for exactly this; if you override `environment`, don't drop it.
- CUDA OOM — lower `model.contextLength` or `memFractionStatic`, or use an
  FP8 KV cache (`kvCacheDtype = "fp8_e4m3"`).
