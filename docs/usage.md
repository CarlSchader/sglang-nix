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

### DGX Spark — Qwen3.8-27B NVFP4, 256k context, FP8 KV cache

```nix
services.sglang = {
  enable = true;
  package = sglang-nix.packages.${pkgs.system}.sglangEnv;
  model = {
    hfId = "Inferact/Qwen3.8-27B-NVFP4";
    contextLength = 262144;
  };
  kvCacheDtype = "fp8_e4m3";
  attentionBackend = "triton";
  memFractionStatic = "0.85";
};
```

MTP speculative decoding can be added via `extraArgs`:

```nix
extraArgs = [
  "--speculative-algorithm" "NEXTN"
  "--speculative-num-steps" "3"
  "--speculative-eagle-topk" "1"
  "--speculative-num-draft-tokens" "4"
];
```

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
  `/var/lib/sglang/huggingface`. First start downloads the weights
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
