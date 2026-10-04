# Using sglang-nix on a NixOS host

`sglang-nix` packages a pinned SGLang 0.5.20 environment (built from `uv.lock`
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
reproduced natively — no Docker, no patch overlay: sglang 0.5.20 already
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
  --attention-backend flashinfer --chunked-prefill-size 4096
  --tokenizer-worker-num 4 --detokenizer-worker-num 2
  --speculative-algorithm DFLASH --speculative-draft-model-path z-lab/Qwen3.8-27B-DFlash2
  --speculative-draft-model-revision 50307d4c… --speculative-num-draft-tokens 8
  --speculative-draft-model-quantization unquant
  --enable-torch-compile --torch-compile-max-bs 4 --max-running-requests 8
  --enable-mixed-chunk --disable-prefill-cuda-graph --cuda-graph-max-bs-decode 8 --disable-flashinfer-autotune
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

## Image generation — `services.sd-cpp` (stable-diffusion.cpp)

`packages.<system>.sdcpp` is a pinned CUDA build of
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (the
ggml-based diffusion engine that llama.cpp's author also maintains, and the
tool the Qwen-Image GGUF weights are converted with). `sd-server` ships an
HTTP API with an embedded UI; this module runs it as a hardened systemd
service. It is model-agnostic: any SD1.5/SDXL/SD3/Flux/Qwen-Image family
weights (GGUF or safetensors) work — the preset below pins
[Qwen-Image-2.1 Uncensored GGUF](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF),
whose model card targets ComfyUI but which sd.cpp loads natively, so no
ComfyUI is required.

### Quick start (Qwen-Image-2.1 UC preset)

```nix
imports = [
  sglang-nix.nixosModules.sd-cpp
  sglang-nix.nixosModules."qwen-image-2.1-uc"
];

services.sd-cpp = {
  enable = true;
  package = sglang-nix.packages.${pkgs.system}.sdcpp;
  openFirewall = true;
};
```

The first `nixos-rebuild` fetches ~14.6 GB of SHA256-pinned weights into the
Nix store (4.6 GB `Q4_K_M` diffusion GGUF + 9.4 GB int8 Qwen3-VL-8B text
encoder + 0.7 GB bf16 VAE, per the model card's recommended memory layout:
diffusion in VRAM-speed weights, encoder offloaded to RAM once per prompt).
Everything the preset sets is `mkDefault`; override any of
`diffusionModel`/`vae`/`textEncoder` to point at other files (e.g. a
Q6_K/Q8_0 quantization, or a local copy).

### API

All of these are served on one port (default 1234):

| Family | Endpoints |
| --- | --- |
| OpenAI-compatible | `POST /v1/images/generations`, `POST /v1/images/edits`, `GET /v1/models` |
| Native (async jobs) | `POST /sdcpp/v1/img_gen` → job id, `GET /sdcpp/v1/jobs/<id>`, `POST /sdcpp/v1/jobs/<id>/cancel`, `GET /sdcpp/v1/capabilities` |
| Stable Diffusion WebUI style | `POST /sdapi/v1/txt2img`, `POST /sdapi/v1/img2img`, `GET /sdapi/v1/{samplers,schedulers,sd-models,options}` |

OpenAI example (verified):

```console
$ curl -s -X POST http://<host>:1234/v1/images/generations \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen-image-2.1","prompt":"a lighthouse on a cliff at dusk","
        "n":1,"size":"1024x1024"}' | jq -r '.data[0].b64_json' | base64 -d > img.png
```

Responses carry the image as `data[].b64_json` (PNG unless the request asks
for jpeg/webp). Generation defaults come from the CLI flags the module passes
(`--cfg-scale 6.0 --sampling-method euler` in the preset); requests can
override steps/seed/size/CFG per call. `POST /v1/images/edits` (multipart,
`image` + `prompt`) needs a vision projector: set
`services.sd-cpp.llmVision` to a Qwen3-VL `mmproj` GGUF (not included in the
preset; the base Qwen-Image-2.1 release doesn't ship one in this repo).

### Measured on this machine (RTX 4090, 24 GB)

Preset weights, 20 steps (default), `--offload-to-cpu` + flash attention:

| Output | Wall time | Breakdown |
| --- | --- | --- |
| 512×512 | ~11 s | condition 3.4 s + sampling 6.6 s + VAE 0.7 s |
| 1024×1024 | ~30 s | condition 3.6 s + sampling 25.1 s + VAE 1.2 s |

Model weights total ~13.4 GB in system RAM (encoder ~8.3 GB, diffusion
~4.4 GB, VAE ~0.6 GB); the 4090 holds activations/flash-attention workspaces
in VRAM. A 16 GB host needs ~30 GB RAM for this quantization set; smaller
cards can drop `offloadToCpu` for a VRAM-resident encoder at the cost of
~9 GB VRAM.

### Options

| Option | Default | Description |
| --- | --- | --- |
| `enable` | `false` | Enable the sd-cpp service. |
| `package` | — (required) | Package with `bin/sd-server`; use `sglang-nix.packages.<system>.sdcpp`. |
| `host` / `port` | `"0.0.0.0"` / `1234` | API listen address. |
| `openFirewall` | `false` | Open the API port. |
| `diffusionModel` | — (required) | Diffusion/transformer weights (`.gguf`/`.safetensors`/`.ckpt`). |
| `vae` | — (required) | VAE decoder weights; must match the model family. |
| `textEncoder` | `null` | `--llm` text encoder (GGUF or safetensors; Qwen-Image-2.1 needs it). |
| `clip` | `null` | `--clip` CLIP encoder (SD1.5/SDXL-style models). |
| `llmVision` | `null` | `--llm_vision` vision projector; enables image editing. |
| `offloadToCpu` | `true` | `--offload-to-cpu`: keep weights in RAM, VRAM for activations. |
| `flashAttention` | `true` | `--diffusion-fa`. |
| `vaeTiling` | `false` | `--vae-tiling` (peak-VRAM relief at high resolutions). |
| `cfgScale` | `null` | default `--cfg-scale` (string). |
| `samplingMethod` | `null` | default `--sampling-method` (`euler`, `euler_a`, `dpmpp_2m`, ...). |
| `numThreads` | `null` | CPU thread count (`--threads`; default: physical core count). |
| `memoryMax` | `null` | systemd `MemoryMax=` (runaway offload guard). |
| `cudaToolkit` | `pkgs.cudaPackages_13.cudatoolkit` | toolkit lib dir for `LD_LIBRARY_PATH` (plus the nixpkgs driver). |
| `extraArgs` | `[]` | extra `sd-server` args (e.g. `--main-gpu`, `--seed`, `--vae-tiling`). |

The unit runs as the `sd-cpp` system user with GPU `DeviceAllow`,
`ProtectSystem=strict` and `Restart=always`; model files live in the Nix
store, so nothing is written at boot and restarts are fast.

### Updating the pin

Move `rev` in `nix/sd-cpp.nix` and refresh the tree hash (submodules
included):

```console
nix run nixpkgs#nix-prefetch-git -- --fetch-submodules \
  --url https://github.com/leejet/stable-diffusion.cpp --rev <sha>
```

CUDA kernel architectures are baked in at build time
(`mkSdcpp "89-real;90-virtual;120a-real;121a-real"`); for other GPUs,
`packages.<system>.sdcpp` can be re-derived from the same source via
`overrideAttrs`, or the flake's `mkSdcpp` called with a different arch list
(null falls back to ggml's portable default).

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
| `tokenizerWorkers` | `null` | `--tokenizer-worker-num`. >1 fans HTTP + tokenization + streaming out over N processes so one huge prompt does not stall every other stream. |
| `detokenizerWorkers` | `null` | `--detokenizer-worker-num` (pair with `tokenizerWorkers` > 1). |
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

## Monitoring

`sglang-watch` (flake package `packages.<system>.sglang-watch`) is a
terminal dashboard for a running server — safe to run on the serving host:

```console
nix run .#sglang-watch                   # live TUI, 1s refresh
nix run .#sglang-watch -- --once         # one plain-text snapshot (exit 2 if unreachable)
nix run .#sglang-watch -- --window 120 -i 2
```

The source lives in `tools/sglang-watch/` (Rust; deps: `serde_json`,
`libc`); `cargo build --release` there works too, and the unit tests are
run as part of the nix build (`doCheck`). It's a self-contained binary,
so — unlike the Linux-only CUDA env — it's packaged for every default
system (`eachDefaultSystem`): both Linux and macOS. The only OS-specific
panel is the local `ss` client probe, which just shows `clients —` on
macOS; all telemetry is HTTP.

### Remote use

All telemetry is read-only HTTP, so the dashboard can run on any machine
that can reach the serving port — it doesn't have to be the serving host:

```console
nix run .#sglang-watch -- --url http://<server-ip>:30000
```

(`services.sglang.host` defaults to `0.0.0.0`, so the server already
binds all interfaces; if the host firewall is on, allow the API port with
`services.sglang.openFirewall = true` — it defaults to `false`.) The one
caveat: the `clients` line counts connections *from the machine running
the dashboard* (via local `ss`), so pass `--no-clients` when watching
remotely. Server-side request counts (`running`, `queued`) come from
`/v1/loads` and are correct from anywhere.

Panels (from the always-on `/v1/loads` scheduler snapshot, no server flags
needed):

- **Throughput** — decode tok/s over a sliding window + sparkline
- **Prefill** — busy % of the window spent on prefills that interrupted the
  active decode loop (the thrashing signal: 0 % while idle or pure decode,
  climbs when big prompts keep cutting into in-flight streams); plus pending
  prefill tokens and uncached-prefill rate; `PREFILL-BUSY` chip at ≥ 60 %
- **Requests** — running/max-running, queued, retracted; `SATURATED` chip
  when the batch is full and requests queue
- **KV cache** — pool usage (red ≥ 90 %, `KV-PRESSURE` chip), used/total
  tokens, radix cache hit rate; `CACHE-MISS` chip when fresh-token prefill
  is fast and hit rate < 25 %
- **Spec decode** — accept length / rate (DFlash2, EAGLE, ...)
- **Latency** — mean decode step time (inter-token latency) and batch size
- **VRAM** — weights / KV cache / CUDA-graph breakdown
- **clients** — distinct peer hosts + established connections on the API
  port (via `ss`; skip with `--no-clients`)

If the server was started with `--enable-metrics` (not set by default in
this module — add it via `services.sglang.extraArgs`), the TUI also picks up
`/metrics` and shows TTFT and end-to-end latency p50/p95.

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
