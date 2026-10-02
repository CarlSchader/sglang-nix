# sglang-nix

A pinned [SGLang](https://github.com/sgl-project/sglang) 0.5.19 environment
(uv.lock → Nix via [uv2nix](https://github.com/pyproject-nix/uv2nix), prebuilt
wheels only) plus a NixOS module that runs it as a hardened, auto-restarting
systemd service — optionally fronted by Open WebUI.

Sibling of [vllm-nix](https://github.com/carlschader/vllm-nix); same layout,
different engine. Supports `x86_64-linux` and `aarch64-linux` (DGX Spark).

```nix
imports = [
  sglang-nix.nixosModules.sglang
  sglang-nix.nixosModules.dgx-spark-qwen38   # optional preset, see below
];

services.sglang = {
  enable = true;
  package = sglang-nix.packages.${pkgs.system}.sglangEnv;
  ui.enable = true;
};
```

The `dgx-spark-qwen38` preset reproduces the fast Qwen3.8-27B config from
[hasso5703/dgx-spark-qwen38](https://github.com/hasso5703/dgx-spark-qwen38)
(NVFP4 + DFlash2 speculative decoding) natively, without Docker: measured
here at 55-61 tok/s on math, 31-35 on code single-stream and **155 tok/s
aggregate at 8 streams**. `./bench.sh` reproduces the measurement.

See [docs/usage.md](docs/usage.md) for the full option reference, the preset
details and numbers, update procedure, and troubleshooting.

## Flake outputs

- `packages.<system>.sglangEnv` (also `default`) — the pinned python env (`bin/sglang`, `bin/hf`, `bin/python`)
- `packages.<system>.sglang-watch` — terminal dashboard for a running server (Rust; see [Monitoring](#monitoring))
- `nixosModules.sglang` (also `default`) — the `services.sglang` module
- `nixosModules.dgx-spark-qwen38` — preset: Qwen3.8-27B NVFP4 + DFlash2 on a DGX Spark
- `checks.<system>.{sglangEnvImport,sglangModuleEval}` — env import + module eval smoke tests
- `devShells.<system>.default` — uv/CUDA dev shell for working on the lock file

## Monitoring

`sglang-watch` is a terminal dashboard for a running server (throughput,
prefill load, KV cache, spec-decode accept, clients):
`nix run .#sglang-watch` — details in [docs/usage.md](docs/usage.md#monitoring).

## Development

```console
nix build .#sglangEnv   # build the environment
nix flake check         # import + module-eval checks
```
