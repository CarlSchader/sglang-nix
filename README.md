# sglang-nix

A pinned [SGLang](https://github.com/sgl-project/sglang) 0.5.19 environment
(uv.lock → Nix via [uv2nix](https://github.com/pyproject-nix/uv2nix), prebuilt
wheels only) plus a NixOS module that runs it as a hardened, auto-restarting
systemd service — optionally fronted by Open WebUI.

Sibling of [vllm-nix](https://github.com/carlschader/vllm-nix); same layout,
different engine. Supports `x86_64-linux` and `aarch64-linux` (DGX Spark).

```nix
imports = [ sglang-nix.nixosModules.sglang ];

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
```

See [docs/usage.md](docs/usage.md) for the full option reference, the DGX
Spark preset, update procedure, and troubleshooting.

## Flake outputs

- `packages.<system>.sglangEnv` (also `default`) — the pinned python env (`bin/sglang`, `bin/hf`, `bin/python`)
- `nixosModules.sglang` (also `default`) — the `services.sglang` module
- `checks.<system>.{sglangEnvImport,sglangModuleEval}` — env import + module eval smoke tests
- `devShells.<system>.default` — uv/CUDA dev shell for working on the lock file

## Development

```console
nix build .#sglangEnv   # build the environment
nix flake check         # import + module-eval checks
```
