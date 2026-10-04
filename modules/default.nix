{...}: {
  nixosModules = rec {
    sglang = import ./sglang.nix;
    default = sglang;
    # Opinionated preset (import alongside `sglang`): Qwen3.8-27B NVFP4 +
    # DFlash2 on a DGX Spark, after hasso5703/dgx-spark-qwen38.
    dgx-spark-qwen38 = import ./dgx-spark-qwen38.nix;
    # Image generation: stable-diffusion.cpp `sd-server` as a systemd
    # service (OpenAI-compatible /v1/images API).
    sd-cpp = import ./sd-cpp.nix;
    # Preset for sd-cpp: Qwen-Image-2.1 Uncensored GGUF
    # (abenzerps/Qwen-Image-2.1-Uncensored-GGUF), import alongside `sd-cpp`.
    "qwen-image-2.1-uc" = import ./qwen-image-2.1-uc.nix;
  };
}
