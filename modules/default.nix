{...}: {
  nixosModules = rec {
    sglang = import ./sglang.nix;
    default = sglang;
    # Opinionated preset (import alongside `sglang`): Qwen3.8-27B NVFP4 +
    # DFlash2 on a DGX Spark, after hasso5703/dgx-spark-qwen38.
    dgx-spark-qwen38 = import ./dgx-spark-qwen38.nix;
  };
}
