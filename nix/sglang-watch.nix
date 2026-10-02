{
  nixpkgs,
  flake-utils,
  ...
}:
flake-utils.lib.eachSystem ["x86_64-linux" "aarch64-linux"] (system:
  let
    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
    };
  in
  {
    # Terminal dashboard for a running SGLang server (docs/usage.md,
    # "Monitoring"). Small pure-Rust binary: std + serde_json + libc, with
    # the crates fetched from the crates.io CDN pinned by the committed
    # tools/sglang-watch/Cargo.lock.
    packages.sglang-watch = pkgs.rustPlatform.buildRustPackage {
      pname = "sglang-watch";
      version = "0.1.0";
      # cleanSource respects .gitignore, so the gitignored cargo `target/`
      # build dir is excluded without needing an explicit filter.
      src = pkgs.lib.cleanSource ../tools/sglang-watch;
      # Fetch the exact crates pinned in the committed lock file.
      cargoLock = {
        lockFile = ../tools/sglang-watch/Cargo.lock;
      };

      # Run the unit tests (prom parser, quantiles, sliding-window rates,
      # URL parsing) as part of the build.
      doCheck = true;

      meta = {
        description = "Terminal dashboard for a running SGLang server";
        homepage = "https://github.com/CarlSchader/sglang-nix";
        mainProgram = "sglang-watch";
        platforms = ["x86_64-linux" "aarch64-linux"];
      };
    };
  }
)
