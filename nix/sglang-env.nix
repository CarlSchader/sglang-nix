{
  nixpkgs,
  flake-utils,
  pyproject-nix,
  uv2nix,
  pyproject-build-systems,
  ...
}:
flake-utils.lib.eachSystem ["x86_64-linux" "aarch64-linux"] (system: let
  inherit (nixpkgs) lib;

  pkgs = import nixpkgs {
    inherit system;
    config.allowUnfree = true;
  };

  # Read the uv.lock pinning sglang 0.5.19 (and friends) from the workspace root.
  workspace = uv2nix.lib.workspace.loadWorkspace {
    workspaceRoot = ../.;
  };

  # One derivation per pinned wheel, using the hashes already in uv.lock.
  # "wheel" preference enforces the no-source-builds promise: a dependency
  # that has no prebuilt wheel fails the build loudly instead of compiling.
  # (cuda-tile is pulled from pypi.nvidia.com via [tool.uv.sources]; the PyPI
  # entry is only a download stub.)
  overlay = workspace.mkPyprojectOverlay {
    sourcePreference = "wheel";
  };

  # Extra fixes for the GPU wheels. The manylinux wheels in this lock are
  # self-contained except for libraries that are intentionally absent from
  # the venv at patch time:
  #   - libcuda.so.1 / libnvidia-ml.so.1: provided by the NVIDIA driver at
  #     runtime (/run/opengl-driver/lib on NixOS);
  #   - libtorch*.so / libc10*.so and the nvidia-* CUDA libraries: live in
  #     sibling site-packages and are loaded by python (`import torch`)
  #     before any extension that needs them.
  # autoPatchelf must not fail the build over these, so they are ignored for
  # every GPU wheel in the lock.
  gpuWheels = [
    "apache-tvm-ffi"
    "av"
    "cuda-bindings"
    "cuda-core"
    "cuda-pathfinder"
    "cuda-python"
    "cuda-tile"
    "cuda-toolkit"
    "decord2"
    "flash-attn-4"
    "flashinfer-python"
    "humming-kernels"
    "kernels"
    "kernels-data"
    "llguidance"
    "nccl4py"
    "nvidia-cublas"
    "nvidia-cuda-cccl"
    "nvidia-cuda-crt"
    "nvidia-cuda-cupti"
    "nvidia-cuda-nvcc"
    "nvidia-cuda-nvdisasm"
    "nvidia-cuda-nvrtc"
    "nvidia-cuda-runtime"
    "nvidia-cudnn-cu13"
    "nvidia-cudnn-frontend"
    "nvidia-cufft"
    "nvidia-cufile"
    "nvidia-curand"
    "nvidia-cusolver"
    "nvidia-cusparse"
    "nvidia-cusparselt-cu13"
    "nvidia-cutlass-dsl"
    "nvidia-cutlass-dsl-libs-base"
    "nvidia-cutlass-dsl-libs-core"
    "nvidia-cutlass-dsl-libs-cu12"
    "nvidia-cutlass-dsl-libs-cu13"
    "nvidia-mathdx"
    "nvidia-ml-py"
    "nvidia-nccl-cu13"
    "nvidia-nvjitlink"
    "nvidia-nvshmem-cu13"
    "nvidia-nvtx"
    "nvidia-nvvm"
    "quack-kernels"
    "sgl-deep-ep"
    "sgl-deep-gemm"
    "sglang"
    "sglang-kernel"
    "tilelang"
    "tokenspeed-mla"
    "tokenspeed-triton"
    "torch"
    "torch-c-dlpack-ext"
    "torch-memory-saver"
    "torchaudio"
    "torchcodec"
    "torchvision"
    "triton"
    "xgrammar"
  ];

  fixups = final: prev: let
    relaxed = lib.genAttrs (builtins.filter (n: builtins.hasAttr n prev) gpuWheels) (
      name:
        prev.${name}.overrideAttrs (old: {
          autoPatchelfIgnoreMissingDeps = true;
        })
    );

    # Some wheels accidentally ship a top-level site-packages/build_backend.py
    # (packaging junk), which collides during venv assembly.
    dropBuildBackend = drv:
      drv.overrideAttrs (old: {
        postInstall =
          (old.postInstall or "")
          + ''
            rm -f $out/lib/python*/site-packages/build_backend.py
          '';
      });
  in
    relaxed
    // lib.optionalAttrs (prev ? flashinfer-python) {
      flashinfer-python = dropBuildBackend relaxed.flashinfer-python;
    }
    // lib.optionalAttrs (prev ? torch-c-dlpack-ext) {
      torch-c-dlpack-ext = dropBuildBackend relaxed.torch-c-dlpack-ext;
    }
    // {
      # modelscope and its dependency modelscope-hub both install a
      # bin/modelscope console script with different contents; keep the
      # top-level package's one (pip would overwrite in that order too).
      modelscope-hub = prev.modelscope-hub.overrideAttrs (old: {
        postInstall =
          (old.postInstall or "")
          + ''
            rm -f $out/bin/modelscope $out/bin/ms
          '';
      });

      # uv2nix selects the pure-python soundfile wheel, which has no bundled
      # libsndfile and falls back to dlopen("libsndfile.so"); point it at the
      # nixpkgs library instead (transformers imports soundfile eagerly).
      soundfile = prev.soundfile.overrideAttrs (old: {
        postInstall =
          (old.postInstall or "")
          + ''
            substituteInPlace $out/lib/python*/site-packages/soundfile.py \
              --replace-fail "_explicit_libname = 'libsndfile.so'" \
                             "_explicit_libname = '${lib.getLib pkgs.libsndfile}/lib/libsndfile.so'"
          '';
      });

      # numba's wheel links against oneTBB (libtbb.so.12).
      numba = prev.numba.overrideAttrs (old: {
        buildInputs = (old.buildInputs or []) ++ [pkgs.tbb];
      });
    };

  pythonSet =
    (pkgs.callPackage pyproject-nix.build.packages {
      python = pkgs.python312;
    }).overrideScope (lib.composeManyExtensions [
      pyproject-build-systems.overlays.default
      overlay
      fixups
    ]);

  sglangEnv = pythonSet.mkVirtualEnv "sglang-env" workspace.deps.default;

  # Pure-eval smoke test of the NixOS module: a full nixosSystem evaluation
  # with the module enabled that materialises the rendered ExecStart.
  # No VM, no GPU.
  moduleEval = nixpkgs.lib.nixosSystem {
    inherit system;
    modules = [
      ../modules/sglang.nix
      {
        nixpkgs.hostPlatform = system;
        nixpkgs.config.allowUnfree = true;
        boot.loader.grub.enable = false;
        fileSystems."/" = {
          device = "none";
          fsType = "tmpfs";
        };
        system.stateVersion = "25.05";

        services.sglang = {
          enable = true;
          package = sglangEnv;
          openFirewall = true;
          model = {
            hfId = "Inferact/Qwen3.8-27B-NVFP4";
            servedModelName = "Qwen3.8-27B-NVFP4";
            contextLength = 262144;
          };
          kvCacheDtype = "fp8_e4m3";
          attentionBackend = "triton";
          ui = {
            enable = true;
            webSearch.enable = true;
          };
        };
      }
    ];
  };
in {
  # Reproducible python env for `sglang serve` built from the pinned wheels.
  # Exposes $out/bin/sglang, $out/bin/hf, $out/bin/python.
  packages.sglangEnv = sglangEnv;
  packages.default = sglangEnv;

  checks = {
    # Prove the env is a coherent, importable python environment (no GPU).
    sglangEnvImport = pkgs.stdenv.mkDerivation {
      name = "sglang-env-import-check";
      nativeBuildInputs = [sglangEnv];
      dontUnpack = true;
      dontConfigure = true;
      buildPhase = ''
        runHook preBuild
        # `import sglang` eagerly creates cache dirs under $HOME.
        export HOME=$(mktemp -d)
        python - <<'EOF'
        import torch
        import sglang
        print("sglang", sglang.__version__)
        print("torch", torch.__version__)
        assert sglang.__version__ == "0.5.19", sglang.__version__
        EOF
        hf --help > /dev/null
        runHook postBuild
      '';
      installPhase = ''
        touch $out
      '';
    };

    # Evaluate the NixOS module against a minimal nixosSystem and pin down
    # the rendered unit. Fails at eval time if the module or its interplay
    # with services.open-webui regresses.
    sglangModuleEval = pkgs.writeText "sglang-module-eval" (builtins.toJSON {
      execStart = moduleEval.config.systemd.services.sglang.serviceConfig.ExecStart;
      openWebuiEnabled = moduleEval.config.services.open-webui.enable;
      firewallPorts = moduleEval.config.networking.firewall.allowedTCPPorts;
    });
  };
})
