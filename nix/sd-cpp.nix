# Pinned stable-diffusion.cpp ("sd.cpp") build: CUDA backend, `sd-cli` and
# the `sd-server` HTTP API (OpenAI-compatible /v1/images/..., /sdapi/v1/...,
# /sdcpp/v1/..., embedded web UI).
#
# sd.cpp is what the Qwen-Image-2.1 GGUF weights are built for (it added
# day-0 Qwen-Image-2.1 support and ships the text-encoder/VAE loaders this
# repo's models need). All third-party deps (httplib, stb, oniguruma,
# utf8proc, miniz) are vendored in-tree; the only submodules are the patched
# ggml and the optional web-frontend (disabled here, API-only build).
#
# Updating: move the rev, refresh the hash with
#   nix run nixpkgs#nix-prefetch-git -- --fetch-submodules \
#     --url https://github.com/leejet/stable-diffusion.cpp --rev <sha>
# and keep the ggml submodule pointer in lockstep (it is part of the tree).
{
  nixpkgs,
  flake-utils,
  ...
}:
flake-utils.lib.eachSystem ["x86_64-linux" "aarch64-linux"] (system: let
  inherit (nixpkgs) lib;

  pkgs = import nixpkgs {
    inherit system;
    config.allowUnfree = true;
  };

  src = pkgs.fetchFromGitHub {
    owner = "leejet";
    repo = "stable-diffusion.cpp";
    rev = "3f8527a46c54ecf4cb4ed6003da8e8982283c73c";
    fetchSubmodules = true;
    hash = "sha256-AMWPF0nPpU92MuTHQo9QUJQm5IvVLPEPDOR/Wrgziq4=";
  };

  mkSdcpp = cudaArchs: pkgs.stdenv.mkDerivation {
    pname = "stable-diffusion-cpp";
    version = "3f8527a";
    inherit src;

    nativeBuildInputs = [
      pkgs.cmake
      pkgs.ninja
    ];

    # CUDAToolkit cmake package + cudart/cublas at link time; nvcc for the
    # ggml CUDA backend.
    buildInputs = [
      pkgs.cudaPackages_13.cudatoolkit
      pkgs.cudaPackages_13.cuda_nvcc
    ];

    cmakeFlags = [
      "-DSD_CUDA=ON"
      "-DSD_WEBP=OFF"
      "-DSD_WEBM=OFF"
      "-DSD_BUILD_EXAMPLES=ON"
      # The web UI is a pnpm build of a separate repo (leejet/sdcpp-webui);
      # the HTTP API is complete without it. Re-enable with nodejs + pnpm on
      # PATH if the embedded UI is wanted.
      "-DSD_SERVER_BUILD_FRONTEND=OFF"
      "-DCMAKE_CUDA_COMPILER=${pkgs.cudaPackages_13.cuda_nvcc}/bin/nvcc"
    ]
    ++ lib.optionals (cudaArchs != null) [
      "-DCMAKE_CUDA_ARCHITECTURES=${cudaArchs}"
    ];

    meta = with pkgs.lib; {
      description = "Diffusion-model inference engine (SD/Flux/Qwen-Image/...) in C/C++, with CUDA backend and HTTP server";
      homepage = "https://github.com/leejet/stable-diffusion.cpp";
      license = licenses.mit;
      platforms = platforms.linux;
      mainPrograms = [ "sd-cli" "sd-server" ];
    };
  };

  # Default build: RTX 40 (89), Hopper via PTX JIT (90-virtual), RTX 50
  # (120a) and GB10/DGX Spark (121a). Callers wanting other architectures:
  # `packages.sdcpp.overrideAttrs (old: { cmakeFlags = ...; })` or reuse
  # `mkSdcpp` with a different arch string (null -> ggml's portable list).
  sdcpp = mkSdcpp "89-real;90-virtual;120a-real;121a-real";

  moduleEval = nixpkgs.lib.nixosSystem {
    inherit system;
    modules = [
      ../modules/sd-cpp.nix
      ../modules/qwen-image-2.1-uc.nix
      {
        nixpkgs.hostPlatform = system;
        nixpkgs.config.allowUnfree = true;
        boot.loader.grub.enable = false;
        fileSystems."/" = {
          device = "none";
          fsType = "tmpfs";
        };
        system.stateVersion = "25.05";

        services.sd-cpp = {
          enable = true;
          package = sdcpp;
          openFirewall = true;
        };
      }
    ];
  };
in {
  # Binaries: $out/bin/sd-cli, $out/bin/sd-server.
  packages.sdcpp = sdcpp;

  checks = {
    # The pinned revision must build and the binaries must run.
    sdcppVersion = pkgs.stdenv.mkDerivation {
      name = "sdcpp-version-check";
      nativeBuildInputs = [sdcpp];
      dontUnpack = true;
      dontConfigure = true;
      buildPhase = ''
        runHook preBuild
        sd-cli --version
        sd-server --version
        runHook postBuild
      '';
      installPhase = ''
        touch $out
      '';
    };

    # Pure-eval smoke test of the NixOS module (with the Qwen-Image-2.1 UC
    # preset), mirroring sglangModuleEval: pins the rendered unit.
    sdcppModuleEval = pkgs.writeText "sdcpp-module-eval" (builtins.toJSON {
      execStart = moduleEval.config.systemd.services.sd-cpp.serviceConfig.ExecStart;
      user = moduleEval.config.systemd.services.sd-cpp.serviceConfig.User;
      firewallPorts = moduleEval.config.networking.firewall.allowedTCPPorts;
    });
  };
})
