# NixOS module: services.sd-cpp
#
# Runs stable-diffusion.cpp's `sd-server` (from
# sglang-nix.packages.<system>.sdcpp) as a hardened systemd service. It serves
# an OpenAI-compatible image API on /v1/images/generations and
# /v1/images/edits, a ComfyUI-sdwebui-style API on /sdapi/v1/..., and the
# native async API on /sdcpp/v1/... (see examples/server/api.md upstream).
#
# The model is a standalone diffusion model + VAE + text encoder, e.g. the
# Qwen-Image-2.1 GGUF weights (see the qwen-image-2.1-uc preset module).
# Requires a CUDA-capable NVIDIA GPU managed by nixpkgs.
{
  config,
  lib,
  pkgs,
  ...
}: let
  cfg = config.services.sd-cpp;

  driverLib = "${pkgs.addDriverRunpath.driverLink}/lib";

  stateDir = "/var/lib/sd-cpp";

  serveArgs =
    [
      "--listen-ip"
      cfg.host
      "--listen-port"
      (toString cfg.port)
      "--diffusion-model"
      cfg.diffusionModel
      "--vae"
      cfg.vae
    ]
    ++ lib.optionals (cfg.textEncoder != null) ["--llm" cfg.textEncoder]
    ++ lib.optionals (cfg.clip != null) ["--clip" cfg.clip]
    ++ lib.optionals (cfg.llmVision != null) ["--llm_vision" cfg.llmVision]
    ++ lib.optionals cfg.offloadToCpu ["--offload-to-cpu"]
    ++ lib.optionals cfg.flashAttention ["--diffusion-fa"]
    ++ lib.optionals cfg.vaeTiling ["--vae-tiling"]
    ++ lib.optionals (cfg.cfgScale != null) ["--cfg-scale" cfg.cfgScale]
    ++ lib.optionals (cfg.samplingMethod != null) ["--sampling-method" cfg.samplingMethod]
    ++ lib.optionals (cfg.numThreads != null) ["--threads" (toString cfg.numThreads)]
    ++ cfg.extraArgs;
in {
  options.services.sd-cpp = {
    enable = lib.mkEnableOption "stable-diffusion.cpp image-generation server";

    package = lib.mkOption {
      type = lib.types.package;
      description = ''
        Package providing {file}`bin/sd-server`. Normally
        `sglang-nix.packages.''${pkgs.system}.sdcpp`.
      '';
    };

    host = lib.mkOption {
      type = lib.types.str;
      default = "0.0.0.0";
      description = "Address the HTTP API listens on.";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 1234;
      description = "Port the HTTP API listens on (sd-server's default is 1234).";
    };

    openFirewall = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Open the API port in the firewall.";
    };

    diffusionModel = lib.mkOption {
      type = lib.types.path;
      description = ''
        Diffusion/transformer weights: `.gguf`, `.safetensors` or `.ckpt`
        (e.g. a Qwen-Image-2.1 GGUF quantization).
      '';
    };

    vae = lib.mkOption {
      type = lib.types.path;
      description = "VAE decoder weights (`.safetensors`/`.sft`); must match the model family.";
    };

    textEncoder = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = ''
        Text encoder / language-model weights passed as `--llm`
        (`.gguf` or `.safetensors`; required by transformer models such as
        Qwen-Image-2.1, which uses Qwen3-VL-8B).
      '';
    };

    clip = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "CLIP text encoder passed as `--clip` (SD1.5/SDXL-style models).";
    };

    llmVision = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "Vision projector passed as `--llm_vision`; enables image editing (`/v1/images/edits`) with GGUF text encoders.";
    };

    offloadToCpu = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Pass `--offload-to-cpu`: keep less-than-VRAM weights in system RAM.
        The text encoder runs once per prompt, so RAM-resident encoders cost
        virtually nothing in generation speed.
      '';
    };

    flashAttention = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Pass `--diffusion-fa` (flash attention in the diffusion model).";
    };

    vaeTiling = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Pass `--vae-tiling` (VAE tiling to cut peak VRAM at high resolutions).";
    };

    cfgScale = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "6.0";
      description = "Default CFG scale (`--cfg-scale`, string to avoid float formatting surprises). Requests may still override it.";
    };

    samplingMethod = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "euler";
      description = "Default sampling method (`--sampling-method`).";
    };

    numThreads = lib.mkOption {
      type = lib.types.nullOr lib.types.ints.positive;
      default = null;
      description = "CPU thread count (`--threads`).";
    };

    memoryMax = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "60G";
      description = "systemd `MemoryMax=` for the service (runaway offload guard).";
    };

    extraArgs = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [];
      example = ["--main-gpu" "0" "--threads" "16"];
      description = "Extra arguments appended to `sd-server`.";
    };

    environment = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = {};
      description = "Extra environment variables for the sd-cpp service.";
    };

    cudaToolkit = lib.mkOption {
      type = lib.types.package;
      default = pkgs.cudaPackages_13.cudatoolkit;
      defaultText = lib.literalExpression "pkgs.cudaPackages_13.cudatoolkit";
      description = ''
        CUDA toolkit whose runtime libraries (cudart, cublas, ...) the
        sd.cpp binaries are linked against; its lib dir is added to
        {env}`LD_LIBRARY_PATH` next to the nixpkgs driver.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = cfg.diffusionModel != null;
        message = "services.sd-cpp.diffusionModel must be set.";
      }
      {
        assertion = cfg.vae != null;
        message = "services.sd-cpp.vae must be set.";
      }
    ];

    systemd.services.sd-cpp = {
      description = "stable-diffusion.cpp image-generation server";
      wantedBy = ["multi-user.target"];
      wants = ["network-online.target"];
      after = ["network-online.target"];

      # The sd.cpp build links shared CUDA runtime libraries from the
      # toolkit; libcuda.so.1 comes from the nixpkgs driver.
      environment = {
        LD_LIBRARY_PATH = lib.makeLibraryPath [cfg.cudaToolkit] + ":${driverLib}";
      }
      // cfg.environment;

      serviceConfig = {
        ExecStart = "${cfg.package}/bin/sd-server ${lib.escapeShellArgs serveArgs}";
        User = "sd-cpp";
        Group = "sd-cpp";
        StateDirectory = "sd-cpp";
        WorkingDirectory = stateDir;

        Restart = "always";
        RestartSec = 5;
        # First start loads multi-GB of weights into RAM/VRAM.
        TimeoutStartSec = "10min";

        MemoryMax = lib.mkIf (cfg.memoryMax != null) cfg.memoryMax;

        # GPU access
        SupplementaryGroups = ["video" "render"];
        DeviceAllow = [
          "char-nvidiactl"
          "char-nvidia-caps"
          "char-nvidia-frontend"
          "char-nvidia-uvm"
          "char-drm"
        ];

        # Hardening
        NoNewPrivileges = true;
        PrivateTmp = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ProtectKernelLogs = true;
        ProtectKernelModules = true;
        ProtectKernelTunables = true;
        ProtectControlGroups = true;
        ProtectHostname = true;
        RestrictRealtime = true;
        RestrictSUIDSGID = true;
        LockPersonality = true;
        SystemCallArchitectures = "native";
        RestrictAddressFamilies = ["AF_INET" "AF_INET6" "AF_UNIX" "AF_NETLINK"];
        UMask = "0077";
      };
    };

    users.users.sd-cpp = {
      isSystemUser = true;
      group = "sd-cpp";
      home = stateDir;
      description = "sd-cpp service user";
    };
    users.groups.sd-cpp = {};

    networking.firewall.allowedTCPPorts = lib.optional cfg.openFirewall cfg.port;
  };
}
