# NixOS module: services.sglang
#
# Runs `sglang serve` from a pinned python environment (see
# sglang-nix.packages.<system>.sglangEnv) as a hardened, auto-restarting
# systemd service, and optionally fronts it with nixpkgs' services.open-webui.
#
# Requires a CUDA-capable NVIDIA GPU managed by nixpkgs
# (hardware.nvidia / /run/opengl-driver/lib).
{
  config,
  lib,
  pkgs,
  ...
}: let
  cfg = config.services.sglang;

  driverLib = "${pkgs.addDriverRunpath.driverLink}/lib";

  stateDir = "/var/lib/sglang";

  serveArgs =
    [
      "serve"
      "--model-path"
      cfg.model.hfId
      "--served-model-name"
      cfg.model.servedModelName
      "--host"
      cfg.host
      "--port"
      (toString cfg.port)
      "--context-length"
      (toString cfg.model.contextLength)
      "--mem-fraction-static"
      cfg.memFractionStatic
      "--tp-size"
      (toString cfg.tensorParallelSize)
    ]
    ++ lib.optionals (cfg.model.revision != null) ["--revision" cfg.model.revision]
    ++ lib.optionals (cfg.model.chatTemplate != null) ["--chat-template" (toString cfg.model.chatTemplate)]
    ++ lib.optionals (cfg.speculative.algorithm != null) (
      ["--speculative-algorithm" cfg.speculative.algorithm]
      ++ lib.optionals (cfg.speculative.draftModelPath != null) ["--speculative-draft-model-path" cfg.speculative.draftModelPath]
      ++ lib.optionals (cfg.speculative.draftModelRevision != null) ["--speculative-draft-model-revision" cfg.speculative.draftModelRevision]
      ++ lib.optionals (cfg.speculative.numDraftTokens != null) ["--speculative-num-draft-tokens" (toString cfg.speculative.numDraftTokens)]
      ++ lib.optionals (cfg.speculative.numSteps != null) ["--speculative-num-steps" (toString cfg.speculative.numSteps)]
      ++ lib.optionals (cfg.speculative.eagleTopk != null) ["--speculative-eagle-topk" (toString cfg.speculative.eagleTopk)]
      ++ lib.optionals (cfg.speculative.draftModelQuantization != null) ["--speculative-draft-model-quantization" cfg.speculative.draftModelQuantization]
    )
    ++ lib.optionals cfg.torchCompile.enable (
      ["--enable-torch-compile"]
      ++ lib.optionals (cfg.torchCompile.maxBs != null) ["--torch-compile-max-bs" (toString cfg.torchCompile.maxBs)]
    )
    ++ lib.optionals (cfg.maxRunningRequests != null) ["--max-running-requests" (toString cfg.maxRunningRequests)]
    ++ lib.optionals (cfg.chunkedPrefillSize != null) ["--chunked-prefill-size" (toString cfg.chunkedPrefillSize)]
    ++ lib.optionals (cfg.kvCacheDtype != null) ["--kv-cache-dtype" cfg.kvCacheDtype]
    ++ lib.optionals (cfg.attentionBackend != null) ["--attention-backend" cfg.attentionBackend]
    ++ lib.optionals (cfg.toolCallParser != null) ["--tool-call-parser" cfg.toolCallParser]
    ++ lib.optionals (cfg.reasoningParser != null) ["--reasoning-parser" cfg.reasoningParser]
    ++ lib.optional cfg.enableMultimodal "--enable-multimodal"
    ++ lib.optional cfg.trustRemoteCode "--trust-remote-code"
    ++ cfg.extraArgs;
in {
  options.services.sglang = {
    enable = lib.mkEnableOption "SGLang OpenAI-compatible inference server";

    package = lib.mkOption {
      type = lib.types.package;
      description = ''
        Python environment providing {file}`bin/sglang`. Normally
        `sglang-nix.packages.''${pkgs.system}.sglangEnv`, the environment
        pinned by this flake's uv.lock.
      '';
    };

    host = lib.mkOption {
      type = lib.types.str;
      default = "0.0.0.0";
      description = "Address the OpenAI-compatible API listens on.";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 30000;
      description = "Port the OpenAI-compatible API listens on (sglang's default is 30000).";
    };

    openFirewall = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Open the API port in the firewall.";
    };

    model = {
      hfId = lib.mkOption {
        type = lib.types.str;
        example = "Inferact/Qwen3.8-27B-NVFP4";
        description = "Hugging Face model id (or local path) passed as `--model-path`.";
      };

      servedModelName = lib.mkOption {
        type = lib.types.str;
        default = lib.last (lib.splitString "/" cfg.model.hfId);
        defaultText = lib.literalExpression ''lib.last (lib.splitString "/" cfg.model.hfId)'';
        description = "Model name exposed on the OpenAI API.";
      };

      contextLength = lib.mkOption {
        type = lib.types.ints.positive;
        default = 32768;
        example = 262144;
        description = "Context window (`--context-length`).";
      };

      revision = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        example = "52d1adc5f38aa5ebf099c29ed7025ba34cfbb854";
        description = "Pin the Hugging Face revision (`--revision`) so an upstream push can't change what is served.";
      };

      chatTemplate = lib.mkOption {
        type = lib.types.nullOr lib.types.path;
        default = null;
        description = "Override the chat template (`--chat-template`), e.g. one patched for agentic clients.";
      };
    };

    speculative = {
      algorithm = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        example = "DFLASH";
        description = "`--speculative-algorithm` (EAGLE, EAGLE3, NEXTN, DFLASH, NGRAM, ...). null disables speculative decoding.";
      };

      draftModelPath = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        example = "z-lab/Qwen3.8-27B-DFlash2";
        description = "`--speculative-draft-model-path`.";
      };

      draftModelRevision = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        description = "`--speculative-draft-model-revision`.";
      };

      numDraftTokens = lib.mkOption {
        type = lib.types.nullOr lib.types.ints.positive;
        default = null;
        example = 8;
        description = "`--speculative-num-draft-tokens`.";
      };

      numSteps = lib.mkOption {
        type = lib.types.nullOr lib.types.ints.positive;
        default = null;
        description = "`--speculative-num-steps` (EAGLE/NEXTN).";
      };

      eagleTopk = lib.mkOption {
        type = lib.types.nullOr lib.types.ints.positive;
        default = null;
        description = "`--speculative-eagle-topk` (EAGLE/NEXTN).";
      };

      draftModelQuantization = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        example = "unquant";
        description = "`--speculative-draft-model-quantization`.";
      };
    };

    torchCompile = {
      enable = lib.mkEnableOption "`--enable-torch-compile` (slower first boot, faster decode; inductor cache persists in the state dir)";

      maxBs = lib.mkOption {
        type = lib.types.nullOr lib.types.ints.positive;
        default = null;
        example = 4;
        description = "`--torch-compile-max-bs`.";
      };
    };

    memoryMax = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "100G";
      description = ''
        systemd `MemoryMax=` for the service. On unified-memory hosts (DGX
        Spark) a runaway engine can drive host memory to zero and freeze the
        box; a cap makes the OOM killer take the engine instead. Note the
        cgroup does not see CUDA unified allocations, so `memFractionStatic`
        remains the real guard.
      '';
    };

    maxRunningRequests = lib.mkOption {
      type = lib.types.nullOr lib.types.ints.positive;
      default = 8;
      description = "`--max-running-requests` (null to let sglang decide).";
    };

    chunkedPrefillSize = lib.mkOption {
      type = lib.types.nullOr lib.types.int;
      default = null;
      example = 2048;
      description = "`--chunked-prefill-size` (null to use sglang's default, -1 to disable chunked prefill).";
    };

    memFractionStatic = lib.mkOption {
      type = lib.types.str;
      # On unified-memory hosts (e.g. DGX Spark/GB10) the desktop and driver
      # already hold several GiB at startup, so leave headroom.
      default = "0.85";
      description = "`--mem-fraction-static` (string to avoid float formatting surprises).";
    };

    tensorParallelSize = lib.mkOption {
      type = lib.types.ints.positive;
      default = 1;
      description = "`--tp-size`.";
    };

    kvCacheDtype = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "fp8_e4m3";
      description = "`--kv-cache-dtype` (`auto`, `fp8_e5m2`, `fp8_e4m3`, `nvfp4`, ...).";
    };

    attentionBackend = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "triton";
      description = "`--attention-backend` (`flashinfer`, `triton`, `fa3`, `trtllm_mha`, ...).";
    };

    toolCallParser = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = "qwen3_coder";
      description = "`--tool-call-parser` (null to omit).";
    };

    reasoningParser = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = "qwen3";
      description = "`--reasoning-parser` (null to omit).";
    };

    enableMultimodal = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Pass `--enable-multimodal` (image/video inputs). Off by default, text-only.";
    };

    trustRemoteCode = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Pass `--trust-remote-code`.";
    };

    extraArgs = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [];
      example = ["--speculative-algorithm" "NEXTN" "--speculative-num-steps" "3"];
      description = "Extra arguments appended to `sglang serve`.";
    };

    environment = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = {};
      example = {
        SGLANG_LOGGING_LEVEL = "DEBUG";
      };
      description = "Extra environment variables for the sglang service.";
    };

    environmentFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      example = "/run/secrets/sglang.env";
      description = ''
        Environment file for secrets (e.g. `HF_TOKEN=...` for gated models).
      '';
    };

    cudaToolkit = {
      enable = lib.mkOption {
        type = lib.types.bool;
        default = true;
        description = ''
          Put a CUDA toolkit and gcc on the service's PATH and set
          {env}`CUDA_HOME`. Required for FlashInfer / DeepGEMM runtime JIT
          compilation; disable only if your configuration never JITs.
        '';
      };

      package = lib.mkOption {
        type = lib.types.package;
        default = pkgs.cudaPackages_13.cudatoolkit;
        defaultText = lib.literalExpression "pkgs.cudaPackages_13.cudatoolkit";
        description = "CUDA toolkit used for runtime JIT compilation.";
      };
    };

    ui = {
      enable = lib.mkEnableOption "Open WebUI frontend wired to this SGLang instance";

      package = lib.mkOption {
        type = lib.types.package;
        default = pkgs.open-webui.override {
          # open-webui -> sentence-transformers -> torchaudio. On aarch64 the
          # torch stack is not in the binary cache and gets built from source,
          # where torchaudio's test_batch_melspectrogram fails on a numeric
          # tolerance (flaky). The tests are not worth a failed rebuild.
          python3Packages = pkgs.python3Packages.overrideScope (_: prev: {
            torchaudio = prev.torchaudio.overridePythonAttrs (_: {
              doCheck = false;
            });
          });
        };
        defaultText = lib.literalExpression "pkgs.open-webui (with torchaudio tests disabled)";
        description = "Open WebUI package to run.";
      };

      host = lib.mkOption {
        type = lib.types.str;
        default = "127.0.0.1";
        example = "0.0.0.0";
        description = "Address Open WebUI listens on.";
      };

      port = lib.mkOption {
        type = lib.types.port;
        default = 8080;
        description = "Port Open WebUI listens on.";
      };

      openFirewall = lib.mkOption {
        type = lib.types.bool;
        default = false;
        description = "Open the UI port in the firewall.";
      };

      webSearch = {
        enable = lib.mkEnableOption "web search in Open WebUI";

        engine = lib.mkOption {
          type = lib.types.str;
          default = "duckduckgo";
          description = "Open WebUI web search engine.";
        };

        resultCount = lib.mkOption {
          type = lib.types.ints.positive;
          default = 5;
          description = "Search results per query.";
        };

        concurrentRequests = lib.mkOption {
          type = lib.types.ints.positive;
          default = 10;
          description = "Concurrent web search requests.";
        };
      };

      environment = lib.mkOption {
        type = lib.types.attrsOf lib.types.str;
        default = {};
        description = ''
          Extra environment variables for Open WebUI (merged last, wins over
          the ones set by this module).
        '';
      };
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = cfg.model.hfId != "";
        message = "services.sglang.model.hfId must be set.";
      }
    ];

    systemd.services.sglang = {
      description = "SGLang OpenAI-compatible inference server (${cfg.model.servedModelName})";
      wantedBy = ["multi-user.target"];
      wants = ["network-online.target"];
      after = ["network-online.target"];

      environment =
        {
          HOME = stateDir;
          HF_HOME = "${stateDir}/huggingface";
          XDG_CACHE_HOME = "${stateDir}/.cache";
          # Persist torch.compile / triton / flashinfer JIT artefacts across
          # restarts (they are the bulk of the multi-minute first boot).
          TORCHINDUCTOR_CACHE_DIR = "${stateDir}/.cache/inductor";
          TRITON_CACHE_DIR = "${stateDir}/.cache/triton";
          # The wheels are patched to find everything except the driver and
          # the system C++/zlib libraries; provide those here (mirrors the
          # dev shell).
          LD_LIBRARY_PATH =
            lib.makeLibraryPath [pkgs.stdenv.cc.cc.lib pkgs.zlib]
            + ":${driverLib}";
          TRITON_LIBCUDA_PATH = driverLib;
        }
        // lib.optionalAttrs cfg.cudaToolkit.enable {
          CUDA_HOME = "${cfg.cudaToolkit.package}";
          # gcc needs to find the CUDA runtime stubs when kernels are JITed.
          LIBRARY_PATH = lib.concatStringsSep ":" [
            "${cfg.cudaToolkit.package}/lib"
            "${cfg.cudaToolkit.package}/lib/stubs"
          ];
          # nixpkgs' merged toolkit symlinks bin/nvcc into the split cuda_nvcc
          # package, so nvcc's own $TOP/include lacks cuda_runtime.h. sglang's
          # JIT (unlike FlashInfer's) does not pass -I$CUDA_HOME/include, so
          # inject it via nvcc's env hook.
          NVCC_PREPEND_FLAGS = lib.concatStringsSep " " [
            "-I${cfg.cudaToolkit.package}/include"
            "-L${cfg.cudaToolkit.package}/lib"
            "-L${cfg.cudaToolkit.package}/lib/stubs"
          ];
        }
        // cfg.environment;

      path =
        [
          # The venv itself: FlashInfer's JIT invokes `ninja` (shipped in the
          # venv's bin/) via subprocess and expects it on PATH.
          cfg.package
        ]
        ++ lib.optionals cfg.cudaToolkit.enable [
          pkgs.gcc
          cfg.cudaToolkit.package
        ];

      serviceConfig = {
        ExecStart = "${cfg.package}/bin/sglang ${lib.escapeShellArgs serveArgs}";
        User = "sglang";
        Group = "sglang";
        StateDirectory = "sglang";
        WorkingDirectory = stateDir;

        Restart = "always";
        RestartSec = 5;
        # First start may download tens of GB of weights.
        TimeoutStartSec = "60min";

        EnvironmentFile = lib.optional (cfg.environmentFile != null) cfg.environmentFile;
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

        # Hardening (kept compatible with CUDA + JIT compilation)
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

    users.users.sglang = {
      isSystemUser = true;
      group = "sglang";
      home = stateDir;
      description = "SGLang service user";
    };
    users.groups.sglang = {};

    networking.firewall.allowedTCPPorts = lib.optional cfg.openFirewall cfg.port;

    services.open-webui = lib.mkIf cfg.ui.enable {
      enable = true;
      package = cfg.ui.package;
      host = cfg.ui.host;
      port = cfg.ui.port;
      openFirewall = cfg.ui.openFirewall;
      environment =
        {
          # Defaults from the nixpkgs module (replaced when we set this option).
          SCARF_NO_ANALYTICS = "True";
          DO_NOT_TRACK = "True";
          ANONYMIZED_TELEMETRY = "False";
          # Point at the local SGLang OpenAI endpoint.
          OPENAI_API_BASE_URL = "http://127.0.0.1:${toString cfg.port}/v1";
          OPENAI_API_KEY = "EMPTY";
          ENABLE_OLLAMA_API = "False";
        }
        // lib.optionalAttrs cfg.ui.webSearch.enable {
          ENABLE_WEB_SEARCH = "True";
          WEB_SEARCH_ENGINE = cfg.ui.webSearch.engine;
          WEB_SEARCH_RESULT_COUNT = toString cfg.ui.webSearch.resultCount;
          WEB_SEARCH_CONCURRENT_REQUESTS = toString cfg.ui.webSearch.concurrentRequests;
        }
        // cfg.ui.environment;
    };

    # Start the UI after the API it fronts.
    systemd.services.open-webui = lib.mkIf cfg.ui.enable {
      after = ["sglang.service"];
      wants = ["sglang.service"];
    };
  };
}
