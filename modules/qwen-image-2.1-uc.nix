# Preset: Qwen-Image-2.1 "Uncensored" GGUF text-to-image on any CUDA machine
# with ~30 GB of RAM headroom, served by stable-diffusion.cpp.
#
# Weights: abenzerps/Qwen-Image-2.1-Uncensored-GGUF on Hugging Face
# (Qwen/Qwen-Image-2.1 base weights, GGUF-converted by stable-diffusion.cpp
# commit 1330ceb). This model's model card targets ComfyUI + ComfyUI-GGUF,
# but sd.cpp (the converter, with day-0 Qwen-Image-2.1 support) loads the
# exact same files natively, so no ComfyUI is required.
#
# Model choice (per the model card's memory notes):
#   * diffusion  Q4_K_M GGUF           ~4.6 GB in VRAM (recommended quant)
#   * text enc.  Qwen3-VL-8B int8      ~9.4 GB, offloaded to RAM once per
#              convrot                    prompt
#   * VAE        bf16                  ~0.7 GB
# On a 24 GB card (e.g. RTX 4090) that leaves ~9 GB for attention working
# sets and the (default-on) prefix cache at long prompts.
#
# Everything the preset sets is `mkDefault`, so any option can be overridden
# (e.g. set `llmVision` to enable image editing). Import next to
# nixosModules.sd-cpp and set `services.sd-cpp.enable` + `package`:
#
#   imports = [
#     sglang-nix.nixosModules.sd-cpp
#     sglang-nix.nixosModules."qwen-image-2.1-uc"
#   ];
#   services.sd-cpp = { enable = true; package = sglang-nix.packages.${pkgs.system}.sdcpp; };
{
  config,
  pkgs,
  lib,
  ...
}: let
  cfg = config.services.sd-cpp;

  # https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF
  # (files are pinned by the sha256s from that repo's SHA256SUMS).
  hf = "https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF/resolve/main";
in {
  config = lib.mkIf cfg.enable {
    services.sd-cpp = {
      diffusionModel = lib.mkDefault (
        pkgs.fetchurl {
          url = "${hf}/qwen-image-2.1-UC-Q4_K_M.gguf";
          sha256 = "e79c8a009f2ecbdb6c70fd663d9aea9ee304a0d91f347e4169a756b8ad141b41";
        }
      );

      vae = lib.mkDefault (
        pkgs.fetchurl {
          url = "${hf}/vae/qwen_image_2.1_vae_bf16.safetensors";
          sha256 = "bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9";
        }
      );

      textEncoder = lib.mkDefault (
        pkgs.fetchurl {
          url = "${hf}/text_encoders/qwen3vl_8b_int8_convrot.safetensors";
          sha256 = "8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f";
        }
      );

      # Generation defaults from sd.cpp's Qwen-Image-2.1 docs / examples.
      cfgScale = lib.mkDefault "6.0";
      samplingMethod = lib.mkDefault "euler";
    };
  };
}
