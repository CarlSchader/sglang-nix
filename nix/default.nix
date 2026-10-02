{flake-utils, ...} @ inputs:
flake-utils.lib.meld inputs [
  ./models.nix
  ./shells.nix
  ./sglang-env.nix
  ./sglang-watch.nix
]
