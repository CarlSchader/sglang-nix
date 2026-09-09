{...}: {
  nixosModules = rec {
    sglang = import ./sglang.nix;
    default = sglang;
  };
}
