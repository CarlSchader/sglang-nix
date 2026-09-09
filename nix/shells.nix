{
  nixpkgs,
  flake-utils,
  ...
}:
flake-utils.lib.eachDefaultSystem (system: let
  pkgs = import nixpkgs {
    inherit system;
    config = {
      allowUnfree = true;
    };
  };
in {
  devShells.default = pkgs.mkShell {
    env = {
      LD_LIBRARY_PATH =
        pkgs.lib.makeLibraryPath (with pkgs; [
          stdenv.cc.cc.lib
          zlib
        ])
        + ":${pkgs.addDriverRunpath.driverLink}/lib";
      # So gcc can find the CUDA stubs when FlashInfer / sgl-kernel JIT.
      LIBRARY_PATH = pkgs.lib.concatStringsSep ":" [
        "${pkgs.cudaPackages_13.cudatoolkit}/lib"
        "${pkgs.cudaPackages_13.cudatoolkit}/lib/stubs"
      ];
      # sglang's JIT does not pass -I$CUDA_HOME/include to nvcc; see modules/sglang.nix.
      NVCC_PREPEND_FLAGS = pkgs.lib.concatStringsSep " " [
        "-I${pkgs.cudaPackages_13.cudatoolkit}/include"
        "-L${pkgs.cudaPackages_13.cudatoolkit}/lib"
        "-L${pkgs.cudaPackages_13.cudatoolkit}/lib/stubs"
      ];
      TRITON_LIBCUDA_PATH = "${pkgs.addDriverRunpath.driverLink}/lib";
      CUDA_HOME = "${pkgs.cudaPackages_13.cudatoolkit}";
    };
    buildInputs = with pkgs; [
      uv
      gcc
      cudaPackages_13.cudatoolkit
    ];
  };
})
