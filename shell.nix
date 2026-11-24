# Cross-compiling VIctory for Windows (x86_64-pc-windows-msvc) from NixOS.
#
#   nix-shell
#   cargo xwin build --release --target x86_64-pc-windows-msvc
#
# rustup reads rust-toolchain.toml and installs the pinned toolchain and the
# MSVC target on first use. cargo-xwin downloads the MSVC CRT and Windows SDK
# (Microsoft's license applies) and links with clang-cl / lld-link.
{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  packages = with pkgs; [
    rustup # its cargo/rustc proxies shadow the system Rust inside this shell
    cargo-xwin
    llvmPackages.clang-unwrapped # clang-cl
    llvmPackages.lld # lld-link
    llvmPackages.llvm # llvm-lib, llvm-ar
  ];

  # `cargo xwin test --target x86_64-pc-windows-msvc` runs the test binaries
  # under the system Wine.
  CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUNNER = "wine";

  # A prefix of VIctory's own whose system32 holds VKD3D-Proton (d3d12,
  # d3d12core) and DXVK (dxgi, d3d11, d3d10core), set up as the README says;
  # Wine's builtins are wined3d and upstream vkd3d, which cannot run the
  # title's shaders (SM 6.6).
  shellHook = ''
    export WINEPREFIX="$HOME/.local/share/victory-wine"
    # Otherwise VKD3D-Proton writes its cache into the working directory,
    # which the asset policy refuses.
    export VKD3D_SHADER_CACHE_PATH="$WINEPREFIX/vkd3d-cache"
    export DXVK_STATE_CACHE_PATH="$WINEPREFIX/dxvk-cache"
    mkdir -p "$VKD3D_SHADER_CACHE_PATH" "$DXVK_STATE_CACHE_PATH"
  '';
  WINEDLLOVERRIDES = "d3d12,d3d12core,dxgi,d3d11,d3d10core=n";
}
