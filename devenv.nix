{ pkgs, lib, ... }:

# Pin vcpkg commit to match community-extensions description.yml
let
  vcpkgCommit = "e5a1490e1409d175932ef6014519e9ae149ddb7c";
in
{
  # Only provide dependencies not available in macOS SDK.
  # Metal frameworks come from Xcode.
  # FAISS is provided by vcpkg (matching CI pipeline) — not nix.
  packages = [
    pkgs.git
    pkgs.gh
    pkgs.gnumake
    pkgs.cmake
    pkgs.ninja
    pkgs.llvmPackages.openmp  # needed by vcpkg's faiss build

    # C/C++ tools
    pkgs.autoconf
    pkgs.automake
    pkgs.pkg-config
    pkgs.clang-tools

    # Rust toolchain
    pkgs.rustup

    # vcpkg needs curl and zip/unzip for downloading ports
    pkgs.curl
    pkgs.zip
    pkgs.unzip
  ];

  # Do NOT enable languages.cplusplus -- it pulls in nix apple-sdk which
  # conflicts with the real macOS SDK needed for Metal/MPS headers.

  # Nix overrides DEVELOPER_DIR to its stripped apple-sdk which lacks Metal tools.
  env.DEVELOPER_DIR = lib.mkForce "/Applications/Xcode.app/Contents/Developer";

  enterShell = ''
    export GEN=ninja
    export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"
    unset SDKROOT
    unset NIX_CFLAGS_COMPILE
    unset NIX_LDFLAGS
    export CC=/usr/bin/clang
    export CXX=/usr/bin/clang++

    # Bootstrap vcpkg (matching CI pipeline)
    VCPKG_DIR="$PWD/vcpkg"
    if [ ! -d "$VCPKG_DIR" ]; then
      echo "Fetching vcpkg at pinned commit ${vcpkgCommit}..."
      git init "$VCPKG_DIR"
      git -C "$VCPKG_DIR" fetch --depth 1 https://github.com/microsoft/vcpkg.git ${vcpkgCommit}
      git -C "$VCPKG_DIR" checkout FETCH_HEAD
    fi
    if [ ! -x "$VCPKG_DIR/vcpkg" ]; then
      echo "Bootstrapping vcpkg..."
      "$VCPKG_DIR/bootstrap-vcpkg.sh" -disableMetrics
    fi
    export VCPKG_TOOLCHAIN_PATH="$VCPKG_DIR/scripts/buildsystems/vcpkg.cmake"
  '';

  claude.code.enable = true;

  claude.code.mcpServers.consult-llm = {
    type = "stdio";
    command = "npx";
    args = [
      "-y"
      "consult-llm-mcp"
    ];
    env = {
      CONSULT_LLM_DEFAULT_MODEL = "gemini-3-pro-preview";
      CONSULT_LLM_ALLOWED_MODELS = "gemini-3-pro-preview";
    };
  };

  git-hooks.hooks = {
    ripsecrets.enable = true;
    clang-format = {
      enable = true;
      types_or = [ "c++" "c" ];
    };
  };
}
