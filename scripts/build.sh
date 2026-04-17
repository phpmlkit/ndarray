#!/bin/bash
set -e

# Get version from Cargo.toml
VERSION=$(grep '^version' rust/Cargo.toml | head -1 | cut -d'"' -f2)
echo "Building NDArray PHP v${VERSION}"

# Detect platform and architecture
PLATFORM=""
ARCH=$(uname -m)

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if [[ "$ARCH" == "aarch64" ]] || [[ "$ARCH" == "arm64" ]]; then
        PLATFORM="linux-arm64"
    else
        PLATFORM="linux-x86_64"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    if [[ "$ARCH" == "arm64" ]]; then
        PLATFORM="darwin-arm64"
    else
        PLATFORM="darwin-x86_64"
    fi
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    PLATFORM="windows-64"
else
    echo "Warning: Unknown platform: $OSTYPE"
    PLATFORM="unknown"
fi

echo "Detected platform: $PLATFORM"

# Parse arguments
BUILD_MODE="release"

for arg in "$@"; do
    case $arg in
        debug)
            BUILD_MODE="debug"
            ;;
    esac
done

# Setup environment for BLAS build on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    if [ -d "/opt/homebrew/Cellar/gcc" ]; then
        GCC_VERSION=$(ls /opt/homebrew/Cellar/gcc | sort -V | tail -1)
        export LIBRARY_PATH="/opt/homebrew/Cellar/gcc/${GCC_VERSION}/lib/gcc/current:${LIBRARY_PATH:-}"
    fi

    # GitHub Actions macOS runners install gfortran as versioned aliases
    # (e.g. gfortran-13, gfortran-14, gfortran-15) but not plain 'gfortran'.
    # Prefer plain 'gfortran' if available, otherwise use the latest versioned alias.
    if command -v gfortran &> /dev/null; then
        export FC=${FC:-gfortran}
        export F77=${F77:-gfortran}
    else
        for version in 15 14 13 12 11; do
            if command -v "gfortran-$version" &> /dev/null; then
                export FC="gfortran-$version"
                export F77="gfortran-$version"
                echo "Using Fortran compiler: gfortran-$version"
                break
            fi
        done
    fi

    if [ -d "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk" ]; then
        export SDKROOT="/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
    fi
fi

if [ "$BUILD_MODE" = "debug" ]; then
    echo "Building Rust library (debug mode)..."
    cd rust
    cargo build --features ffi
    cd ..

    SOURCE_DIR="rust/target/debug"
else
    echo "Building Rust library (release mode)..."
    cd rust
    cargo build --release --features ffi
    cd ..

    SOURCE_DIR="rust/target/release"
fi

# Create platform-specific directory
PLATFORM_DIR="lib/$PLATFORM"
mkdir -p "$PLATFORM_DIR"

# Copy binary with versioned name based on platform conventions
LIB_FILE=""
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux: lib<name>.so.<version>
    LIB_FILE="${PLATFORM_DIR}/libndarray_php.so.${VERSION}"
    cp "${SOURCE_DIR}/libndarray_php.so" "$LIB_FILE"
    echo "Created: $LIB_FILE"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS: lib<name>-<version>.dylib
    LIB_FILE="${PLATFORM_DIR}/libndarray_php-${VERSION}.dylib"
    cp "${SOURCE_DIR}/libndarray_php.dylib" "$LIB_FILE"
    echo "Created: $LIB_FILE"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows: <name>-<version>.dll
    LIB_FILE="${PLATFORM_DIR}/ndarray_php-${VERSION}.dll"
    cp "${SOURCE_DIR}/ndarray_php.dll" "$LIB_FILE"
    echo "Created: $LIB_FILE"
fi

# Linux: bundle Fortran runtime libraries and set rpath
if [[ "$OSTYPE" == "linux-gnu"* ]] && [ -n "$LIB_FILE" ]; then
    if command -v patchelf &> /dev/null; then
        patchelf --set-rpath '$ORIGIN' "$LIB_FILE"
        echo "Set rpath to \$ORIGIN for bundled libraries"
    else
        echo "Warning: patchelf not found. The shared library may not find bundled dependencies at runtime."
    fi

    lib="libgfortran.so.5"
    lib_path=$(ldd "$LIB_FILE" 2>/dev/null | awk -v lib="$lib" '$1 == lib {print $3}')
    if [ -n "$lib_path" ] && [ -f "$lib_path" ]; then
        cp "$lib_path" "$PLATFORM_DIR/" 2>/dev/null
        echo "Bundled: $lib"
    fi
fi

echo ""
echo "✅ Build complete for $PLATFORM!"
echo ""
echo "Library location: lib/$PLATFORM/"
echo "Header location: include/ndarray_php.h"
echo ""
echo "Usage:"
echo "  ./scripts/build.sh        # Release build (default)"
echo "  ./scripts/build.sh debug  # Debug build (faster)"
