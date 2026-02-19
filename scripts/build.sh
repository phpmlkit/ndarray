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

# Default to release build
BUILD_MODE="${1:-release}"

if [ "$BUILD_MODE" = "debug" ]; then
    echo "Building Rust library (debug mode)..."
    cd rust
    cargo build
    cd ..
    
    SOURCE_DIR="rust/target/debug"
else
    echo "Building Rust library (release mode)..."
    cd rust
    cargo build --release
    cd ..
    
    SOURCE_DIR="rust/target/release"
fi

# Create platform-specific directory
PLATFORM_DIR="lib/$PLATFORM"
mkdir -p "$PLATFORM_DIR"

# Copy binary with versioned name based on platform conventions
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux: lib<name>.so.<version>
    cp "${SOURCE_DIR}/libndarray_php.so" "${PLATFORM_DIR}/libndarray_php.so.${VERSION}"
    echo "Created: ${PLATFORM_DIR}/libndarray_php.so.${VERSION}"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS: lib<name>-<version>.dylib
    cp "${SOURCE_DIR}/libndarray_php.dylib" "${PLATFORM_DIR}/libndarray_php-${VERSION}.dylib"
    echo "Created: ${PLATFORM_DIR}/libndarray_php-${VERSION}.dylib"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows: <name>-<version>.dll
    cp "${SOURCE_DIR}/ndarray_php.dll" "${PLATFORM_DIR}/ndarray_php-${VERSION}.dll"
    echo "Created: ${PLATFORM_DIR}/ndarray_php-${VERSION}.dll"
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
