#!/bin/bash
set -e

# Default to release build
BUILD_MODE="${1:-release}"

if [ "$BUILD_MODE" = "debug" ]; then
    echo "Building Rust library (debug mode)..."
    cd rust
    cargo build
    cd ..
    
    echo "Copying library..."
    mkdir -p lib
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        cp rust/target/debug/libndarray_php.so lib/
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        cp rust/target/debug/libndarray_php.dylib lib/
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        cp rust/target/debug/ndarray_php.dll lib/
    fi
else
    echo "Building Rust library (release mode)..."
    cd rust
    cargo build --release
    cd ..
    
    echo "Copying library..."
    mkdir -p lib
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        cp rust/target/release/libndarray_php.so lib/
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        cp rust/target/release/libndarray_php.dylib lib/
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        cp rust/target/release/ndarray_php.dll lib/
    fi
fi

echo "✅ Build complete!"
echo ""
echo "Usage:"
echo "  ./scripts/build.sh        # Release build (default)"
echo "  ./scripts/build.sh debug  # Debug build (faster)"
