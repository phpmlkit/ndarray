#!/bin/bash
set -e

echo "Building Rust library..."
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

echo "✅ Build complete!"
