#!/bin/bash
set -e

echo "Building Rust library..."
cd rust
cargo build --release
cd ..

echo "Copying library..."
mkdir -p lib

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    cp rust/target/release/libphpndarray.so lib/
elif [[ "$OSTYPE" == "darwin"* ]]; then
    cp rust/target/release/libphpndarray.dylib lib/
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    cp rust/target/release/phpndarray.dll lib/
fi

echo "✅ Build complete!"
