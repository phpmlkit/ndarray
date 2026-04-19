#!/usr/bin/env bash
# Create a distribution archive by explicitly including only release artifacts.
# Run from the repository root after building the native library (see scripts/build.sh).
#
# Usage:
#   ./scripts/package.sh <platform>
#
# Examples:
#   ./scripts/package.sh darwin-arm64
#   ./scripts/package.sh linux-x86_64
#   ./scripts/package.sh windows-x64
#
# Format is inferred from the platform name: windows-* uses zip; all others use tar.gz.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <platform>" >&2
    echo "  platform   e.g. linux-x86_64, darwin-arm64, windows-x64" >&2
    exit 1
fi

PLATFORM="$1"

if [[ "$PLATFORM" == windows-* ]]; then
    FORMAT="zip"
else
    FORMAT="tar.gz"
fi

DIST_NAME="dist-${PLATFORM}"

echo "Creating distribution: ${DIST_NAME} (${FORMAT})"

rm -rf "${DIST_NAME}"
rm -f "${DIST_NAME}.tar.gz" "${DIST_NAME}.zip"

mkdir -p "${DIST_NAME}"

dirs_to_include=(
    "include"
    "lib"
    "scripts"
    "src"
    "tests"
)

files_to_include=(
    "composer.json"
    "phpunit.xml"
    "LICENSE"
    "README.md"
    "AGENTS.md"
    "SPEC.md"
)

for dir in "${dirs_to_include[@]}"; do
    if [[ -d "$dir" ]]; then
        cp -r "$dir" "${DIST_NAME}/"
    else
        echo "warning: expected directory '${dir}' not found, skipping" >&2
    fi
done

for file in "${files_to_include[@]}"; do
    if [[ -f "$file" ]]; then
        cp "$file" "${DIST_NAME}/"
    else
        echo "warning: expected file '${file}' not found, skipping" >&2
    fi
done

if [[ "$FORMAT" == "zip" ]]; then
    if command -v zip >/dev/null 2>&1; then
        (cd . && zip -rq "${DIST_NAME}.zip" "${DIST_NAME}")
        echo "Created ${DIST_NAME}.zip"
    elif command -v 7z >/dev/null 2>&1; then
        7z a -r "${DIST_NAME}.zip" "${DIST_NAME}" >/dev/null
        echo "Created ${DIST_NAME}.zip (using 7z)"
    else
        echo "error: 'zip' or '7z' is required for Windows-style archives" >&2
        echo "  Install zip: https://zip.sourceforge.io/ or use 'choco install zip'" >&2
        echo "  Or install 7-Zip: https://www.7-zip.org/" >&2
        exit 1
    fi
    rm -rf "${DIST_NAME}"
    ls -lh "${DIST_NAME}.zip"
else
    tar -czf "${DIST_NAME}.tar.gz" "${DIST_NAME}"
    rm -rf "${DIST_NAME}"
    ls -lh "${DIST_NAME}.tar.gz"
    echo "Created ${DIST_NAME}.tar.gz"
fi
