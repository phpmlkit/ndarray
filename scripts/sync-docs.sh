#!/bin/bash

# Sync documentation files from root to docs directory
# This script copies CHANGELOG.md and CONTRIBUTING.md to the docs folder
# for local development and documentation preview.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DOCS_DIR="$ROOT_DIR/docs"

echo "Syncing documentation files..."

# Copy CHANGELOG.md
if [ -f "$ROOT_DIR/CHANGELOG.md" ]; then
    cp "$ROOT_DIR/CHANGELOG.md" "$DOCS_DIR/changelog.md"
    echo "✓ Copied CHANGELOG.md → docs/changelog.md"
else
    echo "✗ CHANGELOG.md not found in root directory"
    exit 1
fi

# Copy CONTRIBUTING.md
if [ -f "$ROOT_DIR/CONTRIBUTING.md" ]; then
    cp "$ROOT_DIR/CONTRIBUTING.md" "$DOCS_DIR/contributing.md"
    echo "✓ Copied CONTRIBUTING.md → docs/contributing.md"
else
    echo "✗ CONTRIBUTING.md not found in root directory"
    exit 1
fi

echo ""
echo "Documentation files synced successfully!"
echo ""
echo "To preview the documentation site locally:"
echo "  cd docs && npm run docs:dev"
echo ""
