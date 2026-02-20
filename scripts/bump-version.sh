#!/bin/bash
# Version bump script for NDArray PHP
# Updates both Cargo.toml and composer.json, then creates a git tag

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_error() {
    echo -e "${RED}Error: $1${NC}"
}

print_success() {
    echo -e "${GREEN}$1${NC}"
}

print_info() {
    echo -e "${YELLOW}$1${NC}"
}

# Check if version argument is provided
if [ $# -eq 0 ]; then
    print_error "No version provided"
    echo "Usage: ./scripts/bump-version.sh <version>"
    echo "Example: ./scripts/bump-version.sh 0.1.0"
    exit 1
fi

NEW_VERSION="$1"

# Validate version format (semantic versioning)
if [[ ! $NEW_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$ ]]; then
    print_error "Invalid version format: $NEW_VERSION"
    echo "Expected format: X.Y.Z (e.g., 0.1.0) or X.Y.Z-prerelease"
    exit 1
fi

print_info "Bumping version to: $NEW_VERSION"
echo ""

# Get current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    print_error "Not a git repository"
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    print_error "You have uncommitted changes. Please commit or stash them first."
    echo ""
    echo "Current changes:"
    git status --short
    exit 1
fi

# Update Cargo.toml
CARGO_FILE="rust/Cargo.toml"
if [ -f "$CARGO_FILE" ]; then
    CURRENT_VERSION=$(grep '^version' "$CARGO_FILE" | head -1 | cut -d'"' -f2)
    print_info "Updating $CARGO_FILE: $CURRENT_VERSION → $NEW_VERSION"
    
    # Use sed to update version (works on both macOS and Linux)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/^version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" "$CARGO_FILE"
    else
        sed -i "s/^version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" "$CARGO_FILE"
    fi
    
    print_success "✓ Updated $CARGO_FILE"
else
    print_error "$CARGO_FILE not found"
    exit 1
fi

# Update composer.json
COMPOSER_FILE="composer.json"
if [ -f "$COMPOSER_FILE" ]; then
    # Check if version field exists
    if grep -q '"version"' "$COMPOSER_FILE"; then
        CURRENT_VERSION=$(grep '"version"' "$COMPOSER_FILE" | head -1 | grep -o '"[0-9.]*"' | tail -1 | tr -d '"')
        print_info "Updating $COMPOSER_FILE: $CURRENT_VERSION → $NEW_VERSION"
        
        # Use sed to update version
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s/\"version\": \"$CURRENT_VERSION\"/\"version\": \"$NEW_VERSION\"/" "$COMPOSER_FILE"
        else
            sed -i "s/\"version\": \"$CURRENT_VERSION\"/\"version\": \"$NEW_VERSION\"/" "$COMPOSER_FILE"
        fi
    else
        print_info "Adding version field to $COMPOSER_FILE: $NEW_VERSION"
        
        # Add version after the name field (preserve the existing name)
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "/\"name\":/a\\
  \"version\": \"$NEW_VERSION\"," "$COMPOSER_FILE"
        else
            sed -i "/\"name\":/a\\
  \"version\": \"$NEW_VERSION\"," "$COMPOSER_FILE"
        fi
    fi
    
    print_success "✓ Updated $COMPOSER_FILE"
else
    print_error "$COMPOSER_FILE not found"
    exit 1
fi

echo ""

# Show changes
print_info "Changes made:"
git diff --no-color

echo ""

# Ask for confirmation before committing
read -p "Do you want to commit these changes? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Commit changes
    git add rust/Cargo.toml composer.json
    git commit -m "Bump version to $NEW_VERSION"
    print_success "✓ Committed changes"
    
    echo ""
    print_info "Next steps to complete the release:"
    echo ""
    echo "  1. Push to GitHub:"
    echo "     git push origin main"
    echo ""
    echo "  2. Create a release on GitHub:"
    echo "     - Go to: https://github.com/phpmlkit/ndarray/releases/new"
    echo "     - Tag version: v$NEW_VERSION"
    echo "     - Release title: v$NEW_VERSION"
    echo "     - Publish release"
    echo ""
    echo "  3. The GitHub Actions workflow will automatically:"
    echo "     - Build binaries for all platforms"
    echo "     - Create platform-specific distribution archives"
    echo "     - Upload them to the release"
    echo ""
    print_success "Version bump complete!"
else
    print_info "Changes saved but not committed."
    echo ""
    echo "To commit manually:"
    echo "  git add rust/Cargo.toml composer.json"
    echo "  git commit -m \"Bump version to $NEW_VERSION\""
    echo ""
    echo "Then create a release on GitHub with tag v$NEW_VERSION"
fi
