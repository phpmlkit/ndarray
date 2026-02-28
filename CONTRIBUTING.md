# Contributing to NDArray PHP

Thank you for your interest in contributing to NDArray PHP! This document provides comprehensive guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Features](#suggesting-features)
  - [Pull Requests](#pull-requests)
- [Development Guidelines](#development-guidelines)
  - [PHP Code Style](#php-code-style)
  - [Rust Code](#rust-code)
  - [Testing](#testing)
  - [Documentation](#documentation)
- [Architecture Overview](#architecture-overview)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Release Process](#release-process)

## Code of Conduct

Be respectful and constructive in all interactions. We welcome contributors from all backgrounds and experience levels.

## Getting Started

### Prerequisites

- PHP 8.1 or higher with FFI extension enabled
- Rust toolchain (latest stable)
- Composer
- Node.js (for documentation site)

### Repository Structure

```
ndarray/
├── src/                    # PHP source code
│   ├── FFI/               # FFI bindings and library management
│   ├── Traits/            # Trait implementations for NDArray
│   └── *.php              # Core classes
├── rust/                  # Rust source code
│   └── src/
│       ├── core/          # Core NDArray implementation
│       ├── ffi/           # FFI interface layer
│       └── dtype.rs       # Data type definitions
├── tests/                 # PHP unit tests
├── docs/                  # Documentation (VitePress)
├── scripts/               # Build and utility scripts
├── lib/                   # Compiled Rust libraries (platform-specific)
└── include/              # Generated C headers
```

## Development Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/ndarray.git
cd ndarray

# 2. Install PHP dependencies
composer install

# 3. Install Node dependencies (for docs)
npm install --prefix docs

# 4. Build the Rust library
./scripts/build.sh debug

# 5. Run tests to verify setup
composer test

# 6. Run static analysis
composer lint
```

## How to Contribute

### Reporting Bugs

Before creating a bug report:

1. Check if the bug has already been reported in [Issues](https://github.com/phpmlkit/ndarray/issues)
2. Try to reproduce with the latest version

When creating a bug report, please include:

- **Clear description** of the bug
- **Steps to reproduce** - minimal code example
- **Expected behavior** vs actual behavior
- **Environment details**:
  - PHP version (`php -v`)
  - Platform (Linux/macOS/Windows)
  - NDArray version
- **Error messages** or stack traces

Example:
```php
// Minimal reproduction
$arr = NDArray::array([[1, 2], [3, 4]]);
$result = $arr->matmul($arr);  // Expected: [[7, 10], [15, 22]]
// Actual: Throws ShapeException
```

### Suggesting Features

Feature suggestions are welcome! Please:

1. Check if the feature has already been suggested
2. Explain the use case and why it would be valuable
3. Reference NumPy equivalents if applicable
4. Discuss implementation approach if you have ideas

### Pull Requests

1. **Fork the repository** and create your branch from `main`
   ```bash
   git checkout -b feature/my-new-feature
   # or
   git checkout -b fix/bug-description
   ```

2. **Make your changes**
   - Write clean, focused code
   - Follow the style guidelines below
   - Add tests for new functionality

3. **Test your changes**
   ```bash
   composer test              # Run PHPUnit tests
   composer cs:check          # Check code style
   composer lint              # Run static analysis
   ```

4. **Commit your changes** with clear messages (see [Commit Guidelines](#commit-message-guidelines))

5. **Push to your fork**
   ```bash
   git push origin feature/my-new-feature
   ```

6. **Open a Pull Request** with:
   - Clear title and description
   - Reference any related issues
   - Note any breaking changes
   - Include test results

## Development Guidelines

### PHP Code Style

We follow PSR-12 coding standards. Key points:

- **Use type hints** for all parameters and return types
- **Add PHPDoc** for all public methods with `@param` and `@return`
- **Use descriptive variable names** - avoid abbreviations
- **Keep methods focused** - single responsibility
- **Line length**: 120 characters soft limit
- **Indentation**: 4 spaces (no tabs)

Example:
```php
/**
 * Compute the mean along the specified axis.
 *
 * @param null|int|array<int> $axis     Axis or axes along which to compute
 * @param bool                 $keepdims Keep reduced dimensions
 *
 * @return self Mean values
 */
public function mean(null|int|array $axis = null, bool $keepdims = false): self
{
    // Implementation
}
```

### Rust Code

When modifying Rust code:

- Follow Rust best practices and idioms
- Add `# Safety` documentation for unsafe blocks
- Document all public FFI functions
- Add tests for new FFI functions in `rust/tests/`
- Ensure memory safety - use safe Rust where possible
- Update FFI bindings in `src/FFI/Bindings.php` when adding new functions

Example FFI function:
```rust
/// Compute the sum of elements along axes.
///
/// # Arguments
/// * `handle` - Array handle
/// * `meta` - View metadata
/// * `axes` - Array of axis indices
/// * `num_axes` - Number of axes
/// * `keepdims` - Whether to keep reduced dimensions
/// * `out_handle` - Output handle pointer
///
/// # Safety
/// All pointers must be valid and properly aligned.
#[no_mangle]
pub unsafe extern "C" fn ndarray_sum_axis(...)
```

### Testing

**All PRs must include tests.**

- **Unit tests** in `tests/Unit/` - test individual methods
- **Integration tests** in `tests/Integration/` - test workflows
- **Edge cases** - zero dimensions, empty arrays, type boundaries
- **View tests** - ensure views share memory correctly

Test structure:
```php
public function testSumAxis0(): void
{
    $arr = NDArray::array([[1, 2], [3, 4], [5, 6]]);
    $result = $arr->sum(axis: 0);
    
    $this->assertSame([2], $result->shape());
    $this->assertEquals([9, 12], $result->toArray());
}
```

Run tests:
```bash
composer test              # Run all tests
composer test:pretty       # With readable output
./vendor/bin/phpunit --filter testName  # Specific test
```

### Documentation

Update documentation for all changes:

- **API Reference**: Add/update in `docs/api/`
- **Guides**: Update relevant guide pages in `docs/guide/`
- **Examples**: Add practical examples for new features
- **NumPy Compatibility**: Note NumPy equivalents in migration guide

Documentation workflow:
```bash
# Preview docs locally
cd docs
npm run docs:dev

# Build for production
npm run docs:build
```

## Architecture Overview

### PHP-Rust Bridge

```
PHP Code                    FFI                    Rust Backend
─────────────────────────────────────────────────────────────
$arr->add(5)     ───────▶   ndarray_add      ───────▶  SIMD ops
     │                           │                          │
     │                           ▼                          │
     │                    C ABI boundary                    │
     │                           │                          │
     ▼                           ▼                          ▼
NDArray obj              ViewMetadata              NDArrayWrapper
(handle, shape)          (offset, strides)         (actual data)
```

**Key Principles:**
- **Rust owns the data** - PHP holds opaque pointers
- **Views share memory** - metadata in PHP, data in Rust
- **FFI is one-way** - PHP calls Rust, not vice versa
- **Automatic cleanup** - destructors free Rust memory

### Memory Model

```php
// Root array owns memory
$root = NDArray::random([1000, 1000]);  // Allocated in Rust

// Views share the same handle
$view = $root->slice(['0:100']);  // Different shape/strides, same handle

// When $root is destroyed, Rust memory is freed
// Views keep $root alive via PHP reference counting
```

## Commit Message Guidelines

Follow conventional commits format:

```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, no logic change)
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `test`: Adding/updating tests
- `chore`: Build process, dependencies

**Scopes:**
- `php`: PHP code changes
- `rust`: Rust code changes
- `ffi`: FFI interface changes
- `docs`: Documentation
- `tests`: Test suite

Examples:
```
feat(php): add fromBuffer() for FFI interop

fix(rust): prevent integer overflow in sum operation

docs(api): document all reduction operations

refactor(ffi): simplify error handling in bindings

test(php): add edge case tests for empty arrays
```

## Release Process

For maintainers:

1. Update `CHANGELOG.md` with release notes
2. Update version in `composer.json`
3. Tag the release: `git tag -a v1.0.0 -m "Release version 1.0.0"`
4. Push tags: `git push origin v1.0.0`
5. GitHub Actions will:
   - Build platform binaries
   - Update changelog
   - Deploy documentation
   - Create GitHub release with artifacts

## Questions?

- Open an [Issue](https://github.com/phpmlkit/ndarray/issues) for questions
- Join [Discussions](https://github.com/phpmlkit/ndarray/discussions) for ideas
- Check [Documentation](https://phpmlkit.github.io/ndarray/) for usage help

Thank you for contributing to NDArray PHP! 🎉
