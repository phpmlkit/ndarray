# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial documentation site using VitePress
- Comprehensive guides for installation, quick start, and NumPy migration
- Detailed views vs copies documentation
- Complete API reference for factory methods

## [0.1.0] - 2026-02-21

### Added
- Initial release of NDArray PHP
- Core array class with N-dimensional support
- FFI integration with Rust backend
- Zero-copy views for efficient slicing
- Comprehensive array creation methods
- Mathematical operations (arithmetic, element-wise functions)
- Linear algebra operations (dot, matmul, trace, diagonal)
- Reduction operations (sum, mean, std, min, max)
- Shape manipulation (reshape, transpose, squeeze, expandDims)
- Indexing and slicing with negative indices
- Multiple data types (Int8-64, UInt8-64, Float32/64, Bool)
- Random array generation
- Platform-specific binary distribution

### Features
- 91+ implemented features from SPEC.md
- NumPy-compatible API
- Automatic type inference
- Broadcasting support
- Memory-safe Rust backend
- Cross-platform support (Linux, macOS, Windows)

[Unreleased]: https://github.com/phpmlkit/ndarray/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/phpmlkit/ndarray/releases/tag/v0.1.0
