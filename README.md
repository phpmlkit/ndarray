<p align="center">
  <img src="docs/public/logo.png" width="200" height="200" alt="NDArray PHP Logo">
</p>

<h1 align="center">NDArray PHP</h1>

<p align="center">High-performance N-dimensional arrays for PHP, powered by Rust via FFI</p>

<p align="center">
  <a href="https://packagist.org/packages/phpmlkit/ndarray"><img src="https://img.shields.io/packagist/v/phpmlkit/ndarray?style=flat-square" alt="Latest Version"></a>
  <a href="https://github.com/phpmlkit/ndarray/actions"><img src="https://img.shields.io/github/actions/workflow/status/phpmlkit/ndarray/tests.yml?branch=main&label=tests&style=flat-square" alt="GitHub Workflow Status"></a>
  <a href="https://packagist.org/packages/phpmlkit/ndarray"><img src="https://img.shields.io/packagist/dt/phpmlkit/ndarray?style=flat-square" alt="Total Downloads"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
</p>

## Features

- **Blazing Fast** - Zero-copy views, Rust backend, minimal FFI overhead
- **NumPy Compatible** - Familiar API for Python developers
- **Memory Safe** - Automatic memory management via Rust
- **N-Dimensional** - Support for arbitrary dimensions
- **Type Safe** - Multiple data types with automatic inference
- **Scientific Computing** - Complete toolkit for ML and data science

## Installation

```bash
composer require phpmlkit/ndarray
```

**Requirements:** PHP 8.1+ with FFI extension enabled

## Quick Example

```php
use PhpMlKit\NDArray\NDArray;

// Create arrays
$a = NDArray::array([[1, 2], [3, 4]]);
$b = NDArray::ones([2, 2]);

// Mathematical operations
$c = $a->add($b);
$d = $a->matmul($b);
$e = $a->sum();

// Slicing with zero-copy views
$first_row = $a[0];  // View: no data copied!
$sub_matrix = $a->slice(['0:1', '0:1']);

// Linear algebra
$dot = $a->dot($b);
$trace = $a->trace();
```

## Why NDArray?

### Zero-Copy Views

Work with large datasets efficiently:

```php
$data = NDArray::random([10000, 1000]);  // 80 MB array
$batch = $data->slice(['0:1000']);       // View: 0 bytes copied!
$mean = $batch->mean();                  // Process on view
```

### NumPy Familiarity

Coming from Python? You'll feel at home (with PHP syntax):

| NumPy | NDArray PHP |
|-------|-------------|
| `np.array([1, 2, 3])` | `NDArray::array([1, 2, 3])` |
| `arr.shape` | `$arr->shape()` |
| `arr.sum()` | `$arr->sum()` |
| `a + b` | `$a->add($b)` |
| `a[0, 0]` | `$a['0,0']` or `$a->get(0, 0)` |
| `a[0:5]` | `$a->slice(['0:5'])` |

## Documentation

- [Full Documentation](https://phpmlkit.github.io/ndarray/)
- [Installation Guide](https://phpmlkit.github.io/ndarray/guide/getting-started/installation)
- [Quick Start](https://phpmlkit.github.io/ndarray/guide/getting-started/quick-start)
- [NumPy Migration](https://phpmlkit.github.io/ndarray/guide/getting-started/numpy-migration)
- [API Reference](https://phpmlkit.github.io/ndarray/api)

## Key Concepts

### Views vs Copies

**Views** share memory with the parent array (zero-copy):

```php
$arr = NDArray::array([[1, 2], [3, 4]]);
$row = $arr[0];      // View
$row[0] = 999; // Modifies $arr!
```

**Copies** are independent:

```php
$copy = $arr->copy();
$copy->set([0, 0], 999);  // Doesn't modify $arr
```

**Operations** create copies:

```php
$result = $arr->multiply(2);  // Copy - original unchanged
```

### Important PHP Syntax Differences

PHP does NOT support operator overloading. **Use method calls:**

```php
// ❌ This doesn't work in PHP
$c = $a + $b;
$c = $a * 2;

// ✅ Use method calls instead
$c = $a->add($b);
$c = $a->multiply(2);
```

Multi-dimensional indexing uses strings:

```php
// ❌ Invalid PHP syntax
$value = $matrix[0, 0];

// ✅ Use string syntax
$value = $matrix['0,0'];
// or
$value = $matrix->get(0, 0);
```

Slicing uses string syntax or method calls:

```php
// ❌ Invalid PHP syntax
$slice = $arr[0:5];

// ✅ Use string syntax
$slice = $arr['0:5'];
// or
$slice = $arr->slice(['0:5']);
```

## Supported Data Types

- **Integers:** `Int8`, `Int16`, `Int32`, `Int64`
- **Unsigned:** `UInt8`, `UInt16`, `UInt32`, `UInt64`
- **Floats:** `Float32`, `Float64`
- **Boolean:** `Bool`

## Supported Operations

- ✅ Array creation (zeros, ones, random, arange, linspace)
- ✅ Indexing and slicing (multi-dimensional, negative indices, steps)
- ✅ Views and copies (zero-copy slicing, transpose)
- ✅ Arithmetic operations (add, subtract, multiply, divide, mod, power)
- ✅ Mathematical functions (abs, sqrt, exp, log, trig, etc.)
- ✅ Reductions (sum, mean, std, min, max, argmin, argmax)
- ✅ Linear algebra (dot, matmul, trace, diagonal)
- ✅ Shape manipulation (reshape, transpose, squeeze, expandDims)
- ✅ Comparisons and boolean operations
- ✅ Sorting and searching

## Development

```bash
# Clone repository
git clone https://github.com/phpmlkit/ndarray.git
cd ndarray

# Build the rust library
./scripts/build.sh debug

# Install dependencies
composer install

# Run tests
composer test

# Run static analysis
composer lint

# Format code
composer cs:fix
```

## Documentation Development

```bash
# Install Node dependencies
npm install

# Start docs dev server
npm run docs:dev

# Build docs
npm run docs:build
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Credits

Created by [CodeWithKyrian](https://github.com/codewithkyrian)

Powered by [Rust ndarray](https://github.com/rust-ndarray/ndarray)

---

⭐ **Star this repo if you find it helpful!**
