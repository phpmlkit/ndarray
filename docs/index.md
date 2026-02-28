---
layout: home

hero:
  name: "NDArray"
  text: "PHP"
  tagline: High-performance N-dimensional arrays for PHP, powered by Rust
  image:
    src: /logo.png
    alt: NDArray PHP
  actions:
    - theme: brand
      text: Get Started
      link: /guide/getting-started/what-is-ndarray
    - theme: alt
      text: View on GitHub
      link: https://github.com/phpmlkit/ndarray

features:
  - icon: ⚡
    title: Blazing Fast
    details: Zero-copy views, Rust-powered operations, and FFI integration deliver near-native performance for numerical computing.
  - icon: 🔧
    title: NumPy Compatible
    details: Familiar API inspired by NumPy. Easy migration path for Python developers coming to PHP.
  - icon: 🛡️
    title: Memory Safe
    details: Rust manages all array memory with automatic reference counting. No leaks, no use-after-free.
  - icon: 📊
    title: N-Dimensional
    details: Support for arbitrary dimensions. 1D, 2D, 3D, or ND - all with consistent API and optimal performance.
  - icon: 🎯
    title: Type Safe
    details: Multiple data types supported - int8-64, uint8-64, float32/64, and boolean with automatic inference.
  - icon: 🔬
    title: Scientific Computing
    details: Complete toolkit for linear algebra, statistics, and mathematical operations. Ready for ML and data science.
---

## Quick Example

```php
<?php

use PhpMlKit\NDArray\NDArray;

// Create arrays
$a = NDArray::array([[1, 2], [3, 4]]);
$b = NDArray::ones([2, 2]);

// Mathematical operations (method calls, NOT operators)
$c = $a->add($b);              // Element-wise addition
$d = $a->matmul($b);          // Matrix multiplication
$e = $a->sum();                // Sum all elements

// Slicing with zero-copy views
$first_row = $a[0];            // View: no data copied!
$sub_matrix = $a->slice(['0:1', '0:1']);

// Linear algebra
$dot = $a->dot($b);
$trace = $a->trace();
```

## Why NDArray?

### Performance That Matters

Traditional PHP arrays are powerful but slow for numerical work. NDArray changes the game:

- **10-100x faster** for mathematical operations
- **Zero-copy slicing** - work with subsets without copying data
- **Rust backend** - memory-safe, parallelized operations
- **Minimal FFI overhead** - batch operations to maximize speed

### Views: The Game Changer

```php
// Work with a 1000x1000 matrix subset instantly
$data = NDArray::random([1000, 1000]);  // 8MB array
$batch = $data->slice(['0:32']);       // View: 0 bytes copied!

$batch->set([0, 0], 5);  // Modifies $data!
$mean = $batch->mean();   // Calculate on view
```

### Familiar API

Coming from NumPy? You'll feel right at home (with PHP syntax):

| NumPy | NDArray PHP |
|-------|-------------|
| `np.array([1, 2, 3])` | `NDArray::array([1, 2, 3])` |
| `arr.shape` | `$arr->shape()` |
| `arr.sum()` | `$arr->sum()` |
| `a + b` | `$a->add($b)` |
| `a[0, 0]` | `$a['0,0']` |
| `a[0:5]` | `$a->slice(['0:5'])` |

::: warning PHP Syntax Differences
PHP does NOT support operator overloading or multi-dimensional indexing with commas. **Always use method calls:**

```php
// ❌ This doesn't work
$c = $a + $b;
$value = $matrix[0, 0];

// ✅ Use method calls
$c = $a->add($b);
$value = $matrix['0,0'];
// or
$value = $matrix->get(0, 0);
```
:::

## Installation

```bash
composer require phpmlkit/ndarray
```

Requirements: PHP 8.1+ with FFI extension enabled

## Next Steps

- **[Installation Guide](/guide/getting-started/installation)** - Get up and running
- **[Quick Start](/guide/getting-started/quick-start)** - Your first arrays
- **[NumPy Migration](/guide/getting-started/numpy-migration)** - Coming from Python
- **[Views vs Copies](/guide/fundamentals/views-vs-copies)** - Understanding the magic
