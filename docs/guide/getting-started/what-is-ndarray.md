# What is NDArray?

NDArray is a high-performance numerical computing library for PHP, providing N-dimensional arrays backed by Rust. It bridges the gap between PHP's convenience and systems-level performance.

## The Problem with PHP Arrays

PHP arrays are powerful and flexible, but they are designed for general-purpose programming, not numerical computing:

**Memory Overhead**

PHP arrays are implemented as hash tables with significant per-element overhead:

```
PHP Array (per element):
┌─────────────────┐
│ Hash bucket     │  →  Key hash
│ Key pointer     │  →  String/object key
│ Value pointer   │  →  Zval (type info + value)
│ Next pointer    │  →  Linked list
└─────────────────┘
     ↑
     ~80-100 bytes per element

NDArray (per element):
┌─────────────────┐
│ Raw value       │  →  8 bytes (Float64)
└─────────────────┘
     ↑
     Fixed size, no overhead
```

For a 1000×1000 matrix:
- PHP Array: ~80 MB with per-element overhead
- NDArray: ~8 MB contiguous memory

**Performance Characteristics**

```php
// PHP approach: nested loops (slow)
$result = [];
for ($i = 0; $i < 1000; $i++) {
    for ($j = 0; $j < 1000; $j++) {
        $result[$i][$j] = $a[$i][$j] * $b[$i][$j];
    }
}
// 1,000,000 PHP operations, 1,000,000 hash lookups

// NDArray approach: vectorized (fast)
$result = $a->multiply($b);
// Single FFI call → Rust SIMD loop → 1,000,000 operations in compiled code
```

**Type Flexibility vs Type Safety**

PHP arrays allow mixed types:
```php
$mixed = [1, 2.5, "three", true];  // Valid but problematic for math
```

NDArray requires homogeneous data:
```php
$arr = NDArray::array([1, 2, 3, 4, 5]);  // All Int64
$arr = NDArray::array([1.1, 2.2, 3.3]);  // All Float64
```

This enables:
- Predictable memory layout
- SIMD optimizations
- Type-specific operations

## How NDArray Works

**Architecture**

```
PHP Code                    FFI Boundary           Rust Backend
─────────────────────────────────────────────────────────────────
$arr = NDArray::              →                     Allocate
    array([1, 2, 3])                                contiguous
                                                      memory
                              ↓
                        [1][2][3]  (8 bytes each)
                              ↑
$result = $arr->add(10)       →                     SIMD loop
                                                       in Rust
                              ↓
                        [11][12][13]
```

**The View Model**

NDArray uses a view-based architecture for memory efficiency:

```
Original Array:
Shape: [3, 4]
Memory: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
         └────────┬────────┘
                  │
View (slice [:2, :3]):
Shape: [2, 3]
Shares same memory, different metadata (offset, strides)
No data copied!
```

**Zero-Copy Operations**

Many operations return views rather than copies:
- Slicing: `$arr[0:5]` → view
- Transposing: `$arr->transpose()` → view with different strides
- Reshaping: `$arr->reshape([3, 4])` → view (if contiguous)

This means:
- `$slice = $arr[0:1000]` is instant (O(1), not O(n))
- Multiple views of same data don't multiply memory usage
- Changes to views affect the original (unless you `copy()`)

## Key Capabilities

**1. Multi-Dimensional Arrays**

```php
// 1D - Vector
$vector = NDArray::array([1, 2, 3, 4, 5]);

// 2D - Matrix
$matrix = NDArray::array([
    [1, 2, 3],
    [4, 5, 6]
]);

// 3D+ - Tensor
$tensor = NDArray::array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
]);
```

**2. Broadcasting**

Operations work between arrays of different shapes automatically:

```php
$matrix = NDArray::ones([3, 4]);        // Shape [3, 4]
$row = NDArray::array([1, 2, 3, 4]);    // Shape [4]

$result = $matrix->add($row);            // Shape [3, 4]
// Row is broadcast across all 3 rows
```

**3. Element-Wise Operations**

All operations are element-wise by default:

```php
$a = NDArray::array([1, 2, 3]);
$b = NDArray::array([4, 5, 6]);

$a->add($b);        // [5, 7, 9]
$a->multiply($b);   // [4, 10, 18]
$a->gt(2);          // [false, false, true] (comparison)
```

**4. Type System**

Explicit data types for precision control:

```php
use PhpMlKit\NDArray\DType;

$precise = NDArray::array([1.5, 2.5], DType::Float64);  // ~15 digits
$compact = NDArray::array([1.5, 2.5], DType::Float32);  // ~7 digits, half memory
$int8 = NDArray::array([1, 2, 3], DType::Int8);          // Small integers
```

## When to Use NDArray

**Ideal Use Cases:**

- **Scientific Computing**: Large-scale numerical simulations
- **Machine Learning**: Data preprocessing, feature engineering
- **Data Analysis**: Statistical operations on large datasets
- **Image Processing**: Pixel-level operations, filters
- **Signal Processing**: FFT, filtering, transformations

**When NOT to Use NDArray:**

- **Simple collections**: Shopping carts, user lists (use PHP arrays)
- **Key-value storage**: Configurations, settings (use PHP arrays)
- **Tree structures**: Hierarchical data (use objects)
- **Small datasets (<100 elements)**: PHP array overhead is negligible

## Relationship to NumPy

NDArray follows NumPy conventions where practical:

- **Similar API**: `array()`, `zeros()`, `reshape()`, `transpose()`
- **Indexing syntax**: `arr[0:5]`, `arr[:, 0]`
- **Broadcasting rules**: Same shape compatibility rules
- **Data types**: Int64, Float64, Bool, etc.

**Key Differences:**

- **No operator overloading**: PHP limitation → use methods (`$a->add($b)` not `$a + $b`)
- **Rust backend**: Zero Python dependencies, PHP FFI
- **Type hints**: Full PHP type safety with generics

## Next Steps

Now that you understand what NDArray is and why it exists:

1. **[Installation](/guide/getting-started/installation)** - Get it running
2. **[Quick Start](/guide/getting-started/quick-start)** - Your first code
3. **[Understanding Arrays](/guide/fundamentals/understanding-arrays)** - How arrays work
4. **[NumPy Migration](/guide/getting-started/numpy-migration)** - If coming from Python

::: tip
NDArray is designed to feel natural to PHP developers while providing NumPy-like power. You don't need to understand the Rust internals—just know that the heavy lifting happens there.
:::

---

**Key Concepts to Remember:**

1. **Homogeneous data**: All elements same type
2. **Contiguous memory**: Efficient storage and access
3. **Views**: Share memory, instant operations
4. **Rust backend**: Performance without the complexity