# NDArray Class

The main N-dimensional array class.

## Overview

`NDArray` is the core class of this library. It represents an N-dimensional array backed by Rust via FFI. Each NDArray instance holds:

- A pointer to the underlying Rust array handle
- Metadata about the array (shape, dtype, strides)
- Reference counting information for memory management

An NDArray can represent either:
- **An owned array** - The array owns its data and memory
- **A view of an array** - A slice or view that shares memory with another array

```php
use PhpMlKit\NDArray\NDArray;

// Create an owned array
$arr = NDArray::array([1, 2, 3, 4, 5]);

// Create a view (shares memory)
$view = $arr[0:3];
echo $view->isView();  // true
```

## Properties

Methods to inspect array metadata.

### shape()

Returns the dimensions of the array.

```php
public function shape(): array
```

**Returns:** Array of integers representing size of each dimension

**Examples:**

```php
$arr = NDArray::zeros([3, 4, 5]);
echo $arr->shape();  // [3, 4, 5]

$vector = NDArray::array([1, 2, 3]);
echo $vector->shape();  // [3]
```

---

### ndim()

Returns the number of dimensions.

```php
public function ndim(): int
```

**Returns:** Integer count of dimensions

**Examples:**

```php
$scalar = NDArray::array([1]);
echo $scalar->ndim();  // 1

$matrix = NDArray::zeros([3, 4]);
echo $matrix->ndim();  // 2

$tensor = NDArray::zeros([2, 3, 4, 5]);
echo $tensor->ndim();  // 4
```

---

### size()

Returns the total number of elements.

```php
public function size(): int
```

**Returns:** Total element count

**Examples:**

```php
$arr = NDArray::zeros([3, 4, 5]);
echo $arr->size();  // 60 (3 * 4 * 5)

$vector = NDArray::array([1, 2, 3, 4, 5]);
echo $vector->size();  // 5
```

---

### dtype()

Returns the data type of the array.

```php
public function dtype(): DType
```

**Returns:** DType enum value

**Examples:**

```php
$arr = NDArray::array([1, 2, 3]);
echo $arr->dtype();  // DType::Int64

$int_arr = NDArray::array([1, 2, 3], DType::Int32);
echo $int_arr->dtype();  // DType::Int32
```

**See Also:**
- [Data Types Guide](/guide/fundamentals/data-types)

---

### itemsize()

Returns the size of each element in bytes.

```php
public function itemsize(): int
```

**Returns:** Bytes per element

**Examples:**

```php
$f64 = NDArray::array([1.0], DType::Float64);
echo $f64->itemsize();  // 8

$f32 = NDArray::array([1.0], DType::Float32);
echo $f32->itemsize();  // 4

$i32 = NDArray::array([1], DType::Int32);
echo $i32->itemsize();  // 4
```

---

### nbytes()

Returns the total bytes consumed by the array.

```php
public function nbytes(): int
```

**Returns:** Total bytes (size * itemsize)

**Examples:**

```php
$arr = NDArray::zeros([1000, 1000], DType::Float64);
echo $arr->nbytes();  // 8000000 (8 MB)

// Memory-efficient alternative
$small = NDArray::zeros([1000, 1000], DType::Float32);
echo $small->nbytes();  // 4000000 (4 MB)
```

---

### strides()

Returns the byte steps in each dimension.

```php
public function strides(): array
```

**Returns:** Array of strides

**Examples:**

```php
$arr = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
echo $arr->strides();  // [24, 8]
// 24 bytes to next row, 8 bytes to next column
// (Float64 = 8 bytes per element)
```

::: tip
Strides are used internally for views and non-contiguous arrays.
:::

---

### isView()

Returns whether the array is a view of another array.

```php
public function isView(): bool
```

**Returns:** True if array shares memory with parent

**Examples:**

```php
$arr = NDArray::array([1, 2, 3, 4, 5]);
echo $arr->isView();  // false

$view = $arr[0:3];
echo $view->isView();  // true
```

**See Also:**
- [Views vs Copies](/guide/fundamentals/views-vs-copies)

---

### isContiguous()

Returns whether the array is C-contiguous (row-major).

```php
public function isContiguous(): bool
```

**Returns:** True if elements stored sequentially in memory

**Examples:**

```php
$arr = NDArray::array([[1, 2], [3, 4]]);
echo $arr->isContiguous();  // true

// Transpose changes strides
$transposed = $arr->transpose();
echo $transposed->isContiguous();  // false

// Copy restores contiguity
$copy = $transposed->copy();
echo $copy->isContiguous();  // true
```

---

## String Representation

NDArray implements PHP's `Stringable` interface, allowing arrays to be used in any string context.

### __toString()

Returns a string representation of the array.

```php
public function __toString(): string
```

Uses global print options configured via `setPrintOptions()`. The output includes a header showing the array shape.

**Returns:** String representation of the array

**Examples:**

```php
$arr = NDArray::array([1, 2, 3]);
echo $arr;
// array(3)
// [1 2 3]

$matrix = NDArray::array([[1, 2], [3, 4]]);
echo $matrix;
// array(2, 2)
// [
//  [1 2]
//  [3 4]
// ]
```

**See Also:**
- [Printing Guide](/guide/fundamentals/printing)

---

### NDArray::setPrintOptions()

Configure global print options for all array displays.

```php
public static function setPrintOptions(
    int $threshold = 1000,
    int $edgeitems = 3,
    int $precision = 8
): void
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `threshold` | `int` | 1000 | Maximum elements before truncation |
| `edgeitems` | `int` | 3 | Items to show at each edge when truncating |
| `precision` | `int` | 8 | Decimal places for floating-point numbers |

**Examples:**

```php
// Configure for high precision
NDArray::setPrintOptions(precision: 16);

$values = NDArray::array([1.123456789012345]);
echo $values;
// array(1)
// [1.123456789012345]

// Compact display for large arrays
NDArray::setPrintOptions(threshold: 10, edgeitems: 2);

$arr = NDArray::arange(20);
echo $arr;
// array(20)
// [0 1 ... 18 19]
```

---

### NDArray::getPrintOptions()

Retrieve current print settings.

```php
public static function getPrintOptions(): array
```

**Returns:** Array with keys `threshold`, `edgeitems`, and `precision`

**Examples:**

```php
$options = NDArray::getPrintOptions();
// ['threshold' => 1000, 'edgeitems' => 3, 'precision' => 8]
```

---

### NDArray::resetPrintOptions()

Restore print options to default values.

```php
public static function resetPrintOptions(): void
```

**Examples:**

```php
NDArray::setPrintOptions(precision: 16);
// ... use high precision ...

NDArray::resetPrintOptions();
// Now uses: threshold=1000, edgeitems=3, precision=8
```

---

## Summary

| Property | Returns | Description |
|----------|---------|-------------|
| `shape()` | `array` | Dimensions [rows, cols, ...] |
| `ndim()` | `int` | Number of dimensions |
| `size()` | `int` | Total element count |
| `dtype()` | `DType` | Data type enum |
| `itemsize()` | `int` | Bytes per element |
| `nbytes()` | `int` | Total memory usage |
| `strides()` | `array` | Byte steps per dimension |
| `isView()` | `bool` | Whether shares memory |
| `isContiguous()` | `bool` | Whether row-major |

---

## Next Steps

- [Array Creation](/api/array-creation) - Methods to create arrays
- [Indexing Routines](/api/indexing-routines) - Access and modify elements
- [Mathematical Functions](/api/mathematical-functions) - Arithmetic and math operations
- [Printing Guide](/guide/fundamentals/printing) - Controlling array display