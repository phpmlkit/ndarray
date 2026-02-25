# NumPy Migration Guide

Coming from NumPy? This guide helps you transition to NDArray PHP with side-by-side comparisons and key differences.

## Philosophy

NDArray PHP is designed to feel familiar to NumPy users while respecting PHP idioms. The core philosophy:

- **Familiar API**: Method names and behavior match NumPy where possible
- **PHP Idioms**: Uses PHP conventions (static methods, arrow operator, camelCase)
- **Performance**: Zero-copy views, Rust backend, minimal FFI overhead

## Quick Reference

### Creating Arrays

| NumPy | NDArray PHP | Notes |
|-------|-------------|-------|
| `np.array([1, 2, 3])` | `NDArray::array([1, 2, 3])` | Static method in PHP |
| `np.zeros((3, 3))` | `NDArray::zeros([3, 3])` | Shape as array, not tuple |
| `np.ones((2, 4))` | `NDArray::ones([2, 4])` | |
| `np.full((2, 2), 5)` | `NDArray::full([2, 2], 5)` | |
| `np.eye(3)` | `NDArray::eye(3)` | Identity matrix |
| `np.arange(0, 10, 2)` | `NDArray::arange(0, 10, 2)` | |
| `np.linspace(0, 1, 50)` | `NDArray::linspace(0, 1, 50)` | |
| `np.random.rand(3, 3)` | `NDArray::random([3, 3])` | |
| `np.random.randn(3, 3)` | `NDArray::randn([3, 3])` | Standard normal |

### Array Properties

| NumPy | NDArray PHP | Notes |
|-------|-------------|-------|
| `arr.shape` | `$arr->shape()` | Method call in PHP |
| `arr.dtype` | `$arr->dtype()` | Returns DType enum |
| `arr.ndim` | `$arr->ndim()` | |
| `arr.size` | `$arr->size()` | |
| `arr.itemsize` | `$arr->itemsize()` | |
| `arr.nbytes` | `$arr->nbytes()` | |
| `arr.T` | `$arr->transpose()` | No property, use method |

### Multi-dimensional Indexing

| NumPy | NDArray PHP | Notes |
|-------|-------------|-------|
| `arr[0, 0]` | `$arr['0,0']` | Comma-separated string |
| `arr[1, 2]` | `$arr['1,2']` | String syntax |
| `arr[-1, -1]` | `$arr['-1,-1']` | Negative indices in string |
| `arr[0, 0]` | `$arr->get(0, 0)` | Alternative: get() method |
| `arr[i, j]` | `$arr->get($i, $j)` | Using get() method |

**Important:** Python's `$arr[0, 0]` is **invalid PHP syntax**. You must use strings or the `get()` method.

### Single Dimension Indexing

| NumPy | NDArray PHP | Notes |
|-------|-------------|-------|
| `arr[0]` | `$arr[0]` | Same syntax |
| `arr[-1]` | `$arr[-1]` | Negative indexing works |
| `arr[0]` | `$arr->get(0)` | Alternative: get() method |

### Slicing

| NumPy | NDArray PHP | Notes |
|-------|-------------|-------|
| `arr[0:5]` | `$arr->slice(['0:5'])` | Method call with array |
| `arr[0:5]` | `$arr['0:5']` | |
| `arr[::2]` | `$arr->slice(['::2'])` | |
| `arr[:, 0]` | `$arr->slice([':', '0'])` | Multi-dimensional slice |

::: warning Negative Step Not Supported
Reversing with negative step (`arr[::-1]`) is not supported. Use `$arr->flip()` instead.
:::
| `arr[0:5, 0:5]` | `$arr->slice(['0:5', '0:5'])` | |

**Important:** Python's `$arr[0:5]` is **invalid PHP syntax**. You must use strings or the `slice()` method.

### Arithmetic Operations

**Critical Difference:** PHP does NOT support operator overloading!

| NumPy | NDArray PHP | Notes |
|-------|-------------|-------|
| `a + b` | `$a->add($b)` | Method call, NOT operator |
| `a - b` | `$a->subtract($b)` | |
| `a * b` | `$a->multiply($b)` | |
| `a / b` | `$a->divide($b)` | |
| `a % b` | `$a->mod($b)` | |
| `a ** 2` | `$a->power(2)` | |
| `-a` | `$a->negative()` | |
| `a + 5` | `$a->add(5)` | Scalar operations are methods |
| `a @ b` | `$a->matmul($b)` | Matrix multiplication |

::: danger No Operator Overloading
PHP does NOT support operators like `+`, `-`, `*` on objects. **You MUST use method calls!**

```php
// ❌ WRONG - This will cause an error!
$c = $a + $b;

// ✅ CORRECT - Use method calls
$c = $a->add($b);
```
:::

### Mathematical Functions

| NumPy | NDArray PHP | Notes |
|-------|-------------|-------|
| `np.abs(a)` | `$a->abs()` | Method call |
| `np.sqrt(a)` | `$a->sqrt()` | |
| `np.exp(a)` | `$a->exp()` | |
| `np.log(a)` | `$a->log()` | |
| `np.log10(a)` | `$a->log10()` | |
| `np.sin(a)` | `$a->sin()` | Trigonometric |
| `np.cos(a)` | `$a->cos()` | |
| `np.floor(a)` | `$a->floor()` | |
| `np.ceil(a)` | `$a->ceil()` | |
| `np.round(a)` | `$a->round()` | |
| `np.clip(a, 0, 1)` | `$a->clip(0, 1)` | |

### Comparisons

| NumPy | NDArray PHP | Notes |
|-------|-------------|-------|
| `a == b` | `$a->eq($b)` | Method call |
| `a != b` | `$a->ne($b)` | |
| `a > b` | `$a->gt($b)` | |
| `a >= b` | `$a->gte($b)` | |
| `a < b` | `$a->lt($b)` | |
| `a <= b` | `$a->lte($b)` | |

::: danger No Comparison Operators
```php
// ❌ WRONG
$mask = $a > $b;

// ✅ CORRECT
$mask = $a->gt($b);
```
:::

### Reductions

| NumPy | NDArray PHP | Notes |
|-------|-------------|-------|
| `a.sum()` | `$a->sum()` | |
| `a.sum(axis=0)` | `$a->sum(axis: 0)` | Named arguments |
| `a.sum(keepdims=True)` | `$a->sum(keepdims: true)` | Boolean lowercase |
| `a.mean()` | `$a->mean()` | |
| `a.std()` | `$a->std()` | |
| `a.min()` | `$a->min()` | |
| `a.max()` | `$a->max()` | |
| `a.argmin()` | `$a->argmin()` | |
| `a.argmax()` | `$a->argmax()` | |

### Linear Algebra

| NumPy | NDArray PHP | Notes |
|-------|-------------|-------|
| `a.dot(b)` | `$a->dot($b)` | |
| `a @ b` | `$a->matmul($b)` | Matrix multiplication |
| `np.matmul(a, b)` | `$a->matmul($b)` | |
| `a.trace()` | `$a->trace()` | |
| `np.diag(a)` | `$a->diagonal()` | Extract diagonal |
| `a.transpose()` | `$a->transpose()` | |
| `a.T` | `$a->transpose()` | Use method instead |

### Shape Manipulation

| NumPy | NDArray PHP | Notes |
|-------|-------------|-------|
| `a.reshape(3, 4)` | `$a->reshape([3, 4])` | Shape as array |
| `a.flatten()` | `$a->flatten()` | |
| `a.ravel()` | `$a->ravel()` | |
| `a.squeeze()` | `$a->squeeze()` | |
| `np.expand_dims(a, 0)` | `$a->expandDims(0)` | camelCase |
| `np.swapaxes(a, 0, 1)` | `$a->swapaxes(0, 1)` | |
| `a.transpose(1, 0)` | `$a->transpose([1, 0])` | Axes as array |

### Type Conversion

| NumPy | NDArray PHP | Notes |
|-------|-------------|-------|
| `a.astype(np.int32)` | `$a->astype(DType::Int32)` | Enum instead of constant |
| `a.tolist()` | `$a->toArray()` | Convert to PHP array |
| `float(a[0])` | `$a[0]` (returns scalar) | Already scalar |
| `print(a)` | `print_r($a->toArray())` | Must convert to array |

### Printing/Display

| NumPy | NDArray PHP | Notes |
|-------|-------------|-------|
| `print(a)` | `print_r($a->toArray())` | Convert to array first |
| `print(a[0])` | `print_r($a[0])` | Scalars print directly |
| `a.shape` | `$a->shape()` | Method, not property |

## Key Differences

### 1. Static Factory Methods

**NumPy:**
```python
import numpy as np
arr = np.array([1, 2, 3])
zeros = np.zeros((3, 3))
```

**NDArray PHP:**
```php
use PhpMlKit\NDArray\NDArray;

$arr = NDArray::array([1, 2, 3]);
$zeros = NDArray::zeros([3, 3]);
```

### 2. Arrow Operator vs Dot

**NumPy:**
```python
shape = arr.shape
dtype = arr.dtype
mean = arr.mean()
```

**NDArray PHP:**
```php
$shape = $arr->shape();
$dtype = $arr->dtype();
$mean = $arr->mean();
```

### 3. No Operator Overloading in PHP

**NumPy:**
```python
result = a + b          # Operator
c = a * 2               # Scalar operator
result = a @ b          # Matrix multiplication operator
```

**NDArray PHP:**
```php
$result = $a->add($b);           // Method call
$c = $a->multiply(2);            // Scalar method call
$result = $a->matmul($b);        // Matrix multiplication method
```

### 4. Named Arguments

**NumPy:**
```python
result = arr.sum(axis=0, keepdims=True)
```

**NDArray PHP:**
```php
$result = $arr->sum(axis: 0, keepdims: true);
```

### 5. Shape as Arrays vs Tuples

**NumPy:**
```python
arr = np.zeros((3, 4))  # Tuple
reshaped = arr.reshape(2, 6)  # Multiple args
```

**NDArray PHP:**
```php
$arr = NDArray::zeros([3, 4]);  // Array
$reshaped = $arr->reshape([2, 6]);  // Single array argument
```

### 6. DType Enum vs Constants

**NumPy:**
```python
arr = np.array([1, 2, 3], dtype=np.int32)
```

**NDArray PHP:**
```php
use PhpMlKit\NDArray\DType;

$arr = NDArray::array([1, 2, 3], DType::Int32);
```

### 7. Multi-dimensional Indexing

**NumPy:**
```python
value = matrix[0, 0]  # Comma in brackets
```

**NDArray PHP:**
```php
// ❌ Invalid PHP syntax!
// $value = $matrix[0, 0];

// ✅ Use string syntax
$value = $matrix['0,0'];

// ✅ Or use get() method
$value = $matrix->get(0, 0);
```

### 8. Slicing

**NumPy:**
```python
slice = arr[0:5]      # Colon in brackets
```

**NDArray PHP:**
```php
// ❌ Invalid PHP syntax!
// $slice = $arr[0:5];

// ✅ Use strings
$slice = $arr['0:5'];

// ✅ Or slice() method
$slice = $arr->slice(['0:5']);

// Multi-dimensional
$sub = $matrix->slice(['0:2', '0:2']);
```

### 9. Comparison Operators

**NumPy:**
```python
mask = a > 2          # Comparison operator
```

**NDArray PHP:**
```php
// ❌ Invalid - no operator overloading!
// $mask = $a > 2;

// ✅ Use method
$mask = $a->gt(2);
```

### 10. Views vs Copies

Both libraries use views for slicing, but NDArray makes this even more explicit:

**NumPy:**
```python
a = np.array([1, 2, 3, 4, 5])
b = a[0:3]  # View
b[0] = 999  # Modifies a!
```

**NDArray PHP:**
```php
$a = NDArray::array([1, 2, 3, 4, 5]);
$b = $a->slice(['0:3']);  // View
$b->set([0], 999);         // Modifies $a!
```

::: warning Important
NDArray operations on views create copies. Only assignments modify the original:
```php
$b = $a->slice(['0:3']);
$b = $b->multiply(2);   // Creates copy, $a unchanged
$b->set([0], 999);         // Assignment modifies $a
```
:::

## Complete Migration Examples

### Example 1: Data Normalization

**NumPy:**
```python
import numpy as np

data = np.random.randn(100, 3)
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
normalized = (data - mean) / std
```

**NDArray PHP:**
```php
use PhpMlKit\NDArray\NDArray;

$data = NDArray::randn([100, 3]);
$mean = $data->mean(axis: 0);
$std = $data->std(axis: 0);
$normalized = $data->subtract($mean)->divide($std);
```

### Example 2: Matrix Operations

**NumPy:**
```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Matrix multiplication
c = a @ b

# Transpose
d = a.T

# Diagonal
diag = np.diag(a)
```

**NDArray PHP:**
```php
$a = NDArray::array([[1, 2], [3, 4]]);
$b = NDArray::array([[5, 6], [7, 8]]);

// Matrix multiplication
$c = $a->matmul($b);

// Transpose
$d = $a->transpose();

// Diagonal
$diag = $a->diagonal();
```

### Example 3: Image Processing Pattern

**NumPy:**
```python
# Batch processing: 100 images, 32x32 pixels, 3 channels
images = np.random.rand(100, 32, 32, 3)

# Normalize each channel
mean = np.mean(images, axis=(0, 1, 2))
std = np.std(images, axis=(0, 1, 2))
normalized = (images - mean) / std

# Extract first 10 images
batch = images[0:10]
```

**NDArray PHP:**
```php
// Batch processing: 100 images, 32x32 pixels, 3 channels
$images = NDArray::random([100, 32, 32, 3]);

// Normalize each channel
$mean = $images->mean(axis: [0, 1, 2]);
$std = $images->std(axis: [0, 1, 2]);
$normalized = $images->subtract($mean)->divide($std);

// Extract first 10 images (zero-copy view!)
$batch = $images['0:10'];
```

## Common Pitfalls

### 1. Forgetting Method Calls

```php
// ❌ Wrong
$shape = $arr->shape;  // Property access

// ✅ Correct
$shape = $arr->shape();  // Method call
```

### 2. Using Invalid Indexing Syntax

```php
// ❌ Wrong - Invalid PHP syntax
$value = $matrix[0, 0];

// ✅ Correct
$value = $matrix['0,0'];
// or
$value = $matrix->get(0, 0);
```

### 3. Using Invalid Slicing Syntax

```php
// ❌ Wrong - Invalid PHP syntax
$slice = $arr[0:5];

// ✅ Correct
$slice = $arr['0:5'];
// or
$slice = $arr->slice(['0:5']);
```

### 4. Trying to Use Operators

```php
// ❌ Wrong - PHP doesn't support operator overloading
$result = $a + $b;

// ✅ Correct
$result = $a->add($b);
```

### 5. Trying to Print Directly

```php
// ❌ Wrong
print_r($arr);

// ✅ Correct
print_r($arr->toArray());
```

### 6. Boolean Values

```php
// ❌ Wrong (Python syntax)
$arr = $arr->sum(keepdims: True);

// ✅ Correct (PHP syntax)
$arr = $arr->sum(keepdims: true);
```

## Performance Tips

### 1. Use Views for Subsets

```php
// Fast: Zero-copy view
$data = NDArray::random([1000, 1000]);
$batch = $data->slice(['0:32']);  // 0 bytes copied!
```

### 2. Avoid PHP Loops

```php
// ❌ Slow: PHP loop
$result = NDArray::zeros([1000]);
for ($i = 0; $i < 1000; $i++) {
    $result->set([$i], $arr->get($i) * 2);
}

// ✅ Fast: Vectorized operation
$result = $arr->multiply(2);
```

### 3. Chain Operations

```php
// Efficient: Single FFI call chain
$result = $arr->abs()->sqrt()->clip(0, 10);
```

## Getting Help

- **Documentation**: Full API reference and guides
- **GitHub Issues**: Report bugs or request features
- **Examples**: Cookbook-style examples for common tasks

## Next Steps

- **[Quick Start](/guide/getting-started/quick-start)** - Practice with basic operations
- **[Views vs Copies](/guide/fundamentals/views-vs-copies)** - Master zero-copy operations
- **[API Reference](/api/ndarray-class)** - Complete method listing
