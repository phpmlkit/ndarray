# Understanding Arrays

This guide explains how NDArrays work conceptually—their structure, creation, and properties.

## Array Structure

An NDArray consists of two parts:

1. **The Data**: Contiguous block of memory storing values
2. **The Metadata**: Shape, strides, data type, and offset

```
┌─────────────────────────────────────────────────┐
│                 NDArray Object                   │
├─────────────────────────────────────────────────┤
│  Shape: [3, 4]          ← Dimensions            │
│  Dtype: Float64         ← Element type          │
│  Strides: [32, 8]       ← Byte steps            │
│  Offset: 0              ← Start position        │
│  Handle: *mut c_void    ← Pointer to Rust data  │
└─────────────────────────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────┐
│              Memory Layout (C-order)            │
│                                                  │
│  [0,0] [0,1] [0,2] [0,3] [1,0] [1,1] ...       │
│    1.0   2.0   3.0   4.0   5.0   6.0           │
│                                                  │
│  Contiguous: Element (i,j) is at:               │
│  address = base + i×32 + j×8                    │
└─────────────────────────────────────────────────┘
```

**Why This Matters**

The metadata allows NDArrays to be views into existing data without copying:

```
Original Array (shape [3, 4]):
┌───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │
├───┼───┼───┼───┤
│ 5 │ 6 │ 7 │ 8 │
├───┼───┼───┼───┤
│ 9 │10 │11 │12 │
└───┴───┴───┴───┘

Slice [:2, :3] (shape [2, 3], same memory):
┌───┬───┬───┐
│ 1 │ 2 │ 3 │
├───┼───┼───┤
│ 5 │ 6 │ 7 │
└───┴───┴───┘
         ↑
    Only metadata changed!
    Offset=0, Strides=[32, 8]
```

## Creating Arrays

### From PHP Arrays

NDArray infers the shape and data type from PHP arrays:

```php
use PhpMlKit\NDArray\NDArray;

// 1D - Vector (shape [5])
$vector = NDArray::array([1, 2, 3, 4, 5]);

// 2D - Matrix (shape [3, 3])
$matrix = NDArray::array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]);

// 3D - Tensor (shape [2, 2, 3])
$tensor = NDArray::array([
    [[1, 2, 3], [4, 5, 6]],
    [[7, 8, 9], [10, 11, 12]]
]);
```

**Type Inference Rules:**
- Integers → `Int64`
- Floats → `Float64`
- Booleans → `Bool`

**Explicit Types:**

```php
use PhpMlKit\NDArray\DType;

// Force specific types
$int32 = NDArray::array([1, 2, 3], DType::Int32);
$float32 = NDArray::array([1.5, 2.5], DType::Float32);
```

### Factory Methods

**Initialize with Values:**

```php
// Zeros - useful for initialization
$zeros = NDArray::zeros([3, 4]);           // [[0, 0, 0, 0], ...]

// Ones - useful for masks
$ones = NDArray::ones([2, 3]);             // [[1, 1, 1], ...]

// Full - specific value
$filled = NDArray::full([3, 3], 42);       // All elements are 42

// Identity matrix
$identity = NDArray::eye(3);               // 3×3 identity matrix
```

**Sequences:**

```php
// Range (like PHP range())
$arr = NDArray::arange(0, 10);             // [0, 1, 2, ..., 9]
$arr = NDArray::arange(0, 10, 2);          // [0, 2, 4, 6, 8] (step=2)

// Linear spacing
$arr = NDArray::linspace(0, 1, 5);         // [0, 0.25, 0.5, 0.75, 1]

// Logarithmic spacing
$arr = NDArray::logspace(0, 2, 5);         // 10^0 to 10^2, 5 points
```

**Random Values:**

```php
// Uniform [0, 1)
$random = NDArray::random([3, 3]);

// Normal distribution (mean=0, std=1)
$normal = NDArray::randn([100, 100]);
```

## Array Properties

Every array has metadata describing its structure:

### Shape

The shape is a tuple of dimension sizes:

```php
$arr = NDArray::array([
    [1, 2, 3],
    [4, 5, 6]
]);

$arr->shape();  // [2, 3]
                 // 2 rows, 3 columns
```

**Visualizing Dimensions:**

```
Shape [3]          Shape [2, 3]          Shape [2, 2, 3]
   ↓                    ↓                      ↓
[1, 2, 3]          ┌───┬───┬───┐         [[[1, 2, 3],
                     │ 1 │ 2 │ 3 │           [4, 5, 6]],
                     ├───┼───┼───┤          [[7, 8, 9],
                     │ 4 │ 5 │ 6 │           [10, 11, 12]]]
                     └───┴───┴───┘
```

### Number of Dimensions (ndim)

```php
$vector = NDArray::array([1, 2, 3]);
$vector->ndim();  // 1

$matrix = NDArray::array([[1, 2], [3, 4]]);
$matrix->ndim();  // 2
```

### Size and Memory

```php
$arr = NDArray::zeros([1000, 1000], DType::Float64);

$arr->size();      // 1000000 (total elements)
$arr->itemsize();  // 8 (bytes per element)
$arr->nbytes();    // 8000000 (8 MB total)
```

**Memory Layout Visualization:**

```
Float64 Array [3, 4]:
┌────────────────────────────────────────────┐
│ Index │ Row │ Col │ Value │ Memory Offset │
├────────────────────────────────────────────┤
│   0   │  0  │  0  │  1.0  │      0        │
│   1   │  0  │  1  │  2.0  │      8        │  ← +8 bytes (Float64)
│   2   │  0  │  2  │  3.0  │     16        │
│   3   │  0  │  3  │  4.0  │     24        │
│   4   │  1  │  0  │  5.0  │     32        │  ← New row, +32 bytes
│   5   │  1  │  1  │  6.0  │     40        │
│  ...  │ ... │ ... │  ...  │    ...        │
└────────────────────────────────────────────┘

Strides: [32, 8]
  - Next row: +32 bytes (4 elements × 8 bytes)
  - Next col: +8 bytes (1 element × 8 bytes)
```

### Data Type (dtype)

```php
$arr = NDArray::array([1, 2, 3]);
$arr->dtype();  // DType::Int64

$floats = NDArray::array([1.5, 2.5]);
$floats->dtype();  // DType::Float64
```

**Common Types:**
- `Int8` to `Int64`: Signed integers
- `UInt8` to `UInt64`: Unsigned integers
- `Float32`, `Float64`: Floating point
- `Bool`: Boolean

## Views vs Owned Arrays

Understanding the distinction is crucial:

**Owned Array:**
- Created by `NDArray::array()`, `zeros()`, `ones()`, etc.
- Manages its own memory
- When destroyed, memory is freed

**View:**
- Created by slicing, transposing, reshaping
- Shares memory with parent array
- When destroyed, memory NOT freed (parent still uses it)

```php
$owned = NDArray::array([1, 2, 3, 4, 5]);
$view = $owned->slice(['1:4']);  // [2, 3, 4]

$view->isView();     // true
$owned->isView();    // false
```

**Important:** Modifying a view modifies the parent:

```php
$parent = NDArray::array([1, 2, 3, 4, 5]);
$view = $parent->slice(['1:4']);

$view->setAt(0, 999);
echo $parent->toArray();  // [1, 999, 3, 4, 5]
```

See [Views vs Copies](/guide/fundamentals/views-vs-copies) for complete details.

## Next Steps

Now that you understand arrays conceptually:

- **[Indexing and Slicing](/guide/fundamentals/indexing-and-slicing)** - Access and modify elements
- **[Data Types](/guide/fundamentals/data-types)** - Choosing the right type
- **[Views vs Copies](/guide/fundamentals/views-vs-copies)** - Memory management
- **[Operations](/guide/fundamentals/operations)** - Working with array data