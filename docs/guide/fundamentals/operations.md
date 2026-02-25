# Operations

Overview of the operations available on NDArrays.

NDArray provides a comprehensive set of operations for numerical computing. This page provides a conceptual overview—see the [API Reference](/api/) for complete method details.

## Operation Categories

### 1. Arithmetic Operations

Element-wise operations between arrays or arrays and scalars:

```php
$a = NDArray::array([1, 2, 3]);
$b = NDArray::array([4, 5, 6]);

// Basic arithmetic
$a->add($b);           // [5, 7, 9]
$a->subtract($b);     // [-3, -3, -3]
$a->multiply($b);     // [4, 10, 18]
$a->divide($b);       // [0.25, 0.4, 0.5]

// With scalars (broadcast automatically)
$a->add(10);          // [11, 12, 13]
$a->multiply(2);      // [2, 4, 6]
```

**Key Concept:** All arithmetic is element-wise. Broadcasting applies automatically when shapes differ.

**Available Operations:**
- `add()`, `subtract()`, `multiply()`, `divide()`
- `rem()`, `mod()` - Remainder/modulo
- `abs()`, `negative()` - Absolute value and negation
- `pow2()`, `powi()`, `powf()` - Power operations
- `sqrt()`, `cbrt()` - Roots

See [Mathematical Functions API](/api/mathematical-functions) for complete reference.

### 2. Comparison Operations

Compare arrays element-wise, returning boolean masks:

```php
$scores = NDArray::array([85, 92, 78, 95, 88]);

// Create boolean mask
$passing = $scores->gte(80);  // [true, true, false, true, true]

// Combine comparisons
$range = $scores->gte(80)->and($scores->lte(90));
// [true, false, false, false, true]
```

**Key Concept:** Comparisons return boolean arrays (masks) useful for filtering.

**Available Operations:**
- `eq()` - Equal (==)
- `ne()` - Not equal (!=)
- `gt()`, `gte()` - Greater than (>), greater or equal (>=)
- `lt()`, `lte()` - Less than (<), less or equal (<=)

See [Logic Functions API](/api/logic-functions) for complete reference.

### 3. Mathematical Functions

Element-wise mathematical functions:

```php
$angles = NDArray::array([0, M_PI/2, M_PI]);

// Trigonometry
$angles->sin();   // [0.0, 1.0, 0.0]
$angles->cos();   // [1.0, 0.0, -1.0]

// Exponentials and logarithms
$values = NDArray::array([1, 2, 3]);
$values->exp();   // [2.718, 7.389, 20.086]
$values->log();   // [0.0, 0.693, 1.099]

// Rounding
$nums = NDArray::array([1.2, 2.5, 3.7]);
$nums->floor();   // [1.0, 2.0, 3.0]
$nums->cell();    // [2.0, 3.0, 4.0]
$nums->round();   // [1.0, 2.0, 4.0]
```

**Categories:**
- **Trigonometric:** `sin()`, `cos()`, `tan()`, `asin()`, `acos()`, `atan()`
- **Hyperbolic:** `sinh()`, `cosh()`, `tanh()`
- **Exponential/Log:** `exp()`, `exp2()`, `log()`, `log2()`, `log10()`
- **Rounding:** `floor()`, `ceil()`, `round()`
- **Other:** `sigmoid()`, `softmax()`, `clamp()`

See [Mathematical Functions API](/api/mathematical-functions) for complete reference.

### 4. Logical Operations

Combine boolean arrays (works on any type, converts to bool):

```php
$a = NDArray::array([true, false, true]);
$b = NDArray::array([true, true, false]);

$a->and($b);   // [true, false, false]
$a->or($b);    // [true, true, true]
$a->not();     // [false, true, false]
$a->xor($b);   // [false, true, true]

// Also works on non-boolean types (truthy/falsy)
$nums = NDArray::array([5, 0, 3]);
$nums->not();  // [false, true, false]
```

**Key Concept:** Logical operations always return boolean arrays, regardless of input type.

See [Logic Functions API](/api/logic-functions) for complete reference.

### 5. Reduction Operations

Aggregate values across dimensions:

```php
$matrix = NDArray::array([
    [1, 2, 3],
    [4, 5, 6]
]);

// Sum all elements
$matrix->sum();       // 21

// Sum along axis 0 (columns)
$matrix->sum(axis: 0);  // [5, 7, 9]

// Sum along axis 1 (rows)
$matrix->sum(axis: 1);  // [6, 15]

// Keep dimensions
$matrix->sum(axis: 1, keepdims: true);  // [[6], [15]]
```

**Available Reductions:**
- `sum()` - Sum of elements
- `mean()` - Arithmetic mean
- `std()`, `var()` - Standard deviation and variance
- `min()`, `max()` - Minimum and maximum
- `product()` - Product of elements

See [Statistics API](/api/statistics) for complete reference.

### 6. Linear Algebra

Matrix and vector operations:

```php
$a = NDArray::array([[1, 2], [3, 4]]);
$b = NDArray::array([[5, 6], [7, 8]]);

// Element-wise multiplication (not matrix multiplication!)
$a->multiply($b);   // [[5, 12], [21, 32]]

// Matrix multiplication
$a->dot($b);        // [[19, 22], [43, 50]]
$a->matmul($b);     // Same as dot() for 2D

// Vector dot product
$v1 = NDArray::array([1, 2, 3]);
$v2 = NDArray::array([4, 5, 6]);
$v1->dot($v2);      // 32 (scalar)
```

**Important Distinction:**
- `$a->multiply($b)` - Element-wise (Hadamard product)
- `$a->dot($b)` - Matrix multiplication

**Available Operations:**
- `dot()`, `matmul()` - Matrix/vector multiplication
- `transpose()` - Transpose matrix
- `diagonal()` - Extract diagonal
- `trace()` - Sum of diagonal
- `norm()` - Vector/matrix norms

See [Linear Algebra API](/api/linear-algebra) for complete reference.

### 7. Shape Manipulation

Change array shape and layout:

```php
$arr = NDArray::arange(12);  // [0, 1, 2, ..., 11]

// Reshape
$matrix = $arr->reshape([3, 4]);
// [[0, 1, 2, 3],
//  [4, 5, 6, 7],
//  [8, 9, 10, 11]]

// Transpose
$matrix->transpose();
// [[0, 4, 8],
//  [1, 5, 9],
//  [2, 6, 10],
//  [3, 7, 11]]

// Flatten to 1D
$matrix->flatten();  // Copy
$matrix->ravel();    // View if possible

// Add/remove dimensions
$arr = NDArray::array([1, 2, 3]);
$arr->expandDims(0);   // Shape [1, 3]
$arr->squeeze();        // Remove size-1 dimensions
```

**Views vs Copies:**
- `reshape()`, `transpose()`, `expandDims()`, `squeeze()` - Return views when possible
- `flatten()` - Always returns a copy
- `ravel()` - Returns view if contiguous, otherwise copy

See [Array Manipulation API](/api/array-manipulation) for complete reference.

## How Operations Work

### Broadcasting

When operating on arrays of different shapes, NDArray broadcasts them to a compatible shape:

```php
$matrix = NDArray::ones([3, 4]);      // Shape [3, 4]
$row = NDArray::array([1, 2, 3, 4]);  // Shape [4]

// Broadcasting: row is stretched to [3, 4]
$result = $matrix->add($row);
// [[2, 3, 4, 5],
//  [2, 3, 4, 5],
//  [2, 3, 4, 5]]
```

See [Broadcasting](/guide/fundamentals/broadcasting) for complete rules.

### Type Promotion

When operating on arrays of different types, the result type is the more general type:

```php
$int = NDArray::array([1, 2, 3]);
$float = NDArray::array([1.5, 2.5, 3.5]);

$result = $int->add($float);
$result->dtype();  // Float64 (not Int64)
```

**Promotion Hierarchy:** Bool → Int → Float

### In-Place Operations

Most operations return new arrays. For large arrays, use in-place methods to save memory:

```php
$arr = NDArray::ones([1000, 1000]);

// Creates new array
$new = $arr->add(5);  // Memory: +8MB

// In-place (modifies existing)
$arr->addInPlace(5);  // Memory: unchanged
```

## Common Patterns

### Normalizing Data

```php
$data = NDArray::array([10, 20, 30, 40, 50]);

// Min-max normalization to [0, 1]
$min = $data->min();
$max = $data->max();
$normalized = $data->subtract($min)->divide($max->subtract($min));
// [0.0, 0.25, 0.5, 0.75, 1.0]

// Z-score normalization
$mean = $data->mean();
$std = $data->std();
$zscore = $data->subtract($mean)->divide($std);
```

### Filtering with Masks

```php
$data = NDArray::array([10, 25, 30, 45, 50, 65, 70]);

// Find values in range [30, 50]
$mask = $data->gte(30)->and($data->lte(50));
$result = NDArray::where($mask, $data, NDArray::zeros([7]));
// [0, 0, 30, 45, 50, 0, 0]
```

### Row and Column Operations

```php
$matrix = NDArray::array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]);

// Subtract row means (normalize each row)
$rowMeans = $matrix->mean(axis: 1, keepdims: true);
$centered = $matrix->subtract($rowMeans);

// Divide by column std (standardize each column)
$colStd = $matrix->std(axis: 0, keepdims: true);
$standardized = $matrix->divide($colStd);
```

## Next Steps

- **[Indexing and Slicing](/guide/fundamentals/indexing-and-slicing)** - Access array elements
- **[Broadcasting](/guide/fundamentals/broadcasting)** - Understanding shape compatibility
- **[Printing Arrays](/guide/fundamentals/printing)** - Controlling array display
- **[API Reference](/api/)** - Complete method documentation