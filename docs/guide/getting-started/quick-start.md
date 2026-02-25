# Quick Start

Get up and running with NDArray in 5 minutes. This guide covers the essentials to start working with N-dimensional arrays.

## Your First Array

```php
<?php

require_once 'vendor/autoload.php';

use PhpMlKit\NDArray\NDArray;

// Create an array from a PHP array
$arr = NDArray::array([1, 2, 3, 4, 5]);

// Convert to PHP array to view/print
print_r($arr->toArray());
// Output: [1, 2, 3, 4, 5]
```

## Array Creation Methods

NDArray provides multiple ways to create arrays:

### From PHP Arrays

```php
// 1D array
$vector = NDArray::array([1, 2, 3, 4, 5]);

// 2D array (matrix)
$matrix = NDArray::array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]);

// 3D array
$cube = NDArray::array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
]);
```

### Factory Functions

```php
$zeros = NDArray::zeros([3, 3]);        // 3x3 matrix of zeros
$ones = NDArray::ones([2, 4]);         // 2x4 matrix of ones
$full = NDArray::full([2, 2], 5);      // 2x2 matrix filled with 5

$identity = NDArray::eye(3);  // 3x3 identity matrix
// [[1. 0. 0.]
//  [0. 1. 0.]
//  [0. 0. 1.]]
```

### Sequences and Ranges

```php
// arange: evenly spaced values
$seq = NDArray::arange(0, 10);     // [0 1 2 3 4 5 6 7 8 9]
$step = NDArray::arange(0, 10, 2); // [0 2 4 6 8]

// linspace: linear spacing
$linear = NDArray::linspace(0, 1, 5);  // [0. 0.25 0.5 0.75 1.]

// Random arrays
$random = NDArray::random([3, 3]);         // Uniform [0, 1)
$normal = NDArray::randn([3, 3]);          // Standard normal distribution
$uniform = NDArray::uniform(0, 10, [3, 3]); // Uniform [0, 10)
$integers = NDArray::randomInt(0, 100, [5]); // Random integers
```

## Array Properties

Every array has properties that describe its structure:

```php
$arr = NDArray::array([[1, 2, 3], [4, 5, 6]]);

print_r($arr->shape());    // [2, 3] - dimensions
print_r($arr->ndim());     // 2 - number of dimensions
print_r($arr->size());     // 6 - total elements
print_r($arr->dtype());    // DType::Int64 - data type
print_r($arr->itemsize()); // 8 - bytes per element
print_r($arr->nbytes());   // 48 - total bytes
```

## Basic Operations

### Arithmetic

**Important:** PHP does not support operator overloading. Use method calls:

```php
$a = NDArray::array([1, 2, 3]);
$b = NDArray::array([4, 5, 6]);

// Element-wise operations (method calls)
$sum = $a->add($b);           // [5 7 9]
$diff = $b->subtract($a);     // [3 3 3]
$product = $a->multiply($b);  // [4 10 18]
$quotient = $b->divide($a);   // [4. 2.5 2.]

// Scalar operations
$doubled = $a->multiply(2);   // [2 4 6]
$incremented = $a->add(10);   // [11 12 13]
```

### Mathematical Functions

```php
$arr = NDArray::array([1, 2, 3, 4]);

$sqrt = $arr->sqrt();     // [1. 1.414 1.732 2.]
$exp = $arr->exp();       // [2.718 7.389 20.086 54.598]
$log = $arr->log();       // [0. 0.693 1.099 1.386]
$sin = $arr->sin();       // Sine of each element

// Rounding
$floats = NDArray::array([1.2, 2.5, 3.7]);
$floor = $floats->floor();  // [1. 2. 3.]
$ceil = $floats->ceil();    // [2. 3. 4.]
```

## Indexing and Slicing

### Basic Indexing

```php
$arr = NDArray::array([10, 20, 30, 40, 50]);

print_r($arr[0]);   // 10 - first element
print_r($arr[-1]);  // 50 - last element (negative indexing)
print_r($arr[2]);   // 30
```

### Multi-dimensional Indexing

**Important:** Use comma-separated strings for multi-dimensional indexing:

```php
$matrix = NDArray::array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]);

// Single dimension works with integers
print_r($matrix[0]);        // First row as NDArray

// Multi-dimensional requires comma-separated string
print_r($matrix->get(0, 0));  // 1 - first row, first column (using get method)
print_r($matrix['0,0']);       // 1 - using string syntax
print_r($matrix['1,2']);       // 6 - second row, third column
print_r($matrix['-1,-1']);     // 9 - last row, last column
```

### Slicing

```php
$arr = NDArray::array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

$slice1 = $arr->slice(['0:5']);    // [0 1 2 3 4] - first 5 elements
$slice2 = $arr['2:5'];             // [2 3 4] - elements 2, 3, 4
$slice3 = $arr->slice(['5:']);     // [5 6 7 8 9] - from index 5 to end
$slice4 = $arr['::2'];             // [0 2 4 6 8] - every 2nd element
// Note: Negative step slices like ['::-1'] are not supported.
// Use $arr->flip() to reverse an array.

$matrix = NDArray::random([5, 5]);
$sub_matrix = $matrix->slice(['1:4', '1:4']);  // 3x3 center
```

::: tip Important
Slicing returns **views**, not copies. Modifying a slice modifies the original array!
:::

## Reduction Operations

```php
$arr = NDArray::array([[1, 2, 3], [4, 5, 6]]);

// Sum all elements
$total = $arr->sum();        // 21

// Sum along axis
$row_sum = $arr->sum(axis: 1);  // [6 15] - sum each row
$col_sum = $arr->sum(axis: 0);  // [5 7 9] - sum each column

// Other reductions
$mean = $arr->mean();       // 3.5
$min = $arr->min();         // 1
$max = $arr->max();         // 6
$std = $arr->std();         // Standard deviation

// Keep dimensions
$sums = $arr->sum(axis: 1, keepdims: true);
// [[6]
//  [15]]
```

## Linear Algebra

```php
$a = NDArray::array([[1, 2], [3, 4]]);
$b = NDArray::array([[5, 6], [7, 8]]);

// Matrix multiplication
$product = $a->matmul($b);

// Dot product
$dot = $a->dot($b);

// Transpose
$transposed = $a->transpose();
// [[1 3]
//  [2 4]]

// Trace (sum of diagonal)
$trace = $a->trace();  // 1 + 4 = 5

// Extract diagonal
$diag = $a->diagonal();  // [1 4]
```

## Shape Manipulation

```php
$arr = NDArray::array([[1, 2, 3], [4, 5, 6]]);

// Reshape (returns view when possible)
$reshaped = $arr->reshape([3, 2]);
// [[1 2]
//  [3 4]
//  [5 6]]

// Flatten to 1D
$flat = $arr->flatten();  // [1 2 3 4 5 6]

// Add dimension
$expanded = $arr->expandDims(0);
print_r($expanded->shape());  // [1, 2, 3]

// Remove single dimensions
$squeezed = $expanded->squeeze();
print_r($squeezed->shape());  // [2, 3]
```

## Type Conversion

```php
$arr = NDArray::array([1, 2, 3]);  // Defaults to Float64

// Convert to different type
$int_arr = $arr->astype(DType::Int32);
$float_arr = $arr->astype(DType::Float32);

// Convert to PHP array
$php_array = $arr->toArray();  // [1, 2, 3]

// Get scalar from 0-d array
$scalar = $arr->sum()->toScalar();  // 6 (as PHP float)
```

## Complete Example

Here's a practical example combining multiple operations:

```php
<?php

require_once 'vendor/autoload.php';

use PhpMlKit\NDArray\NDArray;

// Create sample data: 100 samples, 3 features
$data = NDArray::random([100, 3]);

// Calculate mean of each feature
$mean = $data->mean(axis: 0);
print_r($mean->toArray());

// Center the data (subtract mean)
$centered_data = $data->subtract($mean);

// Calculate standard deviation
$std = $centered_data->std(axis: 0);
print_r($std->toArray());

// Normalize (z-score normalization)
$normalized = $centered_data->divide($std);

// Verify: mean should be ~0, std should be ~1
print_r($normalized->mean(axis: 0)->toArray());
print_r($normalized->std(axis: 0)->toArray());

// Split into training and test sets using slices
$train = $normalized['0:80'];   // First 80 samples
$test = $normalized['80:100'];  // Last 20 samples

print_r($train->shape());  // [80, 3]
print_r($test->shape());   // [20, 3]
```

## Important Differences from NumPy

### No Operator Overloading

PHP doesn't support operator overloading, so you **must use method calls**:

```php
// ❌ This doesn't work in PHP
$c = $a + $b;
$c = $a * 2;

// ✅ Use method calls instead
$c = $a->add($b);
$c = $a->multiply(2);
```

### Multi-dimensional Indexing Uses Strings

```php
// ❌ This is invalid PHP syntax
$value = $matrix[0, 0];

// ✅ Use string or get() method
$value = $matrix['0,0'];
$value = $matrix->get(0, 0);
```

### Slicing Uses Method Calls & String indexing

```php
// ❌ This doesn't work
$slice = $arr[0:5];

// ✅ Use slice() method
$slice = $arr->slice(['0:5']);

// ✅ Use brackets with strings
$slice = $arr['0:5'];
```

## Next Steps

Now that you know the basics:

- **[Arrays](/guide/fundamentals/arrays)** - Deep dive into array concepts
- **[Views vs Copies](/guide/fundamentals/views-vs-copies)** - Critical for performance
- **[NumPy Migration](/guide/getting-started/numpy-migration)** - If coming from Python
- **[API Reference](/api/ndarray-class)** - Complete method listing
