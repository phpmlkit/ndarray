# Broadcasting

Broadcasting allows operations between arrays of different shapes by automatically expanding dimensions to make them compatible.

## How Broadcasting Works

When operating on two arrays, NDArray compares their shapes element-wise starting from the trailing (rightmost) dimension. Two dimensions are compatible when:

1. They are equal, or
2. One of them is 1

Missing dimensions are treated as 1.

## Rules of Broadcasting

1. **Start from the trailing (rightmost) dimension** and work left
2. **Compare dimensions one by one**:
   - If dimensions are equal, they are compatible
   - If one dimension is 1, the array is stretched (broadcast) along that axis to match the other
   - If dimensions are different and neither is 1, broadcasting fails
3. **Missing dimensions** in the smaller array are treated as 1

## Common Broadcasting Patterns

### Scalar Broadcasting

Scalars are broadcast to match any array shape:

```php
$arr = NDArray::ones([3, 3]);
$result = $arr->add(5);  // 5 is broadcast to [3, 3]
// Result: all elements are 6

$matrix = NDArray::zeros([3, 4]);
$result = $matrix->add(10);  // Every element becomes 10
```

### Row Broadcasting

A 1D array can be broadcast across rows of a 2D matrix:

```php
$matrix = NDArray::zeros([3, 4]);
$row = NDArray::array([1, 2, 3, 4]);  // Shape [4]
$result = $matrix->add($row);  // Row broadcast to [3, 4]
// Result:
// [[1, 2, 3, 4]
//  [1, 2, 3, 4]
//  [1, 2, 3, 4]]
```

### Column Broadcasting

An array with shape `[n, 1]` can be broadcast across columns:

```php
$matrix = NDArray::zeros([3, 4]);
$col = NDArray::array([[1], [2], [3]]);  // Shape [3, 1]
$result = $matrix->add($col);  // Column broadcast to [3, 4]
// Result:
// [[1, 1, 1, 1]
//  [2, 2, 2, 2]
//  [3, 3, 3, 3]]
```

### 3D Broadcasting

Broadcasting works with any number of dimensions:

```php
// Shape [2, 3, 4] broadcast with shape [3, 4]
$tensor = NDArray::ones([2, 3, 4]);
$matrix = NDArray::array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
$result = $tensor->add($matrix);  // Shape [2, 3, 4]
// The [3, 4] matrix is broadcast across the first dimension
```

### Shape Compatibility Examples

```php
// Compatible: (5, 4) and (4,)
$a = NDArray::ones([5, 4]);
$b = NDArray::ones([4]);
$result = $a->add($b);  // OK: b is broadcast to [5, 4]

// Compatible: (5, 4) and (5, 1)
$a = NDArray::ones([5, 4]);
$b = NDArray::ones([5, 1]);
$result = $a->add($b);  // OK: b is broadcast to [5, 4]

// Compatible: (3, 1, 1) and (1, 5)
$a = NDArray::ones([3, 1, 1]);
$b = NDArray::ones([1, 5]);
$result = $a->add($b);  // OK: shapes become [3, 1, 5] and [3, 1, 5]

// NOT Compatible: (5, 4) and (5,)
$a = NDArray::ones([5, 4]);
$b = NDArray::ones([5]);
// $a->add($b);  // Error: trailing dimensions don't match (4 vs 5)
```

## Broadcasting in Arithmetic Operations

All arithmetic operations support broadcasting:

```php
$a = NDArray::array([[1, 2], [3, 4]]);  // Shape [2, 2]
$b = NDArray::array([10, 20]);          // Shape [2]

$sum = $a->add($b);           // [[11, 22], [13, 24]]
$diff = $a->subtract($b);     // [[-9, -18], [-7, -16]]
$prod = $a->multiply($b);     // [[10, 40], [30, 80]]
$quot = $a->divide($b);       // [[0.1, 0.1], [0.3, 0.2]]
```

## Broadcasting with Comparison Operations

Comparison operations also broadcast:

```php
$matrix = NDArray::array([[1, 2, 3], [4, 5, 6]]);
$threshold = NDArray::array([[2], [5]]);  // Shape [2, 1]

$mask = $matrix->gt($threshold);
// Result:
// [[false, false, true]
//  [false, false, true]]
```

## Practical Examples

### Normalizing Data

```php
// Normalize each column by its mean
$data = NDArray::array([[1, 10], [2, 20], [3, 30]]);
$colMeans = $data->mean(axis: 0)->reshape([1, 2]);  // Shape [1, 2]
$normalized = $data->subtract($colMeans);  // Broadcast subtraction
```

### Adding Bias to Neural Network Activations

```php
// activations: shape [batch_size, num_neurons]
// bias: shape [num_neurons]
$activations = NDArray::zeros([32, 100]);
$bias = NDArray::ones([100]);
$output = $activations->add($bias);  // Broadcast bias to each batch
```

### Computing Distances

```php
// Compute (x - mean)^2 for each element
$x = NDArray::array([[1, 2], [3, 4], [5, 6]]);
$mean = NDArray::array([3, 4])->reshape([1, 2]);
$squaredDiff = $x->subtract($mean)->power(2);
```

## Common Mistakes

### 1. Mismatched Dimensions

```php
// WRONG: dimensions don't match from the right
$a = NDArray::ones([5, 4]);
$b = NDArray::ones([5]);  // Error: 4 != 5
// $a->add($b);  // Broadcasting error
```

**Fix**: Reshape to add a dimension of size 1:
```php
$b = NDArray::ones([5])->reshape([5, 1]);  // Now compatible
$result = $a->add($b);
```

### 2. Confusing Row and Column Vectors

```php
// Row vector
$row = NDArray::array([1, 2, 3]);        // Shape [3]

// Column vector  
$col = NDArray::array([[1], [2], [3]]);  // Shape [3, 1]

$matrix = NDArray::ones([3, 3]);

$rowResult = $matrix->add($row);  // Adds to each row
$colResult = $matrix->add($col);  // Adds to each column
```

## Performance Considerations

Broadcasting is memory-efficient - it doesn't actually create copies of the broadcasted array. The operations are applied on-the-fly during computation.

However, if you need to use a broadcasted array multiple times, consider explicitly creating a full-size copy with `tile()` or `repeat()` methods (when available).

## See Also

- **[Arithmetic Operations](/guide/operations/arithmetic)** - Element-wise operations with broadcasting
- **[Shape Manipulation](/guide/operations/shape-manipulation)** - Reshaping arrays for broadcasting
- **[NDArray Class](/api/ndarray-class)** - Check array shapes with `shape()`
