# Linear Algebra Methods

Reference for linear algebra operations.

## norm()

```php
public function norm(
    float|int|string|null $ord = null,
    ?int $axis = null,
    bool $keepdims = false
): float|NDArray
```

Compute vector or matrix norm.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$ord` | `float\|int\|string\|null` | Norm order. Optional. Default: `null`. |
| `$axis` | `int\|null` | Reduction axis. If null, reduces all elements. Optional. Default: `null`. |
| `$keepdims` | `bool` | Keep reduced axis with size 1 (axis mode only). Optional. Default: `false`. |

### Returns

- `float|NDArray` - Scalar if axis is null, otherwise an NDArray.

### Raises

- `InvalidArgumentException` - If norm order is unsupported or incompatible with array dimensions.

### Supported Orders

| Order | Description |
|-------|-------------|
| `null` | L2 norm (Euclidean), or Frobenius for 2D matrices |
| `1` | L1 norm (sum of absolute values) |
| `2` | L2 norm (Euclidean) |
| `INF` | Infinity norm (maximum absolute value) |
| `-INF` | Negative infinity norm (minimum absolute value) |
| `'fro'` | Frobenius norm (matrix only, axis must be null) |

### Examples

```php
$vector = NDArray::array([3, 4]);

// L2 norm (Euclidean)
echo $vector->norm();
// Output: 5.0

// L1 norm
echo $vector->norm(1);
// Output: 7.0

// Infinity norm
echo $vector->norm('inf');
// Output: 4.0

// Matrix norm
$matrix = NDArray::array([
    [1, 2],
    [3, 4]
]);
echo $matrix->norm();
// Output: 5.477... (Frobenius norm)
```

## dot()

```php
public function dot(NDArray $other): NDArray
```

Compute dot product of two arrays.

For 1D arrays: returns scalar
For 2D arrays: returns matrix multiplication

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$other` | `NDArray` | The other array. |

### Returns

- `NDArray` - Dot product result.

### Examples

```php
// Matrix multiplication
$a = NDArray::array([
    [1, 2],
    [3, 4]
]);

$b = NDArray::array([
    [5, 6],
    [7, 8]
]);

$c = $a->dot($b);
print_r($c->toArray());
// Output: [[19, 22], [43, 50]]

// Vector dot product
$v1 = NDArray::array([1, 2, 3]);
$v2 = NDArray::array([4, 5, 6]);
$result = $v1->dot($v2);
print_r($result->toArray());
// Output: 32
```

## matmul()

```php
public function matmul(NDArray $other): NDArray
```

Compute matrix multiplication.

Requires both arrays to be at least 2D.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$other` | `NDArray` | The other array. |

### Returns

- `NDArray` - Matrix multiplication result.

### Examples

```php
$a = NDArray::array([
    [1, 2],
    [3, 4]
]);

$b = NDArray::array([
    [5, 6],
    [7, 8]
]);

$c = $a->matmul($b);
print_r($c->toArray());
// Output: [[19, 22], [43, 50]]
```

## diagonal()

```php
public function diagonal(): NDArray
```

Extract diagonal elements from a 2D array.

Returns a 1D array containing the diagonal.

### Returns

- `NDArray` - 1D array of diagonal elements.

### Examples

```php
$matrix = NDArray::array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]);

$diag = $matrix->diagonal();
print_r($diag->toArray());
// Output: [1, 5, 9]
```

## trace()

```php
public function trace(): NDArray
```

Compute trace (sum of diagonal elements).

Returns a scalar array.

### Returns

- `NDArray` - Scalar array containing the trace.

### Examples

```php
$matrix = NDArray::array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]);

$tr = $matrix->trace();
print_r($tr->toArray());
// Output: 15 (1 + 5 + 9)
```
