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
public function diagonal(int $offset = 0): NDArray
```

Extract diagonal elements from a 2D array.

Returns a 1D array containing the diagonal. Use `$offset` to extract a super-diagonal (positive) or sub-diagonal (negative).

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$offset` | `int` | Offset from the main diagonal. Optional. Default: `0`. |

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

$super = $matrix->diagonal(1);
print_r($super->toArray());
// Output: [2, 6]
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

## svd()

```php
public function svd(bool $computeUv = true): array|NDArray
```

Compute Singular Value Decomposition.

Decomposes matrix A into U * S * VT where U and VT are orthogonal matrices and S contains the singular values.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$computeUv` | `bool` | If true, compute U and VT matrices. Optional. Default: `true`. |

### Returns

- `array{0: NDArray, 1: NDArray, 2: NDArray}|NDArray` - `[U, S, VT]` if `$computeUv` is true, otherwise just the singular values `S`.

### Examples

```php
$a = NDArray::array([
    [3, 0],
    [0, 2]
]);

[$u, $s, $vt] = $a->svd();

print_r($u->shape());   // [2, 2]
print_r($s->shape());   // [2]
print_r($vt->shape());  // [2, 2]
```

## qr()

```php
public function qr(): array
```

Compute QR decomposition.

Decomposes matrix A into Q * R where Q is orthogonal and R is upper triangular.

### Returns

- `array{0: NDArray, 1: NDArray}` - `[Q, R]`

### Examples

```php
$a = NDArray::array([
    [12, -51, 4],
    [6, 167, -68],
    [-4, 24, -41]
]);

[$q, $r] = $a->qr();

// Q * R reconstructs A
$reconstructed = $q->matmul($r);
```

## cholesky()

```php
public function cholesky(bool $upper = false): NDArray
```

Compute Cholesky decomposition of a Hermitian positive-definite matrix.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$upper` | `bool` | If true, return upper triangular U. Optional. Default: `false`. |

### Returns

- `NDArray` - Lower triangular L such that A = L * L^T (or upper U such that A = U^T * U).

### Examples

```php
$a = NDArray::array([
    [4, 12, -16],
    [12, 37, -43],
    [-16, -43, 98]
]);

$l = $a->cholesky();
// L * L^T reconstructs A
```

## inv()

```php
public function inv(): NDArray
```

Compute the inverse of a square matrix.

### Returns

- `NDArray` - The inverse matrix.

### Examples

```php
$a = NDArray::array([
    [4, 7],
    [2, 6]
]);

$inv = $a->inv();

// A * A^-1 is identity
$identity = $a->matmul($inv);
```

## det()

```php
public function det(): float
```

Compute the determinant of a square matrix.

### Returns

- `float` - The determinant.

### Examples

```php
$a = NDArray::array([
    [4, 7],
    [2, 6]
]);

echo $a->det();
// Output: 10.0
```

## solve()

```php
public function solve(NDArray $b): NDArray
```

Solve a linear system A * x = b.

A must be a 2D square matrix. b can be 1D or 2D.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$b` | `NDArray` | Right-hand side array. |

### Returns

- `NDArray` - Solution array x.

### Examples

```php
$a = NDArray::array([
    [3, 2, -1],
    [2, -2, 4],
    [-2, 1, -2]
]);
$b = NDArray::array([1, -2, 0]);

$x = $a->solve($b);
```

## lstsq()

```php
public function lstsq(NDArray $b): array
```

Solve a least-squares problem min ||Ax - b||_2.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$b` | `NDArray` | Right-hand side array (1D or 2D). |

### Returns

- `array{0: NDArray, 1: NDArray|null, 2: int, 3: NDArray}` - `[x, residuals, rank, s]` where `x` is the least-squares solution, `residuals` is the sum of residuals (or null if not applicable), `rank` is the effective rank of A, and `s` is the singular values of A.

### Examples

```php
$a = NDArray::array([
    [1, 1, 1],
    [2, 3, 4],
    [3, 5, 2],
    [4, 2, 5],
    [5, 4, 3]
]);
$b = NDArray::array([-10, 12, 14, 16, 18]);

[$x, $residuals, $rank, $s] = $a->lstsq($b);
```

## pinv()

```php
public function pinv(?float $rcond = null): NDArray
```

Compute the Moore-Penrose pseudo-inverse of a matrix using SVD.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$rcond` | `float\|null` | Cutoff for small singular values. Optional. Default: machine precision. |

### Returns

- `NDArray` - The pseudo-inverse matrix.

### Examples

```php
$a = NDArray::array([
    [1, 2],
    [3, 4],
    [5, 6]
]);

$pinv = $a->pinv();

// A * A^+ * A ≈ A
```

## cond()

```php
public function cond(): float
```

Compute the 2-norm condition number of a matrix using SVD.

### Returns

- `float` - The condition number.

### Examples

```php
$a = NDArray::array([
    [1, 2],
    [3, 4]
]);

echo $a->cond();
// Output: ~14.93
```

## rank()

```php
public function rank(?float $tol = null): int
```

Compute the rank of a matrix using SVD.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$tol` | `float\|null` | Threshold below which SVD values are considered zero. Optional. Default: `max(m,n) * eps * max(singular_value)`. |

### Returns

- `int` - The effective rank.

### Examples

```php
$a = NDArray::array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]);

echo $a->rank();
// Output: 2
```
