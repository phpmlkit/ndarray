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
public function dot(NDArray $other): float|int|Complex|NDArray
```

Generalized dot product for 1D and 2D operands. Operand dtypes are promoted to a common type before the operation.

- **1D × 1D**: inner product → scalar
- **2D × 2D**: matrix product → 2D array
- **1D × 2D** or **2D × 1D**: vector–matrix product → 1D array

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$other` | `NDArray` | The other array. |

### Returns

- `float|int|Complex|NDArray` - Scalar when the result is 0-D (1D·1D), otherwise an NDArray.

### Examples

```php
// Matrix multiplication (2D × 2D)
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

// Vector dot product (1D × 1D) — returns scalar
$v1 = NDArray::array([1, 2, 3]);
$v2 = NDArray::array([4, 5, 6]);
$result = $v1->dot($v2);
echo $result;
// Output: 32

// Complex vector dot product
use PhpMlKit\NDArray\Complex;

$v1 = NDArray::array([new Complex(1, 1), new Complex(2, 2)]);
$v2 = NDArray::array([new Complex(3, 3), new Complex(4, 4)]);
$result = $v1->dot($v2);
// $result is a Complex instance
```

## matmul()

```php
public function matmul(NDArray $other): float|int|Complex|NDArray
```

Matrix multiplication for 1D and 2D operands. Operand dtypes are promoted to a common type. Operands with more than two dimensions are not supported.

- **2D × 2D**: matrix × matrix → 2D array
- **2D × 1D** or **1D × 2D**: matrix × vector → 1D array
- **1D × 1D**: inner product → scalar

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$other` | `NDArray` | The other array. |

### Returns

- `float|int|Complex|NDArray` - Scalar when the result is 0-D (1D·1D), otherwise an NDArray.

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
public function trace(): float|int|Complex
```

Compute trace (sum of diagonal elements).

Returns a scalar value (not an array). For real arrays, returns `float|int`. For complex arrays, returns a `Complex` instance.

### Returns

- `float|int|Complex` - The trace value.

### Examples

```php
$matrix = NDArray::array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]);

$tr = $matrix->trace();
echo $tr;
// Output: 15 (1 + 5 + 9)

// Complex trace
use PhpMlKit\NDArray\Complex;

$complex = NDArray::array([
    [new Complex(1, 1), new Complex(2, 2)],
    [new Complex(3, 3), new Complex(4, 4)],
], DType::Complex128);

$tr = $complex->trace();
echo $tr->real;  // 5.0
echo $tr->imag;  // 5.0
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

## eig()

```php
public function eig(): array
```

Eigenvalue decomposition of a square matrix: `A v = λ v`.

For real `Float32` / `Float64` input, eigenvalues and eigenvectors are complex. For complex input, output dtypes match the array dtype.

### Returns

- `array{0: NDArray, 1: NDArray}` - `[eigenvalues, eigenvectors]`

### Examples

```php
$a = NDArray::array([
    [4.0, 2.0],
    [1.0, 3.0],
]);

[$w, $v] = $a->eig();
// $w: eigenvalues (complex dtype for real input)
// $v: eigenvector columns
```

## eigvals()

```php
public function eigvals(): NDArray
```

Eigenvalues only for a general square matrix (no eigenvectors).

For real input, eigenvalues are complex. For complex input, output dtype matches the array dtype.

### Returns

- `NDArray` - 1D vector of eigenvalues

### Examples

```php
$a = NDArray::array([[4.0, 2.0], [1.0, 3.0]]);
$w = $a->eigvals();
```

## eigh()

```php
public function eigh(bool $upper = false): array
```

Eigen decomposition for Hermitian (or real symmetric) matrices.

Eigenvalues are real. Eigenvectors use the same element type as `A`. Only the stored triangle is read: lower if `$upper` is false, upper if true.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$upper` | `bool` | If true, read the upper triangle; if false, the lower. Optional. Default: `false`. |

### Returns

- `array{0: NDArray, 1: NDArray}` - `[eigenvalues, eigenvectors]`

### Examples

```php
$a = NDArray::array([
    [2.0, 1.0],
    [1.0, 2.0],
]);

[$w, $v] = $a->eigh();
```

## eigvalsh()

```php
public function eigvalsh(bool $upper = false): NDArray
```

Eigenvalues only for a Hermitian (or real symmetric) matrix. Same triangle convention as `eigh()`.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$upper` | `bool` | If true, read the upper triangle; if false, the lower. Optional. Default: `false`. |

### Returns

- `NDArray` - 1D vector of real eigenvalues

### Examples

```php
$a = NDArray::array([
    [2.0, 1.0],
    [1.0, 2.0],
]);

$w = $a->eigvalsh();
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
