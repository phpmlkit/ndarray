# Mathematical Functions

Reference for mathematical operations including arithmetic, transcendental functions, powers, and rounding.

---

## add()

```php
public function add(float|int|NDArray $other): NDArray
```

Add another array or scalar to this array element-wise.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$other` | `float\|int\|NDArray` | Array or scalar to add. |

### Returns

- `NDArray` - New array with element-wise sum.

### Examples

```php
// Array + Array
$a = NDArray::array([1, 2, 3]);
$b = NDArray::array([4, 5, 6]);
$result = $a->add($b);
print_r($result->toArray());
// Output: [5, 7, 9]

// Array + Scalar
$result = $a->add(10);
print_r($result->toArray());
// Output: [11, 12, 13]
```

---

## subtract()

```php
public function subtract(float|int|NDArray $other): NDArray
```

Subtract another array or scalar from this array element-wise.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$other` | `float\|int\|NDArray` | Array or scalar to subtract. |

### Returns

- `NDArray` - New array with element-wise difference.

### Examples

```php
$a = NDArray::array([10, 20, 30]);
$b = NDArray::array([1, 2, 3]);
$result = $a->subtract($b);
print_r($result->toArray());
// Output: [9, 18, 27]

// Scalar subtraction
$result = $a->subtract(5);
print_r($result->toArray());
// Output: [5, 15, 25]
```

---

## multiply()

```php
public function multiply(float|int|NDArray $other): NDArray
```

Multiply this array by another array or scalar element-wise.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$other` | `float\|int\|NDArray` | Array or scalar to multiply by. |

### Returns

- `NDArray` - New array with element-wise product.

### Examples

```php
$a = NDArray::array([2, 3, 4]);
$b = NDArray::array([5, 6, 7]);
$result = $a->multiply($b);
print_r($result->toArray());
// Output: [10, 18, 28]

// Scalar multiplication
$result = $a->multiply(3);
print_r($result->toArray());
// Output: [6, 9, 12]
```

---

## divide()

```php
public function divide(float|int|NDArray $other): NDArray
```

Divide this array by another array or scalar element-wise.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$other` | `float\|int\|NDArray` | Array or scalar to divide by. |

### Returns

- `NDArray` - New array with element-wise quotient.

### Examples

```php
$a = NDArray::array([10.0, 20.0, 30.0]);
$b = NDArray::array([2.0, 4.0, 5.0]);
$result = $a->divide($b);
print_r($result->toArray());
// Output: [5.0, 5.0, 6.0]

// Scalar division
$result = $a->divide(2);
print_r($result->toArray());
// Output: [5.0, 10.0, 15.0]
```

---

## rem()

```php
public function rem(float|int|NDArray $other): NDArray
```

Compute remainder (modulo) with another array or scalar element-wise.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$other` | `float\|int\|NDArray` | Array or scalar. |

### Returns

- `NDArray` - New array with element-wise remainder.

### Examples

```php
$a = NDArray::array([10, 15, 20]);
$b = NDArray::array([3, 4, 6]);
$result = $a->rem($b);
print_r($result->toArray());
// Output: [1, 3, 2]

// Scalar modulo
$result = $a->rem(7);
print_r($result->toArray());
// Output: [3, 1, 6]
```

---

## mod()

```php
public function mod(float|int|NDArray $other): NDArray
```

Compute modulo with another array or scalar element-wise.

Alias for `rem()`.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$other` | `float\|int\|NDArray` | Array or scalar. |

### Returns

- `NDArray` - New array with element-wise modulo.

### Examples

```php
$a = NDArray::array([10, 15, 20]);
$result = $a->mod(7);
print_r($result->toArray());
// Output: [3, 1, 6]
```

---

## abs()

```php
public function abs(): NDArray
```

Compute absolute value element-wise.

### Returns

- `NDArray` - New array with absolute values.

### Examples

```php
$arr = NDArray::array([-1, -2, 3, -4]);
$result = $arr->abs();
print_r($result->toArray());
// Output: [1, 2, 3, 4]
```

---

## negative()

```php
public function negative(): NDArray
```

Compute negation element-wise (-$a).

Not supported for unsigned integers or bool.

### Returns

- `NDArray` - New array with negated values.

### Raises

- `InvalidArgumentException` - For unsigned integer or boolean arrays.

### Examples

```php
$arr = NDArray::array([1, -2, 3, -4]);
$result = $arr->negative();
print_r($result->toArray());
// Output: [-1, 2, -3, 4]
```

---

## sqrt()

```php
public function sqrt(): NDArray
```

Compute square root element-wise.

### Returns

- `NDArray` - New array with square roots.

### Examples

```php
$arr = NDArray::array([1.0, 4.0, 9.0, 16.0]);
$result = $arr->sqrt();
print_r($result->toArray());
// Output: [1.0, 2.0, 3.0, 4.0]
```

---

## cbrt()

```php
public function cbrt(): NDArray
```

Compute cube root element-wise.

### Returns

- `NDArray` - New array with cube roots.

### Examples

```php
$arr = NDArray::array([1.0, 8.0, 27.0, 64.0]);
$result = $arr->cbrt();
print_r($result->toArray());
// Output: [1.0, 2.0, 3.0, 4.0]
```

---

## exp()

```php
public function exp(): NDArray
```

Compute exponential (e^x) element-wise.

### Returns

- `NDArray` - New array with exponential values.

### Examples

```php
$arr = NDArray::array([0.0, 1.0, 2.0]);
$result = $arr->exp();
print_r($result->toArray());
// Output: [1.0, 2.718..., 7.389...]
```

---

## exp2()

```php
public function exp2(): NDArray
```

Compute base-2 exponential (2^x) element-wise.

### Returns

- `NDArray` - New array with base-2 exponential values.

### Examples

```php
$arr = NDArray::array([0, 1, 2, 3]);
$result = $arr->exp2();
print_r($result->toArray());
// Output: [1, 2, 4, 8]
```

---

## log()

```php
public function log(): NDArray
```

Compute natural logarithm element-wise.

### Returns

- `NDArray` - New array with natural logarithm values.

### Examples

```php
$arr = NDArray::array([1.0, 2.718, 7.389]);
$result = $arr->log();
print_r($result->toArray());
// Output: [0.0, 1.0, 2.0]
```

---

## ln()

```php
public function ln(): NDArray
```

Compute natural logarithm element-wise.

Alias for `log()`.

### Returns

- `NDArray` - New array with natural logarithm values.

### Examples

```php
$arr = NDArray::array([1.0, 2.718, 7.389]);
$result = $arr->ln();
print_r($result->toArray());
// Output: [0.0, 1.0, 2.0]
```

---

## log2()

```php
public function log2(): NDArray
```

Compute base-2 logarithm element-wise.

### Returns

- `NDArray` - New array with base-2 logarithm values.

### Examples

```php
$arr = NDArray::array([1, 2, 4, 8, 16]);
$result = $arr->log2();
print_r($result->toArray());
// Output: [0, 1, 2, 3, 4]
```

---

## log10()

```php
public function log10(): NDArray
```

Compute base-10 logarithm element-wise.

### Returns

- `NDArray` - New array with base-10 logarithm values.

### Examples

```php
$arr = NDArray::array([1, 10, 100, 1000]);
$result = $arr->log10();
print_r($result->toArray());
// Output: [0, 1, 2, 3]
```

---

## ln1p()

```php
public function ln1p(): NDArray
```

Compute ln(1+x) element-wise.

More accurate than log(1+x) for small x.

### Returns

- `NDArray` - New array with ln(1+x) values.

### Examples

```php
$arr = NDArray::array([0.0, 0.1, 0.01]);
$result = $arr->ln1p();
print_r($result->toArray());
// Output: [0.0, 0.0953..., 0.00995...]
```

---

## sin()

```php
public function sin(): NDArray
```

Compute sine element-wise.

### Returns

- `NDArray` - New array with sine values.

### Examples

```php
$arr = NDArray::array([0, M_PI/2, M_PI]);
$result = $arr->sin();
print_r($result->toArray());
// Output: [0.0, 1.0, 0.0]
```

---

## cos()

```php
public function cos(): NDArray
```

Compute cosine element-wise.

### Returns

- `NDArray` - New array with cosine values.

### Examples

```php
$arr = NDArray::array([0, M_PI/2, M_PI]);
$result = $arr->cos();
print_r($result->toArray());
// Output: [1.0, 0.0, -1.0]
```

---

## tan()

```php
public function tan(): NDArray
```

Compute tangent element-wise.

### Returns

- `NDArray` - New array with tangent values.

### Examples

```php
$arr = NDArray::array([0, M_PI/4, M_PI/2]);
$result = $arr->tan();
print_r($result->toArray());
// Output: [0.0, 1.0, very large number]
```

---

## asin()

```php
public function asin(): NDArray
```

Compute arc sine element-wise.

### Returns

- `NDArray` - New array with arc sine values.

### Examples

```php
$arr = NDArray::array([0.0, 0.5, 1.0]);
$result = $arr->asin();
print_r($result->toArray());
// Output: [0.0, 0.523..., 1.570...]
```

---

## acos()

```php
public function acos(): NDArray
```

Compute arc cosine element-wise.

### Returns

- `NDArray` - New array with arc cosine values.

### Examples

```php
$arr = NDArray::array([0.0, 0.5, 1.0]);
$result = $arr->acos();
print_r($result->toArray());
// Output: [1.570..., 1.047..., 0.0]
```

---

## atan()

```php
public function atan(): NDArray
```

Compute arc tangent element-wise.

### Returns

- `NDArray` - New array with arc tangent values.

### Examples

```php
$arr = NDArray::array([0.0, 1.0, -1.0]);
$result = $arr->atan();
print_r($result->toArray());
// Output: [0.0, 0.785..., -0.785...]
```

---

## sinh()

```php
public function sinh(): NDArray
```

Compute hyperbolic sine element-wise.

### Returns

- `NDArray` - New array with hyperbolic sine values.

### Examples

```php
$arr = NDArray::array([0.0, 1.0, 2.0]);
$result = $arr->sinh();
print_r($result->toArray());
// Output: [0.0, 1.175..., 3.626...]
```

---

## cosh()

```php
public function cosh(): NDArray
```

Compute hyperbolic cosine element-wise.

### Returns

- `NDArray` - New array with hyperbolic cosine values.

### Examples

```php
$arr = NDArray::array([0.0, 1.0, 2.0]);
$result = $arr->cosh();
print_r($result->toArray());
// Output: [1.0, 1.543..., 3.762...]
```

---

## tanh()

```php
public function tanh(): NDArray
```

Compute hyperbolic tangent element-wise.

### Returns

- `NDArray` - New array with hyperbolic tangent values.

### Examples

```php
$arr = NDArray::array([0.0, 1.0, 2.0]);
$result = $arr->tanh();
print_r($result->toArray());
// Output: [0.0, 0.761..., 0.964...]
```

---

## toDegrees()

```php
public function toDegrees(): NDArray
```

Convert radians to degrees element-wise.

### Returns

- `NDArray` - New array with degree values.

### Examples

```php
$arr = NDArray::array([0, M_PI/2, M_PI]);
$result = $arr->toDegrees();
print_r($result->toArray());
// Output: [0, 90, 180]
```

---

## toRadians()

```php
public function toRadians(): NDArray
```

Convert degrees to radians element-wise.

### Returns

- `NDArray` - New array with radian values.

### Examples

```php
$arr = NDArray::array([0, 90, 180]);
$result = $arr->toRadians();
print_r($result->toArray());
// Output: [0, 1.570..., 3.141...]
```

---

## floor()

```php
public function floor(): NDArray
```

Compute floor element-wise (round down to nearest integer).

### Returns

- `NDArray` - New array with floor values.

### Examples

```php
$arr = NDArray::array([1.2, 2.5, 3.7, -1.2]);
$result = $arr->floor();
print_r($result->toArray());
// Output: [1, 2, 3, -2]
```

---

## ceil()

```php
public function ceil(): NDArray
```

Compute ceiling element-wise (round up to nearest integer).

### Returns

- `NDArray` - New array with ceiling values.

### Examples

```php
$arr = NDArray::array([1.2, 2.5, 3.7, -1.2]);
$result = $arr->ceil();
print_r($result->toArray());
// Output: [2, 3, 4, -1]
```

---

## round()

```php
public function round(): NDArray
```

Compute round element-wise (round to nearest integer).

### Returns

- `NDArray` - New array with rounded values.

### Examples

```php
$arr = NDArray::array([1.2, 2.5, 3.7, -1.5]);
$result = $arr->round();
print_r($result->toArray());
// Output: [1, 2, 4, -2]
```

---

## pow2()

```php
public function pow2(): NDArray
```

Compute x^2 (square) element-wise.

### Returns

- `NDArray` - New array with squared values.

### Examples

```php
$arr = NDArray::array([1, 2, 3, 4]);
$result = $arr->pow2();
print_r($result->toArray());
// Output: [1, 4, 9, 16]
```

---

## powi()

```php
public function powi(int $exp): NDArray
```

Compute x^n where n is an integer, element-wise.

Generally faster than powf() for integer exponents.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$exp` | `int` | Integer exponent. |

### Returns

- `NDArray` - New array with x^n values.

### Examples

```php
$arr = NDArray::array([2, 3, 4]);
$result = $arr->powi(3);
print_r($result->toArray());
// Output: [8, 27, 64]
```

---

## powf()

```php
public function powf(float $exp): NDArray
```

Compute x^y where y is a float, element-wise.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$exp` | `float` | Float exponent. |

### Returns

- `NDArray` - New array with x^y values.

### Examples

```php
$arr = NDArray::array([4.0, 9.0, 16.0]);
$result = $arr->powf(0.5);
print_r($result->toArray());
// Output: [2.0, 3.0, 4.0]
```

---

## recip()

```php
public function recip(): NDArray
```

Compute reciprocal (1/x) element-wise.

### Returns

- `NDArray` - New array with reciprocal values.

### Examples

```php
$arr = NDArray::array([1.0, 2.0, 4.0, 0.5]);
$result = $arr->recip();
print_r($result->toArray());
// Output: [1.0, 0.5, 0.25, 2.0]
```

---

## signum()

```php
public function signum(): NDArray
```

Compute signum element-wise.

Returns -1 for negative values, 0 for zero, and 1 for positive values.

### Returns

- `NDArray` - New array with signum values.

### Examples

```php
$arr = NDArray::array([-5, -1, 0, 1, 5]);
$result = $arr->signum();
print_r($result->toArray());
// Output: [-1, -1, 0, 1, 1]
```

---

## hypot()

```php
public function hypot(float $other): NDArray
```

Compute hypotenuse element-wise with a scalar value.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$other` | `float` | Scalar value. |

### Returns

- `NDArray` - New array with hypotenuse values (sqrt(x^2 + y^2)).

### Examples

```php
$arr = NDArray::array([3.0, 4.0, 5.0]);
$result = $arr->hypot(4.0);
print_r($result->toArray());
// Output: [5.0, 5.656..., 6.403...]
```

---

## clamp()

```php
public function clamp(float|int $min, float|int $max): NDArray
```

Clamp (clip) array values to a specified range.

Values outside [min, max] are set to the nearest boundary.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$min` | `float\|int` | Minimum value. |
| `$max` | `float\|int` | Maximum value. |

### Returns

- `NDArray` - New array with clamped values.

### Raises

- `InvalidArgumentException` - If min > max.

### Examples

```php
$arr = NDArray::array([-5, 0, 5, 10, 15]);
$result = $arr->clamp(0, 10);
print_r($result->toArray());
// Output: [0, 0, 5, 10, 10]
```

---

## clip()

```php
public function clip(float|int $min, float|int $max): NDArray
```

Clip array values to a specified range.

Alias for `clamp()`.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$min` | `float\|int` | Minimum value. |
| `$max` | `float\|int` | Maximum value. |

### Returns

- `NDArray` - New array with clipped values.

### Raises

- `InvalidArgumentException` - If min > max.

### Examples

```php
$arr = NDArray::array([-5, 0, 5, 10, 15]);
$result = $arr->clip(0, 10);
print_r($result->toArray());
// Output: [0, 0, 5, 10, 10]
```

---

## minimum()

```php
public function minimum(NDArray $other): NDArray
```

Element-wise minimum of two arrays.

Compares two arrays element-wise and returns a new array containing the smaller value at each position. Supports broadcasting.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$other` | `NDArray` | The array to compare with. |

### Returns

- `NDArray` - New array with element-wise minimum values.

### Examples

```php
$a = NDArray::array([1, 5, 3, 8]);
$b = NDArray::array([2, 4, 6, 7]);
$result = $a->minimum($b);
print_r($result->toArray());
// Output: [1, 4, 3, 7]

// With broadcasting
$a = NDArray::array([[1, 2, 3], [4, 5, 6]]);
$b = NDArray::array([2, 2, 2]);
$result = $a->minimum($b);
print_r($result->toArray());
// Output: [[1, 2, 2], [2, 2, 2]]
```

---

## maximum()

```php
public function maximum(NDArray $other): NDArray
```

Element-wise maximum of two arrays.

Compares two arrays element-wise and returns a new array containing the larger value at each position. Supports broadcasting.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$other` | `NDArray` | The array to compare with. |

### Returns

- `NDArray` - New array with element-wise maximum values.

### Examples

```php
$a = NDArray::array([1, 5, 3, 8]);
$b = NDArray::array([2, 4, 6, 7]);
$result = $a->maximum($b);
print_r($result->toArray());
// Output: [2, 5, 6, 8]

// With broadcasting
$a = NDArray::array([[1, 2, 3], [4, 5, 6]]);
$b = NDArray::array([2, 2, 2]);
$result = $a->maximum($b);
print_r($result->toArray());
// Output: [[2, 2, 3], [4, 5, 6]]
```

---

## sigmoid()

```php
public function sigmoid(): NDArray
```

Compute sigmoid element-wise: 1 / (1 + exp(-x)).

### Returns

- `NDArray` - New array with sigmoid values.

### Examples

```php
$arr = NDArray::array([0.0, 1.0, -1.0, 2.0, -2.0]);
$result = $arr->sigmoid();
print_r($result->toArray());
// Output: [0.5, 0.731..., 0.268..., 0.880..., 0.119...]
```

---

## softmax()

```php
public function softmax(int $axis = -1): NDArray
```

Compute softmax along axis: exp(x - max) / sum(exp(x - max)).

Numerically stable. Default axis -1 (last axis) for typical logits.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$axis` | `int` | Axis along which to compute softmax. Optional. Default: `-1`. |

### Returns

- `NDArray` - New array with softmax values (sums to 1 along axis).

### Examples

```php
$arr = NDArray::array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
$result = $arr->softmax(axis: 1);
print_r($result->toArray());
// Output: [[0.090..., 0.244..., 0.665...], [0.090..., 0.244..., 0.665...]]
```

---

## Summary Table

### Arithmetic Operations

| Method | Operation | Example |
|--------|-----------|---------|
| `add()` | Addition | `[1, 2] + [3, 4]` → `[4, 6]` |
| `subtract()` | Subtraction | `[5, 6] - [1, 2]` → `[4, 4]` |
| `multiply()` | Multiplication | `[2, 3] * [4, 5]` → `[8, 15]` |
| `divide()` | Division | `[10, 20] / [2, 4]` → `[5, 5]` |
| `rem()` / `mod()` | Modulo | `[10, 15] % [3, 4]` → `[1, 3]` |
| `abs()` | Absolute value | `abs([-1, -2])` → `[1, 2]` |
| `negative()` | Negation | `-[1, -2]` → `[-1, 2]` |

### Powers and Roots

| Method | Operation | Example |
|--------|-----------|---------|
| `sqrt()` | Square root | `sqrt([4, 9])` → `[2, 3]` |
| `cbrt()` | Cube root | `cbrt([8, 27])` → `[2, 3]` |
| `pow2()` | Square | `pow2([2, 3])` → `[4, 9]` |
| `powi()` | Integer power | `powi([2, 3], 3)` → `[8, 27]` |
| `powf()` | Float power | `powf([4, 9], 0.5)` → `[2, 3]` |
| `recip()` | Reciprocal | `recip([2, 4])` → `[0.5, 0.25]` |

### Exponentials and Logarithms

| Method | Operation | Example |
|--------|-----------|---------|
| `exp()` | e^x | `exp([0, 1])` → `[1, 2.718]` |
| `exp2()` | 2^x | `exp2([0, 3])` → `[1, 8]` |
| `log()` / `ln()` | Natural log | `log([1, 2.718])` → `[0, 1]` |
| `log2()` | Base-2 log | `log2([1, 8])` → `[0, 3]` |
| `log10()` | Base-10 log | `log10([1, 100])` → `[0, 2]` |
| `ln1p()` | ln(1+x) | `ln1p([0, 0.1])` → `[0, 0.095]` |

### Trigonometric Functions

| Method | Operation | Example |
|--------|-----------|---------|
| `sin()` | Sine | `sin([0, π/2])` → `[0, 1]` |
| `cos()` | Cosine | `cos([0, π])` → `[1, -1]` |
| `tan()` | Tangent | `tan([0, π/4])` → `[0, 1]` |
| `asin()` | Arc sine | `asin([0, 1])` → `[0, π/2]` |
| `acos()` | Arc cosine | `acos([1, 0])` → `[0, π/2]` |
| `atan()` | Arc tangent | `atan([0, 1])` → `[0, π/4]` |
| `sinh()` | Hyperbolic sine | `sinh([0, 1])` → `[0, 1.175]` |
| `cosh()` | Hyperbolic cosine | `cosh([0, 1])` → `[1, 1.543]` |
| `tanh()` | Hyperbolic tangent | `tanh([0, 1])` → `[0, 0.762]` |
| `toDegrees()` | Radians to degrees | `toDegrees([0, π])` → `[0, 180]` |
| `toRadians()` | Degrees to radians | `toRadians([0, 90])` → `[0, π/2]` |

### Rounding Functions

| Method | Operation | Example |
|--------|-----------|---------|
| `floor()` | Round down | `floor([1.7, -1.2])` → `[1, -2]` |
| `ceil()` | Round up | `ceil([1.2, -1.7])` → `[2, -1]` |
| `round()` | Round nearest | `round([1.4, 1.5])` → `[1, 2]` |

### Miscellaneous

| Method | Operation | Example |
|--------|-----------|---------|
| `signum()` | Sign | `signum([-5, 0, 3])` → `[-1, 0, 1]` |
| `hypot()` | Hypotenuse | `hypot([3, 4], 4)` → `[5, 5.657]` |
| `clamp()` / `clip()` | Clip range | `clamp([-5, 5, 15], 0, 10)` → `[0, 5, 10]` |
| `minimum()` | Element-wise minimum | `minimum([1, 5], [2, 4])` → `[1, 4]` |
| `maximum()` | Element-wise maximum | `maximum([1, 5], [2, 4])` → `[2, 5]` |
| `sigmoid()` | Sigmoid | `sigmoid([0, 1])` → `[0.5, 0.731]` |
| `softmax()` | Softmax | `softmax([1, 2, 3])` → `[0.09, 0.24, 0.67]` |

---

## Next Steps

- [Logic Functions](/api/logic-functions) - Comparison operations
- [Bitwise Operations](/api/bitwise-operations) - Bitwise operations on integers
- [Statistics](/api/statistics) - Statistical reductions