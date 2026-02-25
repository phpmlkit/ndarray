# Array Creation

Complete reference for array creation methods and related utilities.

## NDArray::array()

Create an NDArray from a PHP array.

```php
public static function array(array $data,?DType $dtype = null): self
```

Create arrays from PHP nested arrays.

**Parameters:**
- `array $data` - PHP array containing data
- `?DType $dtype` - Optional data type. If null, inferred from data

**Returns:** NDArray with same shape as input array

**Examples:**

```php
// 1D array
$arr = NDArray::array([1, 2, 3, 4, 5]);
echo implode(',', $arr->shape());  // 5
echo $arr->dtype()->name;  // Float64

// 2D array
$matrix = NDArray::array([
    [1, 2, 3],
    [4, 5, 6]
]);
echo implode(',', $matrix->shape());  // 2,3

// 3D array
$tensor = NDArray::array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
]);
echo implode(',', $tensor->shape());  // 2,2,2

// With specific type
$int_arr = NDArray::array([1, 2, 3], DType::Int32);
echo $int_arr->dtype()->name;  // Int32

// With mixed types (promoted to Float64)
$mixed = NDArray::array([1, 2.5, 3]);
echo $mixed->dtype()->name;  // Float64
```

**Notes:**
- All elements are converted to the specified or inferred type
- Nested arrays must have consistent dimensions
- Empty arrays are not allowed

**See Also:**
- [Data Types](/guide/fundamentals/data-types)
- [Array Creation Guide](/guide/operations/creation)

---

## NDArray::zeros()

Create an array filled with zeros.

```php
public static function zeros(array $shape, DType $dtype = DType::Float64): self
```

**Parameters:**
- `array $shape` - Array dimensions [rows, cols, ...]
- `DType $dtype` - Data type (default: Float64)

**Examples:**

```php
// 1D
$zeros = NDArray::zeros([5]);
echo $zeros;  // [0. 0. 0. 0. 0.]

// 2D
$zeros = NDArray::zeros([3, 3]);
echo $zeros;
// [[0. 0. 0.]
//  [0. 0. 0.]
//  [0. 0. 0.]]

// 3D
$zeros = NDArray::zeros([2, 3, 4]);

// With Int32 type
$zeros = NDArray::zeros([10], DType::Int32);
```

**See Also:**
- [ones()](#ndarray-ones)
- [full()](#ndarray-full)
- [empty()](#ndarray-empty)

---

## NDArray::ones()

Create an array filled with ones.

```php
public static function ones(array $shape, DType $dtype = DType::Float64): self
```

**Parameters:**
- `array $shape` - Array dimensions
- `DType $dtype` - Data type (default: Float64)

**Examples:**

```php
$ones = NDArray::ones([2, 3]);
echo $ones;
// [[1. 1. 1.]
//  [1. 1. 1.]]

// Useful for multiplicative operations
$data = NDArray::random([100, 100]);
$multiplier = NDArray::ones([100, 100])->multiply(2);
```

---

## NDArray::full()

Create an array filled with a specific value.

```php
public static function full(
    array $shape,
    float|int|bool $value,
    ?DType $dtype = null
): self
```

**Parameters:**
- `array $shape` - Array dimensions
- `float|int $fillValue` - Value to fill array with
- `?DType $dtype` - Data type (inferred from fillValue if null)

**Examples:**

```php
// Fill with 5
$full = NDArray::full([2, 2], 5);
echo $full;
// [[5. 5.]
//  [5. 5.]]

// Fill with 3.14
$pi_matrix = NDArray::full([3, 3], 3.14);

// With specific type
$full = NDArray::full([10], 100, DType::Int32);
```

**See Also:**
- [zeros()](#ndarray-zeros)
- [ones()](#ndarray-ones)

---

## NDArray::empty()

Create an uninitialized array.

```php
public static function empty(
    array $shape,
    DType $dtype = DType::Float64
): self
```

::: warning Important
`empty()` allocates memory but does **not** initialize it. Values are undefined (whatever was in memory). Only use if you will fill all elements immediately.
:::

**Parameters:**
- `array $shape` - Array dimensions
- `DType $dtype` - Data type (default: Float64)

**Examples:**

```php
// Fast allocation (memory uninitialized)
$buffer = NDArray::empty([1000, 1000]);

// You MUST fill all values before reading
for ($i = 0; $i < 1000; $i++) {
    for ($j = 0; $j < 1000; $j++) {
        $buffer->set([$i, $j], calculateValue($i, $j));
    }
}
```

**Performance:** Fastest allocation method since memory is not initialized.

**Common Mistake:**
```php
// WRONG: Reading uninitialized values
$arr = NDArray::empty([10]);
echo $arr[0];  // Undefined value!

// CORRECT: Fill before reading
$arr = NDArray::empty([10]);
for ($i = 0; $i < 10; $i++) {
    $arr[$i] = $i;
}
echo $arr[0];  // 0
```

---

## NDArray::zerosLike()

Create an array of zeros with the same shape as the input array.

```php
public static function zerosLike(self $array, ?DType $dtype = null): self
```

**Parameters:**
- `self $array` - Input array defining the output shape
- `?DType $dtype` - Data type (default: same as input array)

**Examples:**

```php
$original = NDArray::array([[1, 2], [3, 4]], DType::Int32);
$zeros = NDArray::zerosLike($original);
echo $zeros->shape();  // [2, 2]
echo $zeros->dtype();  // Int32
echo $zeros;
// [[0 0]
//  [0 0]]

// Override dtype
$zeros = NDArray::zerosLike($original, DType::Float64);
// Float64
```

**See Also:**
- [onesLike()](#ndarray-oneslike)
- [fullLike()](#ndarray-fulllike)
- [zeros()](#ndarray-zeros)

---

## NDArray::onesLike()

Create an array of ones with the same shape as the input array.

```php
public static function onesLike(self $array, ?DType $dtype = null): self
```

**Parameters:**
- `self $array` - Input array defining the output shape
- `?DType $dtype` - Data type (default: same as input array)

**Examples:**

```php
$original = NDArray::array([[1, 2], [3, 4]], DType::Int32);
$ones = NDArray::onesLike($original);
echo $ones->shape();  // [2, 2]
echo $ones->dtype();  // Int32
echo $ones;
// [[1 1]
//  [1 1]]
```

**See Also:**
- [zerosLike()](#ndarray-zeroslike)
- [fullLike()](#ndarray-fulllike)
- [ones()](#ndarray-ones)

---

## NDArray::fullLike()

Create an array filled with a specific value, with the same shape as the input array.

```php
public static function fullLike(
    self $array,
    float|int|bool $value,
    ?DType $dtype = null
): self
```

**Parameters:**
- `self $array` - Input array defining the output shape
- `float|int|bool $value` - Value to fill array with
- `?DType $dtype` - Data type (default: same as input array, or inferred from value)

**Examples:**

```php
$original = NDArray::array([[1, 2], [3, 4]], DType::Int32);

// Same dtype as input
$full = NDArray::fullLike($original, 5);
echo $full;
// [[5 5]
//  [5 5]]

// Override dtype
$full = NDArray::fullLike($original, 3.14, DType::Float64);
echo $full->dtype();  // Float64
echo $full;
// [[3.14 3.14]
//  [3.14 3.14]]
```

**See Also:**
- [zerosLike()](#ndarray-zeroslike)
- [onesLike()](#ndarray-oneslike)
- [full()](#ndarray-full)

---

## NDArray::eye()

Create an identity matrix.

```php
public static function eye(
    int $n,
    ?int $m = null,
    int $k = 0,
    DType $dtype = DType::Float64
): self
```

**Parameters:**
- `int $n` - Number of rows
- `?int $m` - Number of columns (defaults to $n)
- `int $k` - Index of diagonal (0=main, positive=upper, negative=lower)
- `DType $dtype` - Data type (default: Float64)

**Examples:**

```php
// Square identity matrix
$eye = NDArray::eye(3);
echo $eye;
// [[1. 0. 0.]
//  [0. 1. 0.]
//  [0. 0. 1.]]

// Rectangular
$rect = NDArray::eye(3, 5);
echo $rect;
// [[1. 0. 0. 0. 0.]
//  [0. 1. 0. 0. 0.]
//  [0. 0. 1. 0. 0.]]

// Upper diagonal
$upper = NDArray::eye(3, k: 1);
echo $upper;
// [[0. 1. 0.]
//  [0. 0. 1.]
//  [0. 0. 0.]]

// Lower diagonal
$lower = NDArray::eye(3, k: -1);
echo $lower;
// [[0. 0. 0.]
//  [1. 0. 0.]
//  [0. 1. 0.]]
```

**See Also:**
- [diagonal()](/api/linear-algebra#diagonal)
- [Linear Algebra Guide](/guide/operations/linear-algebra)

---

## NDArray::arange()

Create evenly spaced values within a given interval.

```php
public static function arange(
    int|float $start,
    int|float $stop = null,
    int|float $step = 1,
    ?DType $dtype = null
): self
```

**Parameters:**
- `int|float $start` - Start value (inclusive)
- `int|float $stop` - Stop value (exclusive). If null, start becomes 0 and stop becomes start
- `int|float $step` - Spacing between values (default: 1)
- `?DType $dtype` - Data type (inferred if null)

**Examples:**

```php
// 0 to 9
$arr = NDArray::arange(10);
echo $arr;  // [0 1 2 3 4 5 6 7 8 9]

// 5 to 14
$arr = NDArray::arange(5, 15);
echo $arr;  // [5 6 7 8 9 10 11 12 13 14]

// Even numbers
$evens = NDArray::arange(0, 10, 2);
echo $evens;  // [0 2 4 6 8]

// Decrementing
$down = NDArray::arange(10, 0, -1);
echo $down;  // [10 9 8 7 6 5 4 3 2 1]

// Float step
$floats = NDArray::arange(0, 1, 0.1);
echo $floats;  // [0. 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]
```

**See Also:**
- [linspace()](#ndarray-linspace)
- [logspace()](#ndarray-logspace)

---

## NDArray::linspace()

Create linearly spaced values.

```php
public static function linspace(
    float $start,
    float $stop,
    int $num = 50,
    bool $endpoint = true,
    ?DType $dtype = null
): self
```

**Parameters:**
- `float $start` - Starting value
- `float $stop` - Ending value
- `int $num` - Number of samples (default: 50)
- `bool $endpoint` - Include stop value (default: true)
- `?DType $dtype` - Data type (default: Float64)

**Examples:**

```php
// 5 values from 0 to 1
$arr = NDArray::linspace(0, 1, 5);
echo $arr;  // [0. 0.25 0.5 0.75 1.]

// Exclude endpoint
$arr = NDArray::linspace(0, 1, 5, endpoint: false);
echo $arr;  // [0. 0.2 0.4 0.6 0.8]

// Create time points
$time = NDArray::linspace(0, 10, 1000);
```

**Difference from arange():**
- `arange()` uses step size, excludes stop
- `linspace()` uses number of points, includes stop by default

---

## NDArray::logspace()

Create logarithmically spaced values.

```php
public static function logspace(
    float $start,
    float $stop,
    int $num = 50,
    float $base = 10.0,
    ?DType $dtype = null
): self
```

**Parameters:**
- `float $start` - Start exponent (base**start)
- `float $stop` - Stop exponent (base**stop)
- `int $num` - Number of samples (default: 50)
- `float $base` - Base of log space (default: 10.0)
- `?DType $dtype` - Data type (default: Float64)

**Examples:**

```php
// 10^0 to 10^2
$arr = NDArray::logspace(0, 2, 5);
echo $arr;  // [1. 3.162 10. 31.623 100.]

// Base 2
$powers = NDArray::logspace(0, 10, 11, base: 2);
echo $powers;  // [1. 2. 4. 8. 16. 32. 64. 128. 256. 512. 1024.]
```

---

## NDArray::geomspace()

Create geometrically spaced values.

```php
public static function geomspace(
    float $start,
    float $stop,
    int $num = 50,
    ?DType $dtype = null
): self
```

**Parameters:**
- `float $start` - Starting value
- `float $stop` - Ending value
- `int $num` - Number of samples (default: 50)
- `?DType $dtype` - Data type (default: Float64)

**Examples:**

```php
// Geometric progression from 1 to 1000
$arr = NDArray::geomspace(1, 1000, 4);
echo $arr;  // [1. 10. 100. 1000.]
```

---

## NDArray::random()

Create array with uniform random values in [0, 1).

```php
public static function random(
    array $shape,
    DType $dtype = DType::Float64
): self
```

**Examples:**

```php
$random = NDArray::random([3, 3]);
echo $random;
// [[0.234 0.891 0.456]
//  [0.123 0.678 0.901]
//  [0.567 0.345 0.789]]

// Float32 for ML
$weights = NDArray::random([784, 256], DType::Float32);
```

---

## NDArray::randn()

Create array with standard normal distribution (mean=0, std=1).

```php
public static function randn(
    array $shape,
    DType $dtype = DType::Float64
): self
```

**Examples:**

```php
$normal = NDArray::randn([1000]);
$mean = $normal->mean();  // ~0
$std = $normal->std();    // ~1
```

---

## NDArray::normal()

Create array with normal distribution.

```php
public static function normal(
    float $mean,
    float $std,
    array $shape,
    DType $dtype = DType::Float64
): self
```

**Examples:**

```php
// Mean=10, Std=2
$dist = NDArray::normal(10, 2, [1000]);
echo $dist->mean();  // ~10
echo $dist->std();   // ~2
```

---

## NDArray::uniform()

Create array with uniform distribution in [low, high).

```php
public static function uniform(
    float $low,
    float $high,
    array $shape,
    DType $dtype = DType::Float64
): self
```

**Examples:**

```php
// Values between -1 and 1
$uniform = NDArray::uniform(-1, 1, [100]);
```

---

## NDArray::randomInt()

Create array with random integers.

```php
public static function randomInt(
    int $low,
    int $high,
    array $shape,
    DType $dtype = DType::Int64
): self
```

**Parameters:**
- `int $low` - Lower bound (inclusive)
- `int $high` - Upper bound (exclusive)
- `array $shape` - Array dimensions
- `DType $dtype` - Integer type (default: Int64)

**Examples:**

```php
// Random integers 0-99
$ints = NDArray::randomInt(0, 100, [1000]);

// Dice rolls
$dice = NDArray::randomInt(1, 7, [100]);  // 1-6

// For images (0-255)
$pixels = NDArray::randomInt(0, 256, [224, 224, 3], DType::UInt8);
```

---

## copy()

Create a deep copy of the array.

```php
public function copy(): self
```

The returned array is always C-contiguous and owns its data. Modifying the copy does not affect the original.

### Parameters

No parameters.

### Returns

- `NDArray` - Independent copy with new memory.

### Examples

```php
$original = NDArray::array([1, 2, 3]);
$copy = $original->copy();

// Modifying copy doesn't affect original
$copy->set([0], 999);
echo $original->get(0);  // 1 (unchanged)
```

---

## astype()

Convert array to a different data type.

```php
public function astype(DType $dtype): self
```

Returns a new array with the specified dtype. If the target dtype is the same as the current dtype, this is equivalent to `copy()`.

### Parameters

| Name | Type | Description |
|------|------|-------------|
| `$dtype` | `DType` | Target data type |

### Returns

- `NDArray` - New array with converted data (same shape).

### Examples

```php
$floats = NDArray::array([1.5, 2.7, 3.2]);
$ints = $floats->astype(DType::Int32);
print_r($ints->toArray());  // [1, 2, 3]
```

---

## Summary Table

| Method | Purpose | Use Case |
|--------|---------|----------|
| `array()` | From PHP array | Import existing data |
| `zeros()` | Filled with zeros | Initialization |
| `ones()` | Filled with ones | Multiplicative identity |
| `full()` | Filled with value | Specific constant |
| `empty()` | Uninitialized | Pre-allocation (must fill) |
| `zerosLike()` | Zeros like input | Same shape as array |
| `onesLike()` | Ones like input | Same shape as array |
| `fullLike()` | Filled like input | Same shape as array |
| `eye()` | Identity matrix | Linear algebra |
| `arange()` | Evenly spaced | Integer sequences |
| `linspace()` | Linear spacing | Continuous ranges |
| `logspace()` | Logarithmic spacing | Exponential ranges |
| `geomspace()` | Geometric spacing | Multiplicative sequences |
| `random()` | Uniform [0,1) | General random |
| `randn()` | Standard normal | Statistical data |
| `normal()` | Custom normal | Statistical distributions |
| `uniform()` | Uniform range | Bounded random |
| `randomInt()` | Random integers | Discrete random |
| `copy()` | Deep copy | Independent array from existing |
| `astype()` | Type conversion | New array with different dtype |

## Next Steps

- [NDArray Class](/api/ndarray-class)
- [Array Creation Guide](/guide/operations/creation)
- [Data Types](/guide/fundamentals/data-types)
