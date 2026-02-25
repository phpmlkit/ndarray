# Statistics

Reference for statistical reduction operations.

These methods aggregate array values to compute statistical measures like sum, mean, variance, and standard deviation.

---

## sum()

```php
public function sum(?int $axis = null, bool $keepdims = false): float|int|NDArray
```

Sum of array elements over a given axis.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$axis` | `int\|null` | Axis along which to sum. If null, sum over all elements. Optional. Default: `null`. |
| `$keepdims` | `bool` | If true, the reduced axis is retained with size 1. Optional. Default: `false`. |

### Returns

- `float|int|NDArray` - Scalar if axis is null, otherwise an NDArray.

### Examples

```php
$arr = NDArray::array([[1, 2, 3], [4, 5, 6]]);

// Sum all elements
$total = $arr->sum();
echo $total;
// Output: 21

// Sum along axis 0 (columns)
$col_sums = $arr->sum(axis: 0);
print_r($col_sums->toArray());
// Output: [5, 7, 9]

// Sum along axis 1 (rows)
$row_sums = $arr->sum(axis: 1);
print_r($row_sums->toArray());
// Output: [6, 15]

// Keep dimensions
$sums = $arr->sum(axis: 1, keepdims: true);
print_r($sums->shape());
// Output: [2, 1]
```

---

## mean()

```php
public function mean(?int $axis = null, bool $keepdims = false): float|NDArray
```

Mean of array elements over a given axis.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$axis` | `int\|null` | Axis along which to compute mean. If null, compute mean of all elements. Optional. Default: `null`. |
| `$keepdims` | `bool` | If true, the reduced axis is retained with size 1. Optional. Default: `false`. |

### Returns

- `float|NDArray` - Scalar if axis is null, otherwise an NDArray.

### Examples

```php
$arr = NDArray::array([[1, 2, 3], [4, 5, 6]]);

echo $arr->mean();
// Output: 3.5

$row_means = $arr->mean(axis: 1);
print_r($row_means->toArray());
// Output: [2.0, 5.0]
```

---

## var()

```php
public function var(?int $axis = null, int $ddof = 0, bool $keepdims = false): float|NDArray
```

Variance of array elements over a given axis.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$axis` | `int\|null` | Axis along which to compute variance. If null, compute variance of all elements. Optional. Default: `null`. |
| `$ddof` | `int` | Delta degrees of freedom (0 for population, 1 for sample). Optional. Default: `0`. |
| `$keepdims` | `bool` | If true, the reduced axis is retained with size 1. Optional. Default: `false`. |

### Returns

- `float|NDArray` - Scalar if axis is null, otherwise an NDArray.

### Examples

```php
$arr = NDArray::array([1, 2, 3, 4, 5]);

echo $arr->var();
// Output: 2.0 (population variance)

echo $arr->var(ddof: 1);
// Output: 2.5 (sample variance)
```

---

## std()

```php
public function std(?int $axis = null, int $ddof = 0, bool $keepdims = false): float|NDArray
```

Standard deviation of array elements over a given axis.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$axis` | `int\|null` | Axis along which to compute std. If null, compute std of all elements. Optional. Default: `null`. |
| `$ddof` | `int` | Delta degrees of freedom (0 for population, 1 for sample). Optional. Default: `0`. |
| `$keepdims` | `bool` | If true, the reduced axis is retained with size 1. Optional. Default: `false`. |

### Returns

- `float|NDArray` - Scalar if axis is null, otherwise an NDArray.

### Examples

```php
$arr = NDArray::array([1, 2, 3, 4, 5]);

echo $arr->std();
// Output: 1.414... (population std)

echo $arr->std(ddof: 1);
// Output: 1.581... (sample std)
```

---

## min()

```php
public function min(?int $axis = null, bool $keepdims = false): float|int|NDArray
```

Minimum of array elements over a given axis.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$axis` | `int\|null` | Axis along which to find minimum. If null, find minimum of all elements. Optional. Default: `null`. |
| `$keepdims` | `bool` | If true, the reduced axis is retained with size 1. Optional. Default: `false`. |

### Returns

- `float|int|NDArray` - Scalar if axis is null, otherwise an NDArray.

### Examples

```php
$arr = NDArray::array([[1, 2, 3], [4, 5, 6]]);

echo $arr->min();
// Output: 1

$col_mins = $arr->min(axis: 0);
print_r($col_mins->toArray());
// Output: [1, 2, 3]
```

---

## max()

```php
public function max(?int $axis = null, bool $keepdims = false): float|int|NDArray
```

Maximum of array elements over a given axis.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$axis` | `int\|null` | Axis along which to find maximum. If null, find maximum of all elements. Optional. Default: `null`. |
| `$keepdims` | `bool` | If true, the reduced axis is retained with size 1. Optional. Default: `false`. |

### Returns

- `float|int|NDArray` - Scalar if axis is null, otherwise an NDArray.

### Examples

```php
$arr = NDArray::array([[1, 2, 3], [4, 5, 6]]);

echo $arr->max();
// Output: 6

$col_maxs = $arr->max(axis: 0);
print_r($col_maxs->toArray());
// Output: [4, 5, 6]
```

---

## product()

```php
public function product(?int $axis = null, bool $keepdims = false): float|int|NDArray
```

Product of array elements over a given axis.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$axis` | `int\|null` | Axis along which to compute product. If null, compute product of all elements. Optional. Default: `null`. |
| `$keepdims` | `bool` | If true, the reduced axis is retained with size 1. Optional. Default: `false`. |

### Returns

- `float|int|NDArray` - Scalar if axis is null, otherwise an NDArray.

### Examples

```php
$arr = NDArray::array([[1, 2, 3], [4, 5, 6]]);

echo $arr->product();
// Output: 720

$row_prods = $arr->product(axis: 1);
print_r($row_prods->toArray());
// Output: [6, 120]
```

---

## Summary Table

| Method | Description | Returns |
|--------|-------------|---------|
| `sum()` | Sum of elements | Scalar or array |
| `mean()` | Arithmetic mean | Scalar or array |
| `var()` | Variance | Scalar or array |
| `std()` | Standard deviation | Scalar or array |
| `min()` | Minimum value | Scalar or array |
| `max()` | Maximum value | Scalar or array |
| `product()` | Product of elements | Scalar or array |

---

## Next Steps

- [Sorting and Searching](/api/sorting-searching)
- [Mathematical Functions](/api/mathematical-functions)
- [Linear Algebra](/api/linear-algebra)