# Indexing Routines

Methods for accessing and manipulating array elements.

## get()

```php
public function get(int ...$indices): bool|float|int|NDArray
```

Access elements by index.

Full indices (count === ndim) return a scalar via FFI read. Partial indices (count < ndim) return a view (pure PHP, zero FFI).

Supports negative indices: -1 refers to the last element, -2 to second-to-last, etc.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$indices` | `int...` | One or more dimension indices. |

### Returns

- `bool|float|int|NDArray` - Scalar for full indexing, view for partial indexing.

### Raises

- `IndexException` - If no indices provided or too many indices for array dimensions.

### Examples

```php
$arr = NDArray::array([[1, 2, 3], [4, 5, 6]]);

// Full indexing - returns scalar
$value = $arr->get(0, 1);
echo $value;
// Output: 2

// Negative indexing
$value = $arr->get(-1, -1);
echo $value;
// Output: 6

// Partial indexing - returns view
$row = $arr->get(0);
print_r($row->toArray());
// Output: [1, 2, 3]
```

## set()

```php
public function set(array $indices, bool|float|int $value): void
```

Set a scalar value at the given indices.

Requires full indexing (count === ndim).

Supports negative indices: -1 refers to the last element, -2 to second-to-last, etc.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$indices` | `array<int>` | Indices for each dimension. |
| `$value` | `bool\|float\|int` | Value to set. |

### Raises

- `IndexException` - If number of indices doesn't match array dimensions.

### Examples

```php
$arr = NDArray::array([[1, 2, 3], [4, 5, 6]]);

// Set single element
$arr->set([0, 1], 99);
print_r($arr->toArray());
// Output: [[1, 99, 3], [4, 5, 6]]

// With negative indices
$arr->set([-1, -1], 100);
print_r($arr->toArray());
// Output: [[1, 99, 3], [4, 5, 100]]
```

## setAt()

```php
public function setAt(int $flatIndex, bool|float|int $value): void
```

Set a scalar value using a logical flat index (C-order) for this array/view.

Supports negative indices: -1 refers to the last logical element.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$flatIndex` | `int` | Logical flat index into this array/view. |
| `$value` | `bool\|float\|int` | Value to set. |

### Raises

- `IndexException` - If flat index is out of bounds.

### Examples

```php
$arr = NDArray::array([[1, 2, 3], [4, 5, 6]]);

// Set using flat index (C-order)
$arr->setAt(0, 100);
print_r($arr->toArray());
// Output: [[100, 2, 3], [4, 5, 6]]

$arr->setAt(5, 200);
print_r($arr->toArray());
// Output: [[100, 2, 3], [4, 5, 200]]

// With negative index
$arr->setAt(-1, 300);
print_r($arr->toArray());
// Output: [[100, 2, 3], [4, 5, 300]]
```

---

## getAt()

```php
public function getAt(int $flatIndex): bool|float|int
```

Get a scalar value using a logical flat index (C-order) for this array/view.

Supports negative indices: -1 refers to the last logical element.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$flatIndex` | `int` | Logical flat index into this array/view. |

### Returns

- `bool|float|int` - The value at the specified flat index.

### Raises

- `IndexException` - If flat index is out of bounds.

### Examples

```php
$arr = NDArray::array([[1, 2, 3], [4, 5, 6]]);

// Get using flat index (C-order)
echo $arr->getAt(0);   // 1
echo $arr->getAt(3);   // 4
echo $arr->getAt(5);   // 6

// With negative index
echo $arr->getAt(-1);  // 6 (last element)
echo $arr->getAt(-3);  // 4 (third from last)
```

### See Also

- [setAt()](#setat) - Set value at flat index
- [flat()](/api/array-creation#ndarrayflat) - Iterator for flat access

---

## take()

```php
public function take(array|NDArray $indices, ?int $axis = null): NDArray
```

Take elements from an array along an axis.

When `axis` is null, this function gathers elements from the flattened array. When `axis` is provided, it selects entire slices along that axis.

**Important**: This is different from `takeAlongAxis()`. Use `take()` to select entire rows/columns/slices. Use `takeAlongAxis()` for per-position indexing.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$indices` | `array\|NDArray` | Array of indices or NDArray containing indices. |
| `$axis` | `int\|null` | Axis along which to select values. If null, flattens first. Optional. Default: `null`. |

### Returns

- `NDArray` - Gathered values. When axis is null, shape matches indices shape. When axis is provided, shape is `input_shape[:axis] + indices_shape + input_shape[axis+1:]`.

### Examples

**Flat indexing (axis is null):**

```php
$arr = NDArray::array([10, 20, 30, 40, 50]);

// Take from flat array
$result = $arr->take([0, 2, 4]);
print_r($result->toArray());
// Output: [10, 30, 50]
```

**Selecting rows (axis=0):**

```php
$matrix = NDArray::array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]);

// Select rows 0 and 2
$rows = $matrix->take([0, 2], axis: 0);
print_r($rows->toArray());
// Output: [[1, 2, 3], [7, 8, 9]]
```

**Selecting columns (axis=1):**

```php
$matrix = NDArray::array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]);

// Select columns 0 and 2
$cols = $matrix->take([0, 2], axis: 1);
print_r($cols->toArray());
// Output: [[1, 3], [4, 6], [7, 9]]
```

## takeAlongAxis()

```php
public function takeAlongAxis(NDArray $indices, int $axis): NDArray
```

Take values from the input array by matching 1D index and data slices along the specified axis.

**Important**: This is different from `take()`. `takeAlongAxis()` uses the indices array to look up values per-position along the axis. The indices array must have the same number of dimensions as the input array, and all dimensions except the specified axis must match.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$indices` | `NDArray` | Int64 indices array. Must match input dimensions (except the axis dimension can differ). |
| `$axis` | `int` | Axis along which to take. |

### Returns

- `NDArray` - Gathered values with same shape as indices.

### Raises

- `InvalidArgumentException` - If indices array doesn't have Int64 dtype or shapes don't match.

### Examples

**Per-position indexing along axis 1:**

```php
$arr = NDArray::array([[10, 20, 30], [40, 50, 60]]);

// For each row, pick elements at positions [2, 1] and [0, 2] respectively
$indices = NDArray::array([[2, 1], [0, 2]], dtype: DType::Int64);
$result = $arr->takeAlongAxis($indices, axis: 1);
print_r($result->toArray());
// Output: [[30, 20], [40, 60]]
// Row 0: elements at positions 2 and 1 -> [30, 20]
// Row 1: elements at positions 0 and 2 -> [40, 60]
```

**Using with argsort to sort an array:**

```php
$arr = NDArray::array([[3, 1, 2], [6, 4, 5]]);

// Get sort indices
$sortedIndices = $arr->argsort(axis: 1);
// sortedIndices = [[1, 2, 0], [1, 2, 0]] (indices that would sort each row)

// Apply the sort
$sorted = $arr->takeAlongAxis($sortedIndices, axis: 1);
print_r($sorted->toArray());
// Output: [[1, 2, 3], [4, 5, 6]]
```

### See Also

- **[take()](#take)** - Select entire slices along an axis (rows, columns, etc.)

## put()

```php
public function put(
    array|NDArray $indices,
    bool|float|int|NDArray $values,
    string $mode = 'raise'
): NDArray
```

Scatter values by flattened indices and return a mutated copy.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$indices` | `array\|NDArray` | Array of flat indices. |
| `$values` | `bool\|float\|int\|NDArray` | Values to scatter (scalar or array). |
| `$mode` | `string` | Currently supports only 'raise'. Optional. Default: `'raise'`. |

### Returns

- `NDArray` - New array with scattered values.

### Raises

- `InvalidArgumentException` - If mode is not 'raise'.

### Examples

```php
$arr = NDArray::array([10, 20, 30, 40, 50]);

// Put single value at multiple indices
$result = $arr->put([0, 2, 4], 99);
print_r($result->toArray());
// Output: [99, 20, 99, 40, 99]

// Put array values at indices
$result = $arr->put([0, 2, 4], [1, 2, 3]);
print_r($result->toArray());
// Output: [1, 20, 2, 40, 3]
```

## putAlongAxis()

```php
public function putAlongAxis(
    NDArray $indices,
    bool|float|int|NDArray $values,
    int $axis
): NDArray
```

Scatter values along an axis and return a mutated copy.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$indices` | `NDArray` | Int64 indices array. |
| `$values` | `bool\|float\|int\|NDArray` | Values to scatter (scalar or array). |
| `$axis` | `int` | Axis along which to scatter. |

### Returns

- `NDArray` - New array with scattered values.

### Raises

- `InvalidArgumentException` - If indices array doesn't have Int64 dtype.

### Examples

```php
$arr = NDArray::array([[1, 2, 3], [4, 5, 6]]);

$indices = NDArray::array([[0, 2, 1], [2, 0, 1]], dtype: DType::Int64);
$result = $arr->putAlongAxis($indices, 99, axis: 1);
print_r($result->toArray());
// Output: [[99, 3, 99], [99, 99, 99]]
```

## scatterAdd()

```php
public function scatterAdd(array|NDArray $indices, bool|float|int|NDArray $updates): NDArray
```

Add updates by flattened indices and return a mutated copy.

Supports repeated indices (accumulates values).

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$indices` | `array\|NDArray` | Array of flat indices. |
| `$updates` | `bool\|float\|int\|NDArray` | Values to add (scalar or array). |

### Returns

- `NDArray` - New array with added values.

### Examples

```php
$arr = NDArray::zeros([5]);

// Add at indices (can repeat)
$result = $arr->scatterAdd(
    [0, 0, 1, 1, 1],
    [1, 1, 1, 1, 1]
);
print_r($result->toArray());
// Output: [2, 3, 0, 0, 0]
// Index 0: 1+1=2
// Index 1: 1+1+1=3
```

## where()

```php
public static function where(bool|float|int|NDArray $condition, bool|float|int|NDArray $x, bool|float|int|NDArray $y): NDArray
```

Select values from x and y based on a boolean condition.

Returns elements from `x` where the condition is true and elements from `y` where the condition is false.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$condition` | `bool\|float\|int\|NDArray` | Boolean condition array or scalar. |
| `$x` | `bool\|float\|int\|NDArray` | Values to select where condition is true. |
| `$y` | `bool\|float\|int\|NDArray` | Values to select where condition is false. |

### Returns

- `NDArray` - New array with elements chosen from x or y based on condition.

### Raises

- `InvalidArgumentException` - If operands have incompatible shapes or invalid types.

### Examples

```php
$condition = NDArray::array([true, false, true, false]);
$x = NDArray::array([1, 2, 3, 4]);
$y = NDArray::array([10, 20, 30, 40]);
$result = NDArray::where($condition, $x, $y);
print_r($result->toArray());
// Output: [1, 20, 3, 40]

// Using comparison as condition
$a = NDArray::array([1, 5, 3, 8]);
$result = NDArray::where($a->gt(4), $a, NDArray::zeros([4]));
print_r($result->toArray());
// Output: [0, 5, 0, 8]

// Scalar operands
$arr = NDArray::array([1, 2, 3, 4]);
$result = NDArray::where($arr->gt(2), $arr, 0);
print_r($result->toArray());
// Output: [0, 0, 3, 4]

// Creating a masked array
$data = NDArray::array([1, 2, 3, 4]);
$mask = NDArray::array([true, false, true, false]);
$masked = NDArray::where($mask, $data, NDArray::full([4], -999));
print_r($masked->toArray());
// Output: [1, -999, 3, -999]
```
