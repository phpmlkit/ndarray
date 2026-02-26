# Array Manipulation

Reference for shape manipulation, stacking, splitting, and array transformation operations.

---

## reshape()

Reshape the array to a new shape.

```php
public function reshape(array $newShape, string $order = 'C'): NDArray
```

Returns a new array with the specified shape. Supports both C-order (row-major, order='C') and F-order (column-major, order='F').

### Parameters

| Name | Type | Description |
|------|------|-------------|
| `$newShape` | `array<int>` | New shape |
| `$order` | `string` | Memory layout: 'C' for row-major, 'F' for column-major. Default: `'C'` |

### Returns

- `NDArray` - Reshaped array. This will be a new view object if the array is contiguous; otherwise, it will be a copy. Note there is no guarantee of the memory layout (C- or Fortran- contiguous) of the returned array.

### Examples

```php
$arr = NDArray::arange(12);

// Reshape to 2D
$matrix = $arr->reshape([3, 4]);
print_r($matrix->shape());
// Output: [3, 4]

// Reshape to 3D
$tensor = $arr->reshape([2, 2, 3]);
print_r($tensor->shape());
// Output: [2, 2, 3]
```

---

## transpose()

Transpose the array.

```php
public function transpose(): NDArray
```

For 2D arrays, swaps rows and columns. For nD arrays, reverses the order of all axes.

### Parameters

None.

### Returns

- `NDArray` - A view of the array with its axes permuted. A view is returned whenever the array is contiguous; otherwise a copy is made.

### Examples

```php
$matrix = NDArray::array([
    [1, 2, 3],
    [4, 5, 6]
]);

$transposed = $matrix->transpose();
print_r($transposed->shape());
// Output: [3, 2]

print_r($transposed->toArray());
// Output: [[1, 4], [2, 5], [3, 6]]
```

---

## flatten()

Flatten the array to 1D.

```php
public function flatten(): NDArray
```

Always returns a copy in C-order (row-major).

### Parameters

None.

### Returns

- `NDArray` - Flattened 1D array (copy).

### Examples

```php
$matrix = NDArray::array([[1, 2], [3, 4]]);

$flat = $matrix->flatten();
print_r($flat->toArray());
// Output: [1, 2, 3, 4]
```

---

## ravel()

Ravel the array to 1D.

```php
public function ravel(string $order = 'C'): NDArray
```

Similar to `flatten()` but may return a view if the array is contiguous.

### Parameters

| Name | Type | Description |
|------|------|-------------|
| `$order` | `string` | Memory layout: 'C' for row-major, 'F' for column-major. Default: `'C'` |

### Returns

- `NDArray` - A 1D view of the array if it is contiguous; otherwise, a contiguous 1D copy.

### Examples

```php
$matrix = NDArray::array([[1, 2], [3, 4]]);

$raveled = $matrix->ravel();
print_r($raveled->toArray());
// Output: [1, 2, 3, 4]
```

---

## Flattening Comparison

When you need a one-dimensional representation of an array, you have three options:

| Method | Returns | When to Use |
|--------|---------|-------------|
| `flatten()` | 1D NDArray (copy) | When you need a real array and want to ensure no shared memory |
| `ravel()` | 1D NDArray (view or copy) | When you need a real array and want zero-copy when possible |
| `flat()` | FlatIterator | When you just need to iterate or index elements, not a full array |

**Key Differences:**

**`flatten()`** always creates a new contiguous copy. Changes to the result don't affect the original, and vice versa. This is safest when you need independence.

**`ravel()`** returns a view if the array is already contiguous, avoiding a copy. For non-contiguous arrays (like slices or transposed arrays), it falls back to copying. Use this when performance matters and you can handle either views or copies.

**`flat()`** returns an iterator, not an array. It provides element-by-element access without copying any data, working efficiently even on complex views. However, you get an iterator object, not an NDArray, so you can't use array methods like `mean()` or `sum()` on it directly.

```php
$matrix = NDArray::array([[1, 2], [3, 4]]);

// Get a 1D array (copy)
$copy = $matrix->flatten();
echo $copy->sum();  // Works: 10

// Get a 1D array (view if contiguous)
$view = $matrix->ravel();
echo $view->mean();  // Works: 2.5

// Iterate without copying
foreach ($matrix->flat() as $value) {
    echo $value . " ";
}
// Output: 1 2 3 4
```

---

## swap()

Swap two axes of the array.

```php
public function swap(int $axis1, int $axis2): NDArray
```

### Parameters

| Name | Type | Description |
|------|------|-------------|
| `$axis1` | `int` | First axis to swap |
| `$axis2` | `int` | Second axis to swap |

### Returns

- `NDArray` - Array with swapped axes (view).

### Examples

```php
$tensor = NDArray::arange(24)->reshape([2, 3, 4]);

$result = $tensor->swap(0, 2);
print_r($result->shape());
// Output: [4, 3, 2]
```

---

## permute()

Permute axes of the array.

```php
public function permute(int ...$axes): NDArray
```

Reorders the axes according to the given permutation. For example, permute(1, 0) on a 2D array is equivalent to transpose().

### Parameters

| Name | Type | Description |
|------|------|-------------|
| `$axes` | `int...` | New order of axes (variadic) |

### Returns

- `NDArray` - Array with permuted axes (view).

### Examples

```php
$tensor = NDArray::ones([2, 3, 4, 5]);

$permuted = $tensor->permute(2, 0, 3, 1);
print_r($permuted->shape());
// Output: [4, 2, 5, 3]
```

---

## merge()

Merge axes by combining the take axis into the into axis.

```php
public function merge(int $take, int $into): NDArray
```

If possible, merge the axis `take` into `into`. Returns the merged array.

### Parameters

| Name | Type | Description |
|------|------|-------------|
| `$take` | `int` | Axis to merge from |
| `$into` | `int` | Axis to merge into |

### Returns

- `NDArray` - Array with merged axes (view).

### Examples

```php
$tensor = NDArray::arange(24)->reshape([2, 3, 4]);

$result = $tensor->merge(1, 0);
print_r($result->shape());
// Output: [6, 4]
```

---

## flip()

Reverse the order of elements in an array along the given axis or axes.

```php
public function flip(array|int|null $axes = null): NDArray
```

### Parameters

| Name | Type | Description |
|------|------|-------------|
| `$axes` | `array<int>\|int\|null` | Axis or axes to flip. If null, flip over all axes. |

### Returns

- `NDArray` - Array with reversed elements.

### Examples

```php
$arr = NDArray::array([[1, 2, 3], [4, 5, 6]]);

// Flip single axis
$result = $arr->flip(0);
print_r($result->toArray());
// Output: [[4, 5, 6], [1, 2, 3]]

// Flip with negative index
$result = $arr->flip(-1);
print_r($result->toArray());
// Output: [[3, 2, 1], [6, 5, 4]]

// Flip multiple axes
$result = $arr->flip([0, 1]);
print_r($result->toArray());
// Output: [[6, 5, 4], [3, 2, 1]]

// Flip all axes
$result = $arr->flip();
print_r($result->toArray());
// Output: [[6, 5, 4], [3, 2, 1]]
```

---

## insert()

Insert a new axis at the specified position.

```php
public function insert(int $axis): NDArray
```

The new axis always has length 1.

### Parameters

| Name | Type | Description |
|------|------|-------------|
| `$axis` | `int` | Position where new axis is inserted |

### Returns

- `NDArray` - Array with new axis (view).

### Examples

```php
$arr = NDArray::array([1, 2, 3]);

$expanded = $arr->insert(0);
print_r($expanded->shape());
// Output: [1, 3]

$expanded = $arr->insert(1);
print_r($expanded->shape());
// Output: [3, 1]
```

---

## expandDims()

Expand dimensions by inserting a new axis.

```php
public function expandDims(int $axis): NDArray
```

Alias for insert().

### Parameters

| Name | Type | Description |
|------|------|-------------|
| `$axis` | `int` | Position where new axis is inserted |

### Returns

- `NDArray` - A view of the array with the number of dimensions increased by inserting a new axis of length 1. This is always a view into the original array.

### Examples

```php
$arr = NDArray::array([1, 2, 3]);

$expanded = $arr->expandDims(0);
print_r($expanded->shape());
// Output: [1, 3]
```

---

## squeeze()

Remove axes of length 1 from the array.

```php
public function squeeze(?array $axes = null): NDArray
```

If no axes are specified, removes all length-1 axes.

### Parameters

| Name | Type | Description |
|------|------|-------------|
| `$axes` | `array<int>\|null` | Specific axes to squeeze (null for all). Default: `null` |

### Returns

- `NDArray` - The input array, but with all or a subset of the dimensions of length 1 removed. This is always a view into the original array. Note that if all axes are squeezed, the result is a 0D array and not a scalar.

### Examples

```php
$arr = NDArray::ones([1, 3, 1, 4, 1]);

// Remove all single dimensions
$squeezed = $arr->squeeze();
print_r($squeezed->shape());
// Output: [3, 4]

// Remove specific axis
$squeezed = $arr->squeeze([0]);
print_r($squeezed->shape());
// Output: [3, 1, 4, 1]
```

---

## pad()

Pad an array.

```php
public function pad(
    array|int $padWidth,
    PadMode $mode = PadMode::Constant,
    array|bool|float|int $constantValues = 0
): NDArray
```

### Parameters

| Name | Type | Description |
|------|------|-------------|
| `$padWidth` | `array\|int` | Number of elements to pad on each side of each axis. int: same on all sides of every axis; array{int,int}: [before, after] for all axes; array per-axis: int or [before, after] per axis |
| `$mode` | `PadMode` | Padding mode. Default: `PadMode::Constant` |
| `$constantValues` | `array\|bool\|float\|int` | Constant value for PadMode::Constant. Default: `0` |

### Returns

- `NDArray` - Padded array.

### Examples

```php
$arr = NDArray::array([[1, 2], [3, 4]]);

// Pad with zeros (1 element on each side)
$padded = $arr->pad(1);
print_r($padded->shape());
// Output: [4, 4]

// With constant value
$padded = $arr->pad(1, constantValues: 9);
```

---

## tile()

Construct an array by repeating this array the number of times given by reps.

```php
public function tile(array|int|NDArray $reps): NDArray
```

### Parameters

| Name | Type | Description |
|------|------|-------------|
| `$reps` | `array\|int\|NDArray` | Number of repetitions along each axis |

### Returns

- `NDArray` - Tiled output array.

### Examples

```php
$arr = NDArray::array([[1, 2], [3, 4]]);

// Tile 2 times along axis 0 and 3 times along axis 1
$result = $arr->tile([2, 3]);
print_r($result->shape());
// Output: [4, 6]

// Tile with integer (applies to first axis only)
$result = $arr->tile(2);
print_r($result->shape());
// Output: [4, 2]
```

---

## repeat()

Repeat elements of an array.

```php
public function repeat(array|int|NDArray $repeats, ?int $axis = null): NDArray
```

### Parameters

| Name | Type | Description |
|------|------|-------------|
| `$repeats` | `array\|int\|NDArray` | Number of repetitions for each element |
| `$axis` | `int\|null` | Axis along which to repeat. Default: flattened input. Default: `null` |

### Returns

- `NDArray` - Output array (same shape except along the given axis).

### Examples

```php
$arr = NDArray::array([1, 2, 3]);

// Repeat each element twice
$result = $arr->repeat(2);
print_r($result->toArray());
// Output: [1, 1, 2, 2, 3, 3]

// Repeat along axis
$matrix = NDArray::array([[1, 2], [3, 4]]);
$result = $matrix->repeat(2, axis: 1);
print_r($result->toArray());
// Output: [[1, 1, 2, 2], [3, 3, 4, 4]]
```

---

## concatenate()

Join arrays along an existing axis.

```php
public static function concatenate(array $arrays, int $axis = 0): NDArray
```

All arrays must have the same shape except for the dimension along the axis.

### Parameters

| Name | Type | Description |
|------|------|-------------|
| `$arrays` | `array<NDArray>` | Arrays to concatenate |
| `$axis` | `int` | Axis along which to join. Default: `0` |

### Returns

- `NDArray` - Concatenated array.

### Examples

```php
$a = NDArray::array([[1, 2], [3, 4]]);
$b = NDArray::array([[5, 6], [7, 8]]);

$out = NDArray::concatenate([$a, $b], axis: 0);
print_r($out->shape());
// Output: [4, 2]

$out = NDArray::concatenate([$a, $b], axis: 1);
print_r($out->shape());
// Output: [2, 4]
```

---

## stack()

Stack arrays along a new axis.

```php
public static function stack(array $arrays, int $axis = 0): NDArray
```

All arrays must have identical shapes.

### Parameters

| Name | Type | Description |
|------|------|-------------|
| `$arrays` | `array<NDArray>` | Arrays to stack |
| `$axis` | `int` | Axis in the result at which the arrays are stacked. Default: `0` |

### Returns

- `NDArray` - Stacked array.

### Examples

```php
$a = NDArray::array([1, 2, 3]);
$b = NDArray::array([4, 5, 6]);

$out = NDArray::stack([$a, $b]);
print_r($out->shape());
// Output: [2, 3]
```

---

## vstack()

Stack arrays vertically (along axis 0).

```php
public static function vstack(array $arrays): NDArray
```

Equivalent to concatenate(arrays, axis=0).

### Parameters

| Name | Type | Description |
|------|------|-------------|
| `$arrays` | `array<NDArray>` | Arrays to stack |

### Returns

- `NDArray` - Vertically stacked array.

### Examples

```php
$a = NDArray::array([[1, 2], [3, 4]]);
$b = NDArray::array([[5, 6]]);

$out = NDArray::vstack([$a, $b]);
print_r($out->shape());
// Output: [3, 2]
```

---

## hstack()

Stack arrays horizontally (along axis 1).

```php
public static function hstack(array $arrays): NDArray
```

Equivalent to concatenate(arrays, axis=1).

### Parameters

| Name | Type | Description |
|------|------|-------------|
| `$arrays` | `array<NDArray>` | Arrays to stack |

### Returns

- `NDArray` - Horizontally stacked array.

### Examples

```php
$a = NDArray::array([[1, 2], [3, 4]]);
$b = NDArray::array([[5], [6]]);

$out = NDArray::hstack([$a, $b]);
print_r($out->shape());
// Output: [2, 3]
```

---

## split()

Split array along an axis.

```php
public function split(array|int $indicesOrSections, int $axis = 0): array
```

If `$indicesOrSections` is an integer N, split into N equal parts (axis length must be divisible by N). If it is an array of indices, split at those positions.

### Parameters

| Name | Type | Description |
|------|------|-------------|
| `$indicesOrSections` | `array<int>\|int` | Number of equal parts, or array of split indices |
| `$axis` | `int` | Axis along which to split. Default: `0` |

### Returns

- `array<NDArray>` - List of sub-arrays (views).

### Examples

```php
$arr = NDArray::arange(12)->reshape([3, 4]);

// Split into 3 equal parts along axis 0
$parts = $arr->split(3, axis: 0);
// 3 arrays of shape [1, 4]

// Split at indices
$parts = $arr->split([1, 2], axis: 1);
// 3 arrays of shape [3, 1], [3, 1], [3, 2]
```

---

## vsplit()

Split array vertically (along axis 0).

```php
public function vsplit(array|int $indicesOrSections): array
```

### Parameters

| Name | Type | Description |
|------|------|-------------|
| `$indicesOrSections` | `array<int>\|int` | Number of equal parts or split indices |

### Returns

- `array<NDArray>` - List of sub-arrays.

### Examples

```php
$arr = NDArray::arange(12)->reshape([4, 3]);
$parts = $arr->vsplit(2);
// 2 arrays of shape [2, 3]
```

---

## hsplit()

Split array horizontally (along axis 1).

```php
public function hsplit(array|int $indicesOrSections): array
```

### Parameters

| Name | Type | Description |
|------|------|-------------|
| `$indicesOrSections` | `array<int>\|int` | Number of equal parts or split indices |

### Returns

- `array<NDArray>` - List of sub-arrays.

### Examples

```php
$arr = NDArray::arange(12)->reshape([3, 4]);
$parts = $arr->hsplit(2);
// 2 arrays of shape [3, 2]
```

---

## Summary Table

### Shape Manipulation

| Method | Description | Returns View? |
|--------|-------------|---------------|
| `reshape()` | Change array shape | Sometimes |
| `transpose()` | Reverse axes | Yes |
| `flatten()` | Flatten to 1D (copy) | No |
| `ravel()` | Flatten to 1D (view if possible) | Sometimes |
| `swap()` | Swap two axes | Yes |
| `permute()` | Reorder axes | Yes |
| `merge()` | Merge two axes | Yes |
| `flip()` | Reverse elements | No |
| `insert()` / `expandDims()` | Add dimension | Yes |
| `squeeze()` | Remove length-1 dimensions | Yes |
| `pad()` | Pad with values | No |
| `tile()` | Repeat array | No |
| `repeat()` | Repeat elements | No |

### Joining and Splitting

| Method | Description |
|--------|-------------|
| `concatenate()` | Join along existing axis |
| `stack()` | Join along new axis |
| `vstack()` | Stack vertically (axis 0) |
| `hstack()` | Stack horizontally (axis 1) |
| `split()` | Split along axis |
| `vsplit()` | Split vertically (axis 0) |
| `hsplit()` | Split horizontally (axis 1) |

---

## Next Steps

- [Array Creation](/api/array-creation)
- [Indexing Routines](/api/indexing-routines)
- [Array Import and Export](/api/array-import-export)