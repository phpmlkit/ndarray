# Sorting, Searching, and Counting

Reference for sorting, searching, and counting operations.

These methods help you find values, sort arrays, and count occurrences.

---

## argmin()

```php
public function argmin(?int $axis = null, bool $keepdims = false): int|NDArray
```

Index of minimum value over a given axis.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$axis` | `int\|null` | Axis along which to find argmin. If null, find argmin of flattened array. Optional. Default: `null`. |
| `$keepdims` | `bool` | If true, the reduced axis is retained with size 1. Optional. Default: `false`. |

### Returns

- `int|NDArray` - Scalar index if axis is null, otherwise an NDArray of indices.

### Examples

```php
$arr = NDArray::array([3, 1, 4, 1, 5]);

echo $arr->argmin();
// Output: 1 (first occurrence of 1)

$matrix = NDArray::array([
    [3, 1, 4],
    [1, 5, 9]
]);

print_r($matrix->argmin(axis: 0)->toArray());
// Output: [1, 0, 0]

print_r($matrix->argmin(axis: 1)->toArray());
// Output: [1, 0]
```

---

## argmax()

```php
public function argmax(?int $axis = null, bool $keepdims = false): int|NDArray
```

Index of maximum value over a given axis.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$axis` | `int\|null` | Axis along which to find argmax. If null, find argmax of flattened array. Optional. Default: `null`. |
| `$keepdims` | `bool` | If true, the reduced axis is retained with size 1. Optional. Default: `false`. |

### Returns

- `int|NDArray` - Scalar index if axis is null, otherwise an NDArray of indices.

### Examples

```php
$arr = NDArray::array([3, 1, 4, 1, 5]);

echo $arr->argmax();
// Output: 4

$matrix = NDArray::array([
    [3, 1, 4],
    [1, 5, 9]
]);

print_r($matrix->argmax(axis: 0)->toArray());
// Output: [0, 1, 1]

print_r($matrix->argmax(axis: 1)->toArray());
// Output: [2, 2]
```

---

## sort()

```php
public function sort(?int $axis = -1, SortKind $kind = SortKind::QuickSort): NDArray
```

Return a sorted copy of the array.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$axis` | `int\|null` | Axis along which to sort. If null, sort flattened data. Optional. Default: `-1`. |
| `$kind` | `SortKind` | Sorting algorithm. Optional. Default: `SortKind::QuickSort`. |

### Returns

- `NDArray` - Sorted array.

### Examples

```php
$arr = NDArray::array([3, 1, 4, 1, 5]);
$sorted = $arr->sort();
print_r($sorted->toArray());
// Output: [1, 1, 3, 4, 5]

$matrix = NDArray::array([[3, 1, 2], [6, 4, 5]]);
$sorted_rows = $matrix->sort(axis: 1);
print_r($sorted_rows->toArray());
// Output: [[1, 2, 3], [4, 5, 6]]
```

---

## argsort()

```php
public function argsort(?int $axis = -1, SortKind $kind = SortKind::QuickSort): NDArray
```

Return indices that would sort the array.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$axis` | `int\|null` | Axis along which to argsort. If null, argsort flattened data. Optional. Default: `-1`. |
| `$kind` | `SortKind` | Sorting algorithm. Optional. Default: `SortKind::QuickSort`. |

### Returns

- `NDArray` - Int64 indices array.

### Examples

```php
$arr = NDArray::array([3, 1, 4, 1, 5]);
$indices = $arr->argsort();
print_r($indices->toArray());
// Output: [1, 3, 0, 2, 4]

// Use to sort
$sorted = $arr->take($indices);
print_r($sorted->toArray());
// Output: [1, 1, 3, 4, 5]
```

---

## topk()

```php
public function topk(
    int $k,
    ?int $axis = -1,
    bool $largest = true,
    bool $sorted = true,
    SortKind $kind = SortKind::QuickSort
): array
```

Return top-k values and indices like PyTorch topk.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$k` | `int` | Number of elements to select. |
| `$axis` | `int\|null` | Axis along which to select. If null, flatten first. Optional. Default: `-1`. |
| `$largest` | `bool` | If true, select largest values; otherwise smallest values. Optional. Default: `true`. |
| `$sorted` | `bool` | If true, keep selected values sorted by rank. Optional. Default: `true`. |
| `$kind` | `SortKind` | Sorting algorithm. Optional. Default: `SortKind::QuickSort`. |

### Returns

- `array` - Array with keys 'values' (NDArray) and 'indices' (NDArray).

### Raises

- `InvalidArgumentException` - If k < 0.

### Examples

```php
$arr = NDArray::array([3, 1, 4, 1, 5]);
$result = $arr->topk(3);

print_r($result['values']->toArray());
// Output: [5, 4, 3]

print_r($result['indices']->toArray());
// Output: [4, 2, 0]

// Smallest values
$result = $arr->topk(3, largest: false);
print_r($result['values']->toArray());
// Output: [1, 1, 3]
```

---

## bincount()

```php
public function bincount(?int $minlength = null): NDArray
```

Count occurrences of non-negative integer values in flattened input.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$minlength` | `int\|null` | Minimum output length. Optional. Default: `null`. |

### Returns

- `NDArray` - Int64 counts array.

### Raises

- `InvalidArgumentException` - If minlength < 0.

### Examples

```php
$arr = NDArray::array([0, 1, 1, 2, 2, 2, 3]);
$counts = $arr->bincount();
print_r($counts->toArray());
// Output: [1, 2, 3, 1]
// 0 appears 1 time, 1 appears 2 times, 2 appears 3 times, 3 appears 1 time
```

---

## Summary Table

| Method | Description | Use Case |
|--------|-------------|----------|
| `argmin()` | Index of minimum | Find position of smallest value |
| `argmax()` | Index of maximum | Find position of largest value |
| `sort()` | Sort array | Order elements |
| `argsort()` | Indices to sort | Get sort order without sorting |
| `topk()` | Top k elements | Get largest/smallest k values |
| `bincount()` | Count occurrences | Histogram of integer values |

---

## Next Steps

- [Statistics](/api/statistics)
- [Indexing Routines](/api/indexing-routines)
- [Mathematical Functions](/api/mathematical-functions)