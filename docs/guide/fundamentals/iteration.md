# Iterating Over Arrays

Working with array elements one at a time through iteration.

While NDArray is designed for vectorized operations that process entire arrays at once, sometimes you need to work with individual elements. Whether you're debugging, exporting data, or performing operations that can't be vectorized, iteration provides a way to access array contents sequentially.

## How Iteration Works

NDArray implements PHP's `IteratorAggregate` interface, making arrays work seamlessly with `foreach` loops. The behavior depends on the array's dimensionality:

### 1D Arrays: Element by Element

For one-dimensional arrays, iteration yields scalar values directly:

```php
$temps = NDArray::array([72.5, 68.0, 75.3, 71.2]);

foreach ($temps as $temp) {
    echo "Temperature: {$temp}°F\n";
}
// Temperature: 72.5°F
// Temperature: 68°F
// Temperature: 75.3°F
// Temperature: 71.2°F
```

This works naturally for any 1D array—time series, measurements, lists of values.

### 2D Arrays and Higher: Row by Row

For multi-dimensional arrays, iteration traverses the first axis, yielding views of the remaining dimensions:

```php
$scores = NDArray::array([
    [85, 92, 78],  // Student 1
    [90, 88, 94],  // Student 2
    [76, 84, 81],  // Student 3
]);

foreach ($scores as $studentId => $studentScores) {
    $average = $studentScores->mean();
    echo "Student {$studentId}: average = {$average}\n";
}
```

Each yielded value is an NDArray view sharing memory with the original. Modifying a row affects the parent array:

```php
foreach ($scores as $row) {
    $row[0] = 100;  // Modify first element of each row
}

// Original array is now modified
```

### 3D and Higher Dimensions

The same pattern continues for any dimensionality—iteration always happens along the first axis:

```php
$batch = NDArray::random([32, 224, 224, 3]);  // 32 images

foreach ($batch as $image) {
    // $image has shape [224, 224, 3]
    processImage($image);
}
```

## Flat Iteration

Sometimes you need to visit every element regardless of dimensionality. The `flat()` method provides a 1-D view of the entire array:

```php
$matrix = NDArray::array([
    [1, 2, 3],
    [4, 5, 6],
]);

// Visit every element in C-order (row-major)
foreach ($matrix->flat() as $value) {
    echo $value . " ";
}
// Output: 1 2 3 4 5 6
```

The flat iterator always traverses elements in C-contiguous order—row by row, regardless of the array's actual memory layout.

### Accessing by Index

The flat iterator supports array-style indexing with automatic negative index handling:

```php
$data = NDArray::array([10, 20, 30, 40, 50]);
$flat = $data->flat();

echo $flat[0];   // 10 (first element)
echo $flat[-1];  // 50 (last element)
echo $flat[-2];  // 40 (second to last)
```

This works consistently across all array dimensionalities:

```php
$matrix = NDArray::array([[1, 2], [3, 4]]);
$flat = $matrix->flat();

echo $flat[0];   // 1
echo $flat[3];   // 4
echo $flat[-1];  // 4 (last element)
```

### Converting to PHP Arrays

When you need a standard PHP array, use `toArray()`:

```php
$matrix = NDArray::array([[1, 2], [3, 4]]);
$phpArray = $matrix->flat()->toArray();

// $phpArray is now [1, 2, 3, 4]
```

Or get the element count:

```php
echo count($matrix->flat());  // 4
```

## How Flat Iteration Works

The `flat()` iterator uses a hybrid approach optimized for different array sizes:

**Small Arrays** (< 100,000 elements): All elements are extracted in a single FFI call. This is fast and memory-efficient for typical use cases.

**Large Arrays** (≥ 100,000 elements): Elements are loaded in chunks of 10,000 at a time. This prevents excessive memory usage while maintaining good performance.

Both modes are transparent—you simply call `flat()` and iterate:

```php
// Works efficiently regardless of array size
$huge = NDArray::random([1000, 1000]);  // 1 million elements
foreach ($huge->flat() as $value) {
    // Processes in 100 chunks of 10,000 elements each
}
```

### Flat Iterator vs Flat Array

It's important to understand that `flat()` returns an **iterator**, not an actual array. If you need a one-dimensional NDArray that you can call methods like `sum()` or `mean()` on, use `flatten()` or `ravel()` instead:

```php
$matrix = NDArray::array([[1, 2], [3, 4]]);

// flat() returns an iterator - great for iteration, but limited methods
$flat = $matrix->flat();
foreach ($flat as $val) { /* ... */ }
// $flat->sum();  // Won't work - FlatIterator doesn't have array methods

// flatten() returns an actual 1D NDArray
$flattened = $matrix->flatten();
echo $flattened->sum();   // Works: 10
echo $flattened->mean();  // Works: 2.5

// ravel() also returns a 1D NDArray (view if possible, copy if not)
$raveled = $matrix->ravel();
echo $raveled->sum();  // Works: 10
```

**When to use each:**
- Use `flat()` when you only need to iterate or index individual elements
- Use `flatten()` when you need a true 1D array and want to ensure data independence (always copies)
- Use `ravel()` when you want a 1D array but prefer zero-copy when the array is contiguous

## Important Considerations

### Snapshot Semantics

The flat iterator captures a snapshot of the array at creation time. Changes to the original array afterward won't be reflected:

```php
$arr = NDArray::array([1, 2, 3, 4, 5]);
$flat = $arr->flat();

$arr->set([0], 999);

echo $flat[0];  // Still shows 1, not 999
```

This ensures consistent iteration even if the source array is modified elsewhere in your code.

### Assignment via Flat Iterator

You can assign values through the flat iterator using array syntax:

```php
$arr = NDArray::array([1, 2, 3, 4, 5]);
$flat = $arr->flat();

// Modify elements via flat indexing
$flat[0] = 100;   // Sets first element
$flat[-1] = 999;  // Sets last element

// Original array is modified
echo $arr[0];  // 100
echo $arr[4];  // 999
```

This works efficiently by converting the logical flat index to the underlying storage index, handling views correctly. Negative indices are supported just like with regular indexing.

**Note:** Assignment modifies the original array immediately. The snapshot semantics apply to reading only.

### Modification During Iteration

Avoid modifying an array while iterating over it:

```php
$arr = NDArray::array([1, 2, 3, 4, 5]);

// Don't do this
foreach ($arr as $value) {
    $arr->set([0], rand(1, 100));  // Undefined behavior!
    process($value);
}
```

If you need to modify based on iteration, collect changes first:

```php
$changes = [];
foreach ($arr->flat() as $i => $value) {
    if ($value > 3) {
        $changes[$i] = $value * 2;
    }
}

// Apply after iteration
foreach ($changes as $i => $newValue) {
    $arr->set([$i], $newValue);
}
```

## When to Use Iteration

Iteration is appropriate for:

- **Debugging and inspection**: Examining array contents
- **Small arrays**: Processing fewer than 1,000 elements
- **Data export**: Converting to other formats or systems
- **Non-vectorizable operations**: Custom logic that can't be expressed as array operations

For numerical computations on large arrays, always prefer vectorized operations:

```php
// ❌ Slow: PHP loop overhead
$result = [];
foreach ($arr->flat() as $x) {
    $result[] = sin($x) * 2;
}

// ✅ Fast: Rust-accelerated vectorized operation
$result = $arr->sin()->multiply(2);
```

Vectorized operations are typically 10-100x faster because they process data in optimized Rust code without PHP loop overhead.

## Summary

- Use `foreach ($arr as $value)` for 1D arrays (yields scalars)
- Use `foreach ($arr as $row)` for 2D+ arrays (yields views)
- Use `$arr->flat()` for element-by-element access regardless of dimensionality
- Flat iteration supports indexing: `$arr->flat()[5]` or `$arr->flat()[-1]`
- The iterator is a read-only snapshot for consistency
- Prefer vectorized operations over iteration for large-scale numerical work
