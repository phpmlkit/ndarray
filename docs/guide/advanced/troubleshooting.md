# Troubleshooting

Common errors and their solutions when working with NDArray.

## Installation Issues

### "FFI extension not loaded"

**Error:**
```
Error: FFI extension is not loaded
```

**Solution:**
Enable FFI in your `php.ini`:

```ini
extension=ffi
ffi.enable=true
```

Or enable for specific scripts:
```ini
ffi.enable=preload
```

### "Cannot load Rust library"

**Error:**
```
RuntimeException: Failed to load Rust library
```

**Solution:**
1. Check that the Rust library was compiled: `cargo build --release`
2. Verify the library path in your configuration
3. Check library dependencies with `ldd` (Linux) or `otool -L` (macOS)

## Shape Errors

### "Shape mismatch"

**Error:**
```
ShapeException: Shape mismatch for operation
```

**Cause:** Arrays have incompatible shapes for the operation.

**Solution:**
```php
$a = NDArray::zeros([3, 4]);
$b = NDArray::zeros([5, 4]);

// Check shapes before operations
echo implode(',', $a->shape());  // 3,4
echo implode(',', $b->shape());  // 5,4

// Make compatible
if ($a->shape() !== $b->shape()) {
    // Reshape, broadcast, or fix the data
}
```

### "Cannot broadcast"

**Error:**
```
ShapeException: Cannot broadcast shapes [3,4] and [5]
```

**Cause:** Arrays cannot be broadcast to a common shape.

**Solution:**
```php
$a = NDArray::zeros([3, 4]);
$b = NDArray::zeros([4]);  // Works: [3,4] + [4] → [3,4]

// Check broadcasting rules
$c = NDArray::zeros([5]);  // Won't work with [3,4]
```

## Index Errors

### "Index out of bounds"

**Error:**
```
IndexException: Index 10 is out of bounds for dimension 0 with size 5
```

**Cause:** Index exceeds array dimensions.

**Solution:**
```php
$arr = NDArray::zeros([5]);

// Check bounds before accessing
$index = 10;
if ($index >= 0 && $index < $arr->shape()[0]) {
    $value = $arr[$index];
}

// Or use negative indexing
$last = $arr[-1];  // Valid: gets last element
```

### "Too many indices"

**Error:**
```
IndexException: Too many indices: got 3 for array with 2 dimensions
```

**Cause:** More indices provided than array dimensions.

**Solution:**
```php
$matrix = NDArray::zeros([3, 4]);

// 2D array needs exactly 2 indices for scalar access
$value = $matrix->get(1, 2);  // Correct

// For partial indexing, use fewer indices
$row = $matrix->get(1);  // Returns 1D view
```

### "Invalid slice selector"

**Error:**
```
IndexException: Invalid slice selector: 'abc'
```

**Cause:** Slice syntax is invalid.

**Solution:**
```php
$arr = NDArray::arange(10);

// Valid slice syntax
$slice = $arr->slice(['2:5']);     // Range
$slice = $arr->slice(['::2']);     // With step
// Note: Negative step (e.g., '::-1') is not supported
// Use $arr->flip() to reverse arrays

// Invalid: $arr->slice(['abc']);
```

## Type Errors

### "Invalid dtype"

**Error:**
```
InvalidArgumentException: Invalid dtype for operation
```

**Cause:** Operation not supported for the array's data type.

**Solution:**
```php
$arr = NDArray::array([1, 2, 3], DType::Int32);

// Check dtype
echo $arr->dtype()->name;  // "Int32"

// Convert if needed
$float_arr = $arr->astype(DType::Float64);
```

### "Cannot assign array to scalar index"

**Error:**
```
IndexException: Cannot assign array to scalar index
```

**Cause:** Trying to assign an array to a single element position.

**Solution:**
```php
$arr = NDArray::zeros([3, 3]);

// Correct: Assign scalar to scalar index
$arr->set([0, 0], 5);

// Correct: Assign array to slice
$row = NDArray::array([1, 2, 3]);
$arr[0] = $row;  // Assign to row
```

## Memory Issues

### "Out of memory"

**Error:**
```
RuntimeException: Failed to allocate memory
```

**Cause:** Trying to allocate an array larger than available memory.

**Solution:**
```php
// Check memory first
$shape = [100000, 100000];  // 74.5 GB for Float64!
$bytes = array_product($shape) * 8;
$gb = $bytes / (1024 * 1024 * 1024);

if ($gb > 8) {  // More than 8 GB
    // Process in batches
    processInBatches($shape);
}
```

### High Memory Usage

**Symptom:** Script uses more memory than expected.

**Cause:** Creating unnecessary copies instead of views.

**Solution:**
```php
$data = NDArray::random([10000, 10000]);

// Bad: Creates copy
$subset = $data->slice(['0:1000'])->copy();  // 80 MB copied

// Good: Uses view
$subset = $data->slice(['0:1000']);  // 0 bytes copied

// Good: Explicit cleanup when done
unset($large_array);
```

## Performance Issues

### Operations are slow

**Symptom:** Simple operations take seconds.

**Common Causes:**

1. **PHP Loops**
```php
// Bad: 1000 FFI calls
for ($i = 0; $i < 1000; $i++) {
    $sum += $arr[$i];
}

// Good: 1 FFI call
$sum = $arr->sum();
```

2. **Growing Arrays**
```php
// Bad: O(n²) array creation
$result = NDArray::array([]);
for ($i = 0; $i < 1000; $i++) {
    $result = NDArray::concatenate([$result, NDArray::array([$i])]);
}

// Good: Pre-allocate
$result = NDArray::zeros([1000]);
for ($i = 0; $i < 1000; $i++) {
    $result[$i] = $i;
}
```

### Undefined values from empty()

**Symptom:** `empty()` array contains unexpected values.

**Cause:** `NDArray::empty()` creates uninitialized memory.

**Solution:**
```php
// empty() values are undefined - must fill before reading
$arr = NDArray::empty([10]);
echo $arr[0];  // Undefined value!

// Use zeros() for initialized memory
$arr = NDArray::zeros([10]);
echo $arr[0];  // 0
```

## Slicing Issues

### "Only one ellipsis allowed"

**Error:**
```
IndexException: Only one ellipsis (...) allowed per slice
```

**Cause:** Used multiple ellipses in one slice.

**Solution:**
```php
$arr = NDArray::zeros([2, 3, 4, 5]);

// Valid: One ellipsis
$slice = $arr->slice([0, '...']);

// Invalid: Multiple ellipses
// $slice = $arr->slice(['...', 0, '...']);
```

### Slice modifies original

**Symptom:** Changing a slice changes the original array.

**Explanation:** Slices are views - they share memory with the original.

**Solution:**
```php
$original = NDArray::zeros([5, 5]);
$slice = $original->slice(['0:2', '0:2']);

// This modifies both slice AND original
$slice->assign(1);

// Make a copy if you need independence
$independent = $original->slice(['0:2', '0:2'])->copy();
$independent->assign(2);  // Original unchanged
```

## Debugging Tips

### Enable Error Reporting

```php
error_reporting(E_ALL);
ini_set('display_errors', '1');
```

### Check Array Properties

```php
$arr = NDArray::random([10, 20]);

echo "Shape: " . implode(',', $arr->shape()) . "\n";
echo "NDim: " . $arr->ndim() . "\n";
echo "Size: " . $arr->size() . "\n";
echo "DType: " . $arr->dtype()->name . "\n";
echo "Is View: " . ($arr->isView() ? 'Yes' : 'No') . "\n";
echo "Contiguous: " . ($arr->isContiguous() ? 'Yes' : 'No') . "\n";
```

### Validate Before Operations

```php
function safeAdd(NDArray $a, NDArray $b): NDArray {
    if ($a->shape() !== $b->shape()) {
        throw new InvalidArgumentException(
            "Shape mismatch: [" . implode(',', $a->shape()) . 
            "] vs [" . implode(',', $b->shape()) . "]"
        );
    }
    return $a->add($b);
}
```

### Profile Memory Usage

```php
$startMemory = memory_get_usage(true);
$arr = NDArray::random([1000, 1000]);
$endMemory = memory_get_usage(true);

$usedMB = ($endMemory - $startMemory) / (1024 * 1024);
echo "Memory used: {$usedMB} MB\n";
```

## Getting Help

If you encounter an issue not covered here:

1. Check the [API Reference](/api/) for correct method signatures
2. Review the [Examples](/examples/) for usage patterns
3. Check your data types and shapes carefully
4. Enable all error reporting to see full stack traces

## See Also

- **[FFI Internals](/guide/advanced/ffi-internals)** - Understanding the PHP-Rust bridge
- **[Performance Best Practices](/guide/advanced/performance)** - Writing efficient code
- **[Views vs Copies](/guide/fundamentals/views-vs-copies)** - Memory management
