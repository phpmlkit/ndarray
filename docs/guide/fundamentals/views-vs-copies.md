# Views vs Copies

Understanding when arrays share memory versus when they create copies is essential for writing efficient and correct code.

## The Core Concept

A **view** is a new array object that looks at the same memory as the original array, thus modifying one affects the other. A **copy** is an entirely new array with its own independent memory therefore changing one does not affect the other.

**Key differences:**

|              | View                              | Copy                     |
|--------------|-----------------------------------|--------------------------|
| Memory       | Shared with original              | Own, independent         |
| Address      | Same as original                  | New, different           |
| Writable     | Yes (if original is writable)     | Yes                      |
| Fast to Make | Always (O(1) for basic cases)     | Slower (requires copy)   |
| Use case     | Efficient, no data duplication    | Data isolation needed    |

**Visual Example:**

|                |              |                |
|:---------------|:-------------|:-----------------|
| ![Original](https://dummyimage.com/120x80/eeeeee/888888&text=Original+Array) | ![View](https://dummyimage.com/120x80/eeeeee/888888&text=View) | ![Copy](https://dummyimage.com/120x80/eeeeee/888888&text=Copy) |
| **Original Array**  | **View (shares memory)** | **Copy (independent)**   |
| Address: 0x1000     | Address: 0x1000 *(same!)* | Address: 0x2000 *(different)* |
| Data: [1 2 3 4 5 6] | Data: [1 2 3 4 5 6]      | Data: [1 2 3 4 5 6]      |
| Shape: [2,3]        | Shape: [1,3] (example)   | Shape: [2,3]             |
| Owns: Yes           | Owns: No                 | Owns: Yes                |
| Offset: 0           | Offset: 0 or >0 (view)   | Offset: 0                |
| View of: —          | View of: Original        | View of: —               |

<details>
<summary>Equivalent memory diagrams</summary>

```
Original Array:
[Memory Address: 0x1000] [Data: 1, 2, 3, 4, 5, 6] [Shape: 2x3]

       ┌──────────────────────────┐
       │ 1 │ 2 │ 3 │ 4 │ 5 │ 6   │
       └──────────────────────────┘
        ^0x1000 (Owned)

View (shares memory):
[Memory Address: 0x1000] [Shape: 1x3] [Offset: 0]
      (reads/writes from same 0x1000)

Copy (independent):
[Memory Address: 0x2000] [Data: 1, 2, 3, 4, 5, 6] [Shape: 2x3]
      (completely separate memory from original)
```
</details>


```php
$arr = NDArray::array([[1, 2, 3], [4, 5, 6]]);

// View: shares memory with $arr
$view = $arr[0];  // First row

// Copy: independent memory
$copy = $arr->copy();  // Deep copy
```

## When Are Views Created?

Views are created by operations that **access** or **reference** data without modifying it:

| Operation        | Example                  | Memory Impact         |
|------------------|-------------------------|-----------------------|
| Slicing          | `$arr[0:5]`             | No copy (O(1))        |
| Indexing         | `$arr[0]`               | No copy (O(1))        |
| Transpose        | `$arr->transpose()`      | No copy if contiguous |
| Reshape          | `$arr->reshape([3,4])`  | No copy if contiguous |
| expandDims       | `$arr->expandDims(0)`   | No copy (O(1))        |
| squeeze          | `$arr->squeeze()`       | No copy (O(1))        |
| ravel            | `$arr->ravel()`         | No copy if contiguous |

### Slicing Returns Views

```php
$data = NDArray::zeros([1000, 1000]);

$slice = $data->slice(['0:100', '0:100']);  // VIEW: 0 bytes copied!
$rows = $data->slice(['5:10']);             // VIEW
$cols = $data->slice([':', '2']);           // VIEW
```

### Indexing Returns Views (for partial indices)


```php
$matrix = NDArray::array([[1, 2], [3, 4]]);

$first_row = $matrix[0];      // VIEW of first row
$second_row = $matrix->get(1);  // VIEW of second row
```

## When Are Copies Created?

Copies are created by operations that **transform** or **compute** new data:

| Operation       | Example              | Memory Impact     |
|-----------------|---------------------|-------------------|
| Math ops        | `$arr->multiply(2)` | New allocation    |
| Type cast       | `$arr->astype()`     | New allocation    |
| Explicit copy   | `$arr->copy()`       | New allocation    |
| Flatten         | `$arr->flatten()`    | New allocation    |

### Mathematical Operations

```php
$arr = NDArray::array([1, 2, 3]);

$result = $arr->multiply(2);     // COPY
$result = $arr->abs();            // COPY
$result = $arr->sqrt();           // COPY
$result = $arr->add($other);      // COPY
$result = $arr->matmul($b);       // COPY
```

### Type Conversions

```php
$arr = NDArray::array([1, 2, 3]);

$converted = $arr->astype(DType::Int32);  // COPY
```

### Explicit Copy

```php
$copy = $arr->copy();  // Deep copy (always)
```

### Non-contiguous to Contiguous

```php
$transposed = $matrix->transpose();     // VIEW (non-contiguous)
$contiguous = $transposed->copy();      // COPY (contiguous)
```

## The Critical Rule

::: danger Golden Rule
**Modifying a view modifies the parent array.**

**Operations on views create copies and don't affect the parent.**
:::

### Modifying Views

```php
$arr = NDArray::array([[1, 2, 3], [4, 5, 6]]);

// Get a view
$first_row = $arr[0];  // VIEW

// Modify the view
$first_row->set([0], 999);

// Original is changed!
print_r($arr->toArray());
// [[999 2 3]
//  [4 5 6]]
```

### Operations on Views

```php
$arr = NDArray::array([[1, 2, 3], [4, 5, 6]]);

// Get a view
$first_row = $arr[0];  // VIEW

// Perform operation (creates copy!)
$doubled = $first_row->multiply(2);  // COPY

// Original is UNCHANGED
print_r($arr->get(0, 0));  // 1 (not 2!)
print_r($doubled->get(0));  // 2
```

### Assignment vs Operation

```php
$arr = NDArray::array([1, 2, 3, 4, 5]);
$view = $arr->slice(['0:3']);  // VIEW

// Assignment: modifies parent
$view->set([0], 999);
print_r($arr->get(0));  // 999 ✓

// Operation: creates copy
$view = $view->multiply(2);  // Now $view is a COPY!
$view->set([0], 111);  // Doesn't affect $arr
```

## Visual Examples

### Example 1: Modifying a View

```php
$data = NDArray::array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]);

// Create views
$first_row = $data[0];        // [1 2 3]
$center = $data->slice(['1:2', '1:2']);

// Modify views
$first_row->set([1], 999);
$center->set([0, 0], 888);

// Parent is modified
print_r($data->toArray());
// [[  1 999   3]
//  [  4 888   6]
//  [  7   8   9]]
```

**Memory Layout:**
```
Original:     View (first_row):     View (center):
[1 2 3]       → [1 2 3]            →     [5 6]
[4 5 6]                              →     [8 9]
[7 8 9]

All share the same underlying data!
```

### Example 2: Operations Create Copies

```php
$data = NDArray::array([
    [1, 2, 3],
    [4, 5, 6]
]);

// View
$first_row = $data[0];  // VIEW

// Operation creates copy
$processed = $first_row->abs()->sqrt()->multiply(2);

// $data is unchanged!
print_r($data->get(0, 0));  // 1

// $processed is independent
print_r($processed->toArray());  // [2 2.828 3.464]
```

## Checking if Array is a View

```php
$arr = NDArray::array([1, 2, 3, 4, 5]);

print_r($arr->isView());  // false (root array)

$view = $arr->slice(['0:3']);
print_r($view->isView());  // true
```

## Checking Contiguity

Contiguous arrays have elements stored sequentially in memory:

```php
$arr = NDArray::array([[1, 2], [3, 4]]);

// C-contiguous (row-major)
print_r($arr->isContiguous());  // true

// Transpose changes strides, not contiguous
$transposed = $arr->transpose();
print_r($transposed->isContiguous());  // false

// Copy makes it contiguous
$copy = $transposed->copy();
print_r($copy->isContiguous());  // true
```

## Why Views Matter

### Performance Benefits

**Without views (copying):**
```php
$data = NDArray::random([10000, 10000]);  // 800 MB
$batch1 = $data->slice(['0:1000'])->copy();    // +800 MB
$batch2 = $data->slice(['1000:2000'])->copy(); // +800 MB
// Total: 2.4 GB!
```

**With views:**
```php
$data = NDArray::random([10000, 10000]);  // 800 MB
$batch1 = $data->slice(['0:1000']);        // VIEW: 0 MB
$batch2 = $data->slice(['1000:2000']);     // VIEW: 0 MB
// Total: 800 MB!
```

### Speed Benefits

```php
$large = NDArray::random([10000, 10000]);

// Slow: Copying 800 MB
$start = microtime(true);
$copy = $large->copy();
echo "Copy time: " . (microtime(true) - $start) . "s\n";
// ~0.1-0.2 seconds

// Fast: Creating view (microseconds)
$start = microtime(true);
$view = $large->slice(['0:5000']);
echo "View time: " . (microtime(true) - $start) . "s\n";
// ~0.000001 seconds (1000x+ faster)
```

## Common Patterns

### Pattern 1: Batch Processing

```php
$data = NDArray::random([10000, 784]);  // Large dataset

// Process in batches using views
$batch_size = 100;
for ($i = 0; $i < 10000; $i += $batch_size) {
    // Zero-copy view of batch
    $batch = $data->slice(["{$i}:" . ($i + $batch_size)]);
    
    // Process batch (operations create copies internally)
    $normalized = $batch->subtract($batch->mean())->divide($batch->std());
    
    // Train model...
}
// Total memory: ~800 MB (just the original data)
```

### Pattern 2: Modifying Subsets

```php
$image = NDArray::zeros([1024, 1024, 3]);

// Get view of region
$region = $image->slice(['100:200', '100:200']);

// Modify region (modifies original image!)
$region->assign(255);  // Set to white

// Or assign values
$image->slice(['100:200', '100:200', '0'])->assign(255);  // Red channel
```

### Pattern 3: Strided Access

```php
$data = NDArray::arange(0, 100);

// Every 2nd element
$evens = $data->slice(['::2']);   // [0 2 4 ... 98]

// Every 3rd element starting at 1
$pattern = $data->slice(['1::3']); // [1 4 7 ...]

// Note: To reverse, use flip() method (returns a copy, not a view)
// $reversed = $data->flip();
```

### Pattern 4: Column/Row Operations

```php
$matrix = NDArray::random([100, 100]);

// Get column (view)
$col = $matrix->slice([':', '0']);

// Zero out column
$col->assign(0);

// Verify
assert($matrix->get(0, 0) === 0.0);
assert($matrix->get(99, 0) === 0.0);
```

## Gotchas and Solutions

### Gotcha 1: Chained Operations

```php
$arr = NDArray::array([1, 2, 3, 4, 5]);
$view = $arr->slice(['0:3']);

// ❌ Wrong: Reassigning breaks view connection
$view = $view->multiply(2);  // $view is now a COPY!
$view->set([0], 999);       // Doesn't affect $arr!

// ✅ Correct: Work with the result knowing it's separate
```

**Solution:** Be aware that reassignment creates copies.

### Gotcha 2: Unexpected Modifications

```php
$data = NDArray::array([[1, 2], [3, 4]]);

// Get view
$first_row = $data[0];

// Pass to function
function process($row) {
    $row->set([0], 999);  // Modifies original!
}

process($first_row);
print_r($data->get(0, 0));  // 999 (unexpected!)
```

**Solution:** Pass copies when modification isn't intended:
```php
process($first_row->copy());
```

### Gotcha 3: Transpose + Modification

```php
$matrix = NDArray::array([[1, 2], [3, 4]]);
$transposed = $matrix->transpose();  // VIEW

// Modifying transpose affects original!
$transposed->set([0, 1], 999);
print_r($matrix->get(1, 0));  // 999
```

**Solution:** Copy if you need independent data:
```php
$transposed = $matrix->transpose()->copy();
```

### Gotcha 4: Non-contiguous Operations

```php
$arr = NDArray::array([[1, 2, 3], [4, 5, 6]]);
$transposed = $arr->transpose();  // VIEW but non-contiguous

// Some operations may need contiguous data internally
$result = $transposed->reshape([6]);  // May create copy
```

**Solution:** Force copy if needed:
```php
$contiguous = $transposed->copy();
```

## Best Practices

### 1. Use Views for Reading

```php
// Fast: Reading with views
$data = NDArray::random([1000, 1000]);
$mean = $data->slice(['0:100'])->mean();  // Fast, no copy
```

### 2. Copy When Modifying Independently

```php
// If you need to modify without affecting original
$working_copy = $original->slice(['0:10'])->copy();
$working_copy->set([0], 999);  // Safe!
```

### 3. Be Explicit About Intent

```php
// Clear intent: we're modifying a view
$region = $image->slice(['100:200', '100:200']);
$region->assign(0);  // Modifies image

// Clear intent: independent copy
$snapshot = $image->slice(['100:200', '100:200'])->copy();
$snapshot->assign(0);  // Doesn't modify image
```

### 4. Check When Unsure

```php
function safeModify(NDArray $arr): NDArray {
    // If it's a view and we want to modify without side effects
    if ($arr->isView()) {
        return $arr->copy();
    }
    return $arr;
}
```

## Quick Reference

| Operation | Result | Modifies Parent? |
|-----------|--------|------------------|
| `$arr->slice(['0:5'])` | View | Yes (if modified) |
| `$arr->copy()` | Copy | No |
| `$arr->multiply(2)` | Copy | No |
| `$arr->abs()` | Copy | No |
| `$arr->transpose()` | View | Yes (if modified) |
| `$arr->reshape([...])` | View* | Yes (if modified) |
| `$arr->astype(...)` | Copy | No |
| `$arr->add($b)` | Copy | No |
| `$arr->set([0,0], 5)` | N/A | Yes |
| `$view = $view->multiply(2)` | Copy | No (reassignment) |

\* View if contiguous, otherwise copy

## Summary

- **Views** are references to original data - fast, memory-efficient
- **Copies** are independent data - safe, isolated
- **Slicing** returns views
- **Operations** return copies
- **Modifying views** modifies the parent
- **Use views** for reading, **copy when needed** for independent modification

Mastering views vs copies is the key to writing efficient NDArray code!

## Next Steps

- **[Indexing](/guide/fundamentals/indexing)** - Detailed indexing patterns
- **[Slicing](/guide/fundamentals/slicing)** - Working with ranges
- **[Performance](/guide/advanced/performance)** - Optimization techniques
- **[Examples](/examples/basics/working-with-views)** - Practical view patterns
