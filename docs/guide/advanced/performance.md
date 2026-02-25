# Performance Best Practices

NDArray is designed for high performance, but following these best practices will help you get the most out of it.

## The Zero-Copy Advantage

The biggest performance benefit comes from zero-copy views:

```php
$data = NDArray::random([10000, 10000]);  // 800 MB array

// View: 0 bytes copied, microseconds to create
$batch = $data->slice(['0:1000']);  

// Copy: 80 MB copied, milliseconds to create
$copy = $data->slice(['0:1000'])->copy();
```

### When to Use Views

**Use views for:**
- Reading subsets of data
- Processing batches
- Extracting rows/columns
- Temporary operations
- Chain operations on subsets

**Avoid views when:**
- You need to modify data independently
- Passing to functions that might modify
- Long-term storage of subsets

## Understanding FFI Overhead

Each NDArray method call that performs computation crosses the PHP→Rust FFI boundary. This has overhead, but:

- **Vectorized operations are always faster than PHP loops**, even with FFI overhead
- Operations on large arrays amortize the FFI call cost
- Chaining methods creates multiple FFI calls (one per operation)

```php
// Each method call is a separate FFI operation
$result = $data->add(10)->multiply(2)->sqrt();
// 3 FFI calls total, but still much faster than PHP loops
```

## Avoid PHP Loops

Never iterate over NDArray elements in PHP for computation:

### Slow: PHP Loop

```php
$result = NDArray::zeros([1000]);
$data = NDArray::random([1000]);
for ($i = 0; $i < 1000; $i++) {
    $val = $data[$i] * 2 + 10;  // Each access is an FFI call!
    $result[$i] = $val;         // Plus another FFI call
}
// Time: ~100ms (1000+ FFI calls)
```

### Fast: Vectorized

```php
$data = NDArray::random([1000]);
$result = $data->multiply(2)->add(10);
// Time: ~0.1ms (2 FFI calls)
```

## Choose Appropriate Data Types

### Memory Usage

| Type | Size | 1000x1000 Array |
|------|------|-----------------|
| Float64 | 8 bytes | 8 MB |
| Float32 | 4 bytes | 4 MB |
| Int32 | 4 bytes | 4 MB |
| Int8 | 1 byte | 1 MB |

```php
// For ML: Float32 is usually sufficient
$weights = NDArray::random([1000, 1000], DType::Float32);
// Memory: 4 MB instead of 8 MB
```

### Speed

- **Float32**: Faster on modern CPUs, sufficient precision for ML
- **Float64**: Higher precision, needed for scientific computing
- **Integers**: Faster for counting and indexing

## Array Creation Strategies

### Pre-allocate When Building Incrementally

When you must build arrays element-by-element (not recommended), use `zeros()` or pre-created arrays:

### Slow: Growing Array

```php
$result = NDArray::array([]);
for ($i = 0; $i < 1000; $i++) {
    // This creates a new array each iteration - very slow!
    $result = NDArray::concatenate([$result, NDArray::array([$i])]);
}
```

### Fast: Pre-allocated with Zeros

```php
$result = NDArray::zeros([1000]);
for ($i = 0; $i < 1000; $i++) {
    $result->set([$i], $i);  // Or use: $result[$i] = $i
}
```

### Even Faster: Vectorized

```php
$result = NDArray::arange(0, 1000);
// Single FFI call, fastest option
```

## Batch Processing

Process large datasets in batches to balance memory and speed:

```php
$data = NDArray::random([100000, 784]);  // Large dataset
$batch_size = 1000;

for ($i = 0; $i < 100000; $i += $batch_size) {
    $end = min($i + $batch_size, 100000);
    $batch = $data->slice(["{$i}:{$end}"]);  // View - no copy!
    
    // Process batch
    $normalized = $batch->subtract($batch->mean())->divide($batch->std());
    
    // Train, predict, etc.
    process($normalized);
}
```

## Memory Management

### Automatic Cleanup

NDArray uses automatic memory management. Arrays are freed when no longer referenced:

```php
function process() {
    $data = NDArray::random([10000, 10000]);  // 800 MB
    // ... process ...
}  // $data freed automatically when function returns
```

### Explicit Cleanup (Rarely Needed)

```php
$large = NDArray::random([100000, 1000]);
// ... use $large ...
unset($large);  // Free memory immediately
```

## Profile Your Code

Use timing to identify bottlenecks:

```php
$start = microtime(true);
$result = expensive_operation();
$elapsed = microtime(true) - $start;
echo "Operation took: {$elapsed}s\n";
```

## Performance Summary

| Pattern | Recommendation |
|---------|----------------|
| Use views | Always prefer views over copies for reading |
| Avoid PHP loops | Use vectorized operations exclusively |
| Choose types | Float32 for ML, Float64 for precision |
| Pre-allocate | Use `zeros()` when building incrementally |
| Batch processing | Process large data in chunks using views |
| Memory | Let PHP's GC handle cleanup, use `unset()` only when needed |

## Benchmarks

Typical performance on modern hardware:

| Operation | 1000x1000 | vs PHP Arrays |
|-----------|-----------|---------------|
| Element-wise add | 0.1 ms | 100x faster |
| Matrix multiply | 5 ms | 500x faster |
| Sum | 0.05 ms | 200x faster |
| Slicing | 0.001 ms | Infinite (PHP copies) |

## Next Steps

- **[Views vs Copies](/guide/fundamentals/views-vs-copies)** - Master zero-copy operations
- **[FFI Internals](/guide/advanced/ffi-internals)** - Understanding the PHP-Rust bridge
