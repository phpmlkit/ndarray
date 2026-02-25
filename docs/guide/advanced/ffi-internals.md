# FFI Internals

Understanding how NDArray uses PHP's Foreign Function Interface (FFI) to communicate with Rust will help you write more efficient code.

## Architecture Overview

```
┌─────────────────────────────────────────┐
│           PHP Code                      │
│  $array->add(5)->multiply(2)           │
└──────────────┬──────────────────────────┘
               │ FFI Call
               ▼
┌─────────────────────────────────────────┐
│        Rust ndarray Library             │
│  - Memory allocation                    │
│  - Computation (SIMD, parallel)         │
│  - Data storage                         │
└─────────────────────────────────────────┘
```

## What is FFI?

PHP's FFI (Foreign Function Interface) allows PHP code to call functions written in C (or languages that can produce C-compatible libraries, like Rust). NDArray uses FFI to:

1. **Allocate memory** in Rust for array data
2. **Perform computations** using optimized Rust code
3. **Transfer data** between PHP and Rust

## Memory Management

### Rust Owns the Data

All NDArray data lives in Rust-allocated memory:

```php
$arr = NDArray::zeros([1000, 1000]);
// Memory allocated in Rust, PHP holds only a pointer
```

### PHP Holds Opaque Pointers

PHP NDArray objects contain:
- An opaque pointer to Rust memory (`$handle`)
- View metadata (shape, strides, offset) managed in PHP
- Reference to parent array (for views)

### Automatic Cleanup

When an NDArray is destroyed:

```php
{
    $arr = NDArray::random([1000, 1000]);
    // ... use $arr ...
} // Destructor called → Rust memory freed (only for root arrays)
```

**Root arrays** (where `$base` is null) free Rust memory on destruction.
**Views** keep their root alive through PHP's reference counting.

### View Memory Model

```php
$root = NDArray::random([1000, 1000]);  // Root array
$view = $root->slice(['0:100']);         // View shares handle

// $view->handle === $root->handle (same Rust memory)
// $view has different shape/strides/offset
// $view->base points to $root
```

Views do not free memory - only the root array does.

## FFI Call Patterns

### Each Method = One (or more) FFI Calls

```php
// Each operation makes FFI call(s) to Rust
$a = $data->add(5);        // 1 FFI call
$b = $a->multiply(2);      // 1 FFI call  
$c = $b->sqrt();           // 1 FFI call

// Chaining is the same as separate calls
$result = $data->add(5)->multiply(2)->sqrt();
// 3 FFI calls total
```

### FFI Call Overhead

- **Small arrays (< 100 elements)**: FFI overhead is significant
- **Large arrays (> 1000 elements)**: FFI overhead is negligible
- **Always prefer vectorized ops over PHP loops**, even with FFI overhead

## Data Type Handling

### Type Conversion

When data moves between PHP and Rust:

```php
// PHP → Rust: Conversion happens during array creation
$arr = NDArray::array([1, 2, 3], DType::Float32);
// PHP integers converted to Rust f32

// Rust → PHP: Conversion during scalar access
$value = $arr[0];
// Rust f32 converted to PHP float
```

### Supported Types

NDArray supports all FFI-compatible types:

| DType | Rust Type | PHP Type | Size |
|-------|-----------|----------|------|
| Int8 | i8 | integer | 1 byte |
| Int16 | i16 | integer | 2 bytes |
| Int32 | i32 | integer | 4 bytes |
| Int64 | i64 | integer | 8 bytes |
| UInt8 | u8 | integer | 1 byte |
| UInt16 | u16 | integer | 2 bytes |
| UInt32 | u32 | integer | 4 bytes |
| UInt64 | u64 | integer | 8 bytes |
| Float32 | f32 | float | 4 bytes |
| Float64 | f64 | float | 8 bytes |
| Bool | u8 | boolean | 1 byte |

## Error Handling

### Rust Panics → PHP Exceptions

Rust code never panics to PHP. All errors are converted to exceptions:

```php
try {
    $result = $a->divide($b);  // Division by zero in Rust
} catch (NDArrayException $e) {
    // Caught as PHP exception
    echo $e->getMessage();
}
```

### Error Types

- **`NDArrayException`**: General errors (invalid operations, allocation failures)
- **`ShapeException`**: Shape mismatch errors
- **`IndexException`**: Invalid indexing

## Zero-Copy Operations

Some operations require no FFI calls at all:

### View Creation (Pure PHP)

```php
$view = $arr->slice(['0:10']);  // No FFI call!
// Shape/strides/offset calculated in PHP
// Same Rust handle shared
```

### Partial Indexing (Pure PHP)

```php
$row = $matrix[0];  // No FFI call!
// Returns view with updated metadata
```

## Best Practices

### 1. Minimize PHP Loops

```php
// Bad: Thousands of FFI calls
for ($i = 0; $i < 1000; $i++) {
    $sum += $arr[$i];  // FFI call per access
}

// Good: One FFI call
$sum = $arr->sum();  // Single FFI call, Rust handles iteration
```

### 2. Use Vectorized Operations

```php
// Bad: Multiple small operations
for ($i = 0; $i < $n; $i++) {
    $result[$i] = sin($arr[$i]) * 2;
}

// Good: Single vectorized operation
$result = $arr->sin()->multiply(2);
```

### 3. Batch Element Access

```php
// Bad: Individual element access
$values = [];
for ($i = 0; $i < 100; $i++) {
    $values[] = $arr[$i];
}

// Good: Convert to PHP array once
$values = $arr->slice(['0:100'])->toArray();
```

## Troubleshooting FFI Issues

### Memory Leaks

If memory grows unexpectedly:

```php
// Check for circular references
$arr = NDArray::random([1000, 1000]);
$view = $arr->slice(['0:10']);
// Both reference each other - but PHP GC handles this

// Explicit cleanup if needed
unset($view);
unset($arr);
```

### Performance Debugging

```php
// Profile FFI calls
$start = microtime(true);
$result = $arr->sum();
$ffiTime = microtime(true) - $start;
echo "FFI call took: {$ffiTime}s\n";
```

### Type Issues

```php
// Check dtype if operations fail
$arr = NDArray::array([1, 2, 3], DType::Int32);
echo $arr->dtype()->value;  // "int32"

// Some ops require specific types
$arr->sqrt();  // Works on all types, returns appropriate type
```

## See Also

- **[Performance Best Practices](/guide/advanced/performance)** - Writing efficient NDArray code
- **[Troubleshooting](/guide/advanced/troubleshooting)** - Common errors and solutions
- **[Views vs Copies](/guide/fundamentals/views-vs-copies)** - Understanding memory sharing
