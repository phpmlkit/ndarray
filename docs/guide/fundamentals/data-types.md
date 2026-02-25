# Data Types

NDArray supports multiple data types to balance precision, memory usage, and performance. Understanding when to use each type is crucial for effective numerical computing.

## Supported Data Types

NDArray provides these data types through the `DType` enum:

### Integer Types

| Type | Range | Size | Use Case |
|------|-------|------|----------|
| `Int8` | -128 to 127 | 1 byte | Memory-constrained integer data |
| `Int16` | -32,768 to 32,767 | 2 bytes | Small integers, audio data |
| `Int32` | -2B to 2B | 4 bytes | General integer operations |
| `Int64` | -9×10¹⁸ to 9×10¹⁸ | 8 bytes | Large integers, timestamps |

### Unsigned Integer Types

| Type | Range | Size | Use Case |
|------|-------|------|----------|
| `UInt8` | 0 to 255 | 1 byte | Images (pixel values), binary data |
| `UInt16` | 0 to 65,535 | 2 bytes | Images, audio |
| `UInt32` | 0 to 4B | 4 bytes | Large positive integers |
| `UInt64` | 0 to 1.8×10¹⁹ | 8 bytes | Large positive integers |

### Floating Point Types

| Type | Precision | Size | Use Case |
|------|-----------|------|----------|
| `Float32` | ~7 digits | 4 bytes | ML models, GPU computing |
| `Float64` | ~15 digits | 8 bytes | Scientific computing (default) |

### Boolean

| Type | Values | Size | Use Case |
|------|--------|------|----------|
| `Bool` | true/false | 1 byte | Masks, conditions, flags |

## Using Data Types

### Automatic Type Inference

When creating arrays from PHP values, NDArray automatically selects the appropriate type:

```php
// Integers default to Int64
$ints = NDArray::array([1, 2, 3]);
echo $ints->dtype();  // DType::Int64

// Floats default to Float64
$floats = NDArray::array([1.5, 2.5, 3.5]);
echo $floats->dtype();  // DType::Float64

// Booleans become Bool
$bools = NDArray::array([true, false, true]);
echo $bools->dtype();  // DType::Bool
```

::: tip Note
Type inference follows NumPy conventions: integers become `Int64`, floats become `Float64`, and booleans become `Bool`.
:::

### Explicit Type Specification

```php
use PhpMlKit\NDArray\DType;

// Specify type when creating
$int_arr = NDArray::array([1, 2, 3], DType::Int32);
$float_arr = NDArray::array([1, 2, 3], DType::Float32);

// Specify type for factory methods
$zeros = NDArray::zeros([100, 100], DType::Float32);
$ones = NDArray::ones([1000], DType::Int64);
$random = NDArray::random([10, 10], DType::Float32);
```

### Type Conversion

Convert arrays between types:

```php
$float_arr = NDArray::array([1.5, 2.7, 3.2]);

// Convert to integers (truncation)
$ints = $float_arr->astype(DType::Int32);
print_r($ints->toArray());  // [1, 2, 3]

// Convert to float32 (less precision)
$f32 = $float_arr->astype(DType::Float32);

// Convert to bool (non-zero = true)
$bool_arr = $float_arr->astype(DType::Bool);
print_r($bool_arr->toArray());  // [true, true, true]
```

## Type Properties

### Getting Type Information

```php
$arr = NDArray::array([1, 2, 3]);

// Get DType enum
$dtype = $arr->dtype();

// Get item size in bytes
echo $arr->itemsize();  // 8 (Float64)
```

## Type Promotion Rules

When combining arrays of different types, NDArray promotes to a common type:

### Promotion Hierarchy

```
Bool → UInt8 → UInt16 → UInt32 → UInt64
  ↓
Int8 → Int16 → Int32 → Int64
  ↓
Float32 → Float64
```

### Examples

```php
$a = NDArray::array([1, 2], DType::Int32);
$b = NDArray::array([1.5, 2.5], DType::Float64);

// Int32 + Float64 = Float64
$result = $a->add($b);
echo $result->dtype();  // DType::Float64

// UInt8 + Int8 = Int16 (safe promotion)
$u8 = NDArray::array([1, 2], DType::UInt8);
$i8 = NDArray::array([1, 2], DType::Int8);
$result = $u8->add($i8);
echo $result->dtype();  // DType::Int16
```

## Type-Specific Behavior

### Integer Arithmetic

Integer operations truncate (not round):

```php
$a = NDArray::array([5, 7], DType::Int32);
$b = NDArray::array([2, 3], DType::Int32);

// Integer division truncates
$div = $a->divide($b);
print_r($div->toArray());  // [2, 2] (not [2.5, 2.33])

// Modulo works
$mod = $a->mod($b);
print_r($mod->toArray());  // [1, 1]
```

### Boolean Operations

Boolean arrays can be used with comparison operations and the `NDArray::where()` method for conditional logic:

```php
$a = NDArray::array([true, false, true], DType::Bool);
$b = NDArray::array([1, 2, 3], DType::Int64);
$c = NDArray::array([10, 20, 30], DType::Int64);

// Use boolean mask to select values
$result = NDArray::where($a, $c, $b);
print_r($result->toArray());  // [10, 2, 30]
```

### Float Precision

```php
// Float32 has ~7 decimal digits of precision
$f32 = NDArray::array([1.123456789], DType::Float32);
print_r($f32->toArray());  // [1.1234568]

// Float64 has ~15 decimal digits
$f64 = NDArray::array([1.123456789012345], DType::Float64);
print_r($f64->toArray());  // [1.123456789012345]
```

## Best Practices

### 1. Choose Types for Memory Efficiency

```php
// For images (0-255)
$image = NDArray::randomInt(0, 256, [224, 224, 3], DType::UInt8);
echo $image->nbytes() / (1024 * 1024);  // 3 MB

// Same as Float64 would be:
$image_f64 = $image->astype(DType::Float64);
echo $image_f64->nbytes() / (1024 * 1024);  // 24 MB
```

### 2. Use Float32 for ML

```php
// Most ML models use Float32
$weights = NDArray::random([1000, 1000], DType::Float32);
$activations = NDArray::zeros([32, 1000], DType::Float32);
```

### 3. Use Integers for Counting

```php
// Counts, indices, IDs
$counts = NDArray::array([10, 20, 30], DType::Int32);
$indices = NDArray::arange(0, 100, dtype: DType::Int64);
```

### 4. Be Careful with Unsigned

```php
$u8 = NDArray::array([0, 1, 2], DType::UInt8);

// Underflow! Wraps around
$result = $u8->subtract(1);
print_r($result->toArray());  // [255, 0, 1]
```

### 5. Check Types Before Operations

```php
function safeDivide(NDArray $a, NDArray $b): NDArray {
    // Ensure float division
    if ($a->dtype() === DType::Int64 || $b->dtype() === DType::Int64) {
        $a = $a->astype(DType::Float64);
        $b = $b->astype(DType::Float64);
    }
    return $a->divide($b);
}
```

## Common Patterns

### Image Processing

```php
use PhpMlKit\NDArray\DType;

// RGB image: UInt8 for pixel values (0-255)
$image = NDArray::randomInt(0, 256, [224, 224, 3], DType::UInt8);

// Normalize to Float32 for processing
$normalized = $image->astype(DType::Float32)->divide(255.0);

// Process...

// Convert back to UInt8 for saving
$output = $normalized->multiply(255.0)->astype(DType::UInt8);
```

### Audio Processing

```php
// Audio samples: Int16 or Float32
$audio = NDArray::randomInt(-32768, 32768, [44100 * 5], DType::Int16);

// Convert to Float32 for processing (-1.0 to 1.0)
$audio_f32 = $audio->astype(DType::Float32)->divide(32768.0);

// Apply effects...

// Convert back
$output = $audio_f32->multiply(32768.0)->astype(DType::Int16);
```

### Financial Calculations

```php
// Use Int64 for cents to avoid float errors
$prices_cents = NDArray::array([1999, 2995, 999], DType::Int64);

// Calculate
$total_cents = $prices_cents->sum();
$total_dollars = $total_cents / 100.0;

echo "Total: $$total_dollars";  // Total: $59.93
```

### Neural Networks

```php
// Mixed precision
$weights = NDArray::random([784, 256], DType::Float32);
$bias = NDArray::zeros([256], DType::Float32);

// Inference
$input = NDArray::random([1, 784], DType::Float32);
$hidden = $input->matmul($weights)->add($bias);
$activated = $hidden->sigmoid();
```

## Type Conversions Reference

| From | To | Behavior |
|------|-----|----------|
| Float | Int | Truncation (1.9 → 1) |
| Int | Float | Exact conversion |
| Bool | Int | true → 1, false → 0 |
| Int | Bool | 0 → false, non-zero → true |
| UInt | Int | Direct if fits, wraps if overflow |
| Int | UInt | Direct if positive, wraps if negative |
| Float32 | Float64 | Exact |
| Float64 | Float32 | Precision loss |

## Next Steps

- **[Indexing](/guide/fundamentals/indexing)** - Access array elements
- **[Views vs Copies](/guide/fundamentals/views-vs-copies)** - Understanding memory
- **[Performance](/guide/advanced/performance)** - Type optimization tips
