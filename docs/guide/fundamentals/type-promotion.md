# Type Promotion

When performing operations on arrays with different data types, NDArray must decide what type the result should be. This process is called **type promotion**, and it follows NumPy-compatible rules that balance precision, memory efficiency, and mathematical correctness.

## The Two Axes of Promotion

Type promotion operates along two independent axes:

1. **Kind**: The fundamental category of the data type
2. **Precision**: The width within a given kind

### Kind Hierarchy

The kind hierarchy determines the broad category of the result:

```
Integer → Float → Complex
```

When types of different kinds are combined, the result is the "higher" kind:

| Combination | Result Kind | Reason |
|-------------|-------------|--------|
| Integer + Float | Float | Float can represent all integers |
| Integer + Complex | Complex | Complex can represent all real numbers |
| Float + Complex | Complex | Complex includes the real line |

### Precision Rules

Within the same kind, the wider (higher precision) type wins:

| Combination | Result | Reason |
|-------------|--------|--------|
| Int32 + Int64 | Int64 | Higher precision |
| Float32 + Float64 | Float64 | Higher precision |
| Complex64 + Complex128 | Complex128 | Higher precision |

### Putting It Together

The full promotion ladder:

```
Bool → UInt8 → UInt16 → UInt32 → UInt64
  ↓
Int8 → Int16 → Int32 → Int64
  ↓
Float32 → Float64
  ↓
Complex64 → Complex128
```

## Binary Type Promotion (Array × Array)

When two arrays are combined in an operation, both kind and precision are considered:

```php
use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\NDArray;

// Different precision, same kind
$a = NDArray::array([1, 2], DType::Int32);
$b = NDArray::array([1, 2], DType::Int64);
$result = $a->add($b);
echo $result->dtype();  // Int64

// Different kinds
$int = NDArray::array([1, 2], DType::Int64);
$float = NDArray::array([1.5, 2.5], DType::Float32);
$result = $int->add($float);
echo $result->dtype();  // Float32 (float kind wins)

// Integer + Complex
$int = NDArray::array([1, 2], DType::Int64);
$complex = NDArray::array([
    new Complex(1, 2),
    new Complex(3, 4),
], DType::Complex128);
$result = $int->add($complex);
echo $result->dtype();  // Complex128 (complex kind wins)
```

## Scalar Type Promotion (Array × Scalar)

When an array is combined with a scalar, the scalar's type is first determined, then promotion proceeds as with two arrays:

```php
// Int array + float scalar = Float64
$int = NDArray::array([1, 2, 3], DType::Int64);
$result = $int->add(1.5);
echo $result->dtype();  // Float64

// Float array + int scalar = Float64
$float = NDArray::array([1.0, 2.0, 3.0], DType::Float64);
$result = $float->add(10);
echo $result->dtype();  // Float64 (float kind wins)

// Int array + complex scalar = Complex128
$int = NDArray::array([1, 2, 3], DType::Int64);
$result = $int->add(new Complex(1, 2));
echo $result->dtype();  // Complex128
```

### Scalar Type Inference

Scalars are assigned types based on their PHP type:

| PHP Type | Inferred DType |
|----------|----------------|
| `int` | Int64 |
| `float` | Float64 |
| `Complex` | Complex128 |

This means `$intArray + 1.5` promotes the scalar to Float64 first, then promotes the result to Float64.

## Unsigned + Signed Promotion

A special rule handles the combination of unsigned and signed integers. Since unsigned types cannot represent negative values, the result must be a signed type wide enough to hold the full range:

```php
$u8 = NDArray::array([1, 2], DType::UInt8);   // 0 to 255
$i8 = NDArray::array([1, 2], DType::Int8);    // -128 to 127

// UInt8 + Int8 = Int16 (neither Int8 nor UInt8 can hold all results)
$result = $u8->add($i8);
echo $result->dtype();  // Int16
```

The rule is: **unsigned + signed → signed type with at least one extra bit of precision**.

## Complex Number Promotion

Complex types follow the same principles, with an additional sub-kind hierarchy:

```
Real (Int/Float) → Complex
```

When a real array is combined with a complex array or scalar, the result is complex:

```php
$real = NDArray::array([1.0, 2.0], DType::Float64);
$complex = NDArray::array([
    new Complex(1, 2),
    new Complex(3, 4),
], DType::Complex64);

// Float64 + Complex64 = Complex64
$result = $real->add($complex);
echo $result->dtype();  // Complex64

// Float64 + Complex128 = Complex128
$complex128 = NDArray::array([
    new Complex(1, 2),
], DType::Complex128);
$result = $real->add($complex128);
echo $result->dtype();  // Complex128
```

::: tip Note
The real part of a complex result retains the original real value, and the imaginary part is zero for elements that were originally real.
:::

## Why This Matters

Understanding type promotion helps you:

1. **Predict result types**: Know what dtype your operation will produce
2. **Avoid precision loss**: Be aware when Float32 results get promoted to Float64
3. **Manage memory**: Complex arrays use twice the memory of float arrays
4. **Debug unexpected results**: Integer division truncates, float division does not

## Common Pitfalls

### Integer Division

```php
$a = NDArray::array([5, 7], DType::Int64);
$b = NDArray::array([2, 3], DType::Int64);

// Integer division truncates
$result = $a->divide($b);
print_r($result->toArray());  // [2, 2] — not [2.5, 2.33]

// Solution: cast to float first
$result = $a->astype(DType::Float64)->divide($b);
print_r($result->toArray());  // [2.5, 2.33]
```

### Unexpected Complex Results

```php
$int = NDArray::array([1, 2, 3], DType::Int64);

// Adding a complex scalar promotes everything to Complex128
$result = $int->add(new Complex(0, 1));
echo $result->dtype();  // Complex128

// If you only need the real part, extract it
$realOnly = $result->real();
echo $realOnly->dtype();  // Float64
```

### Float32 vs Float64

```php
$f32 = NDArray::array([1.0, 2.0], DType::Float32);
$f64 = NDArray::array([1.0, 2.0], DType::Float64);

// Result is Float64 (higher precision)
$result = $f32->add($f64);
echo $result->dtype();  // Float64
```

## Best Practices

### 1. Be Explicit When It Matters

```php
// If you need Float32 results, ensure both inputs are Float32
$a = NDArray::array([1, 2, 3], DType::Float32);
$b = NDArray::array([4, 5, 6], DType::Float32);
$result = $a->add($b);  // Float32
```

### 2. Check Types in Debug Mode

```php
function debugOperation(NDArray $a, NDArray $b, string $op): void {
    $result = match($op) {
        'add' => $a->add($b),
        'mul' => $a->multiply($b),
        default => throw new \InvalidArgumentException("Unknown op: $op"),
    };

    echo "Input A: {$a->dtype()->name}, Input B: {$b->dtype()->name}, Result: {$result->dtype()->name}\n";
}
```

### 3. Use `astype()` for Control

```php
$data = NDArray::array([1, 2, 3]);  // Int64

// Force float division
$result = $data->astype(DType::Float64)->divide(2);
print_r($result->toArray());  // [0.5, 1.0, 1.5]
```

## Next Steps

- **[Data Types](/guide/fundamentals/data-types)** - Complete list of supported types
- **[Operations](/guide/fundamentals/operations)** - How operations work with different types
- **[Performance](/guide/advanced/performance)** - Type optimization for memory and speed
