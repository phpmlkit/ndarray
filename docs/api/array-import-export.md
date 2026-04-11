# Array Import and Export

Reference for converting arrays to and from other formats.

These methods allow you to convert NDArray objects to PHP native types, raw bytes, or FFI buffers for interoperability.

---

## toArray()

Convert to nested PHP array matching the array shape.

```php
public function toArray(): array|bool|float|int
```

Uses flat typed extraction from Rust and builds nested structure in PHP. For 0-dimensional arrays, returns a scalar.

### Parameters

None.

### Returns

- `array<mixed>|bool|float|int` - Nested array for N-dimensional arrays, scalar for 0-dimensional.

### Examples

```php
$arr = NDArray::array([[1, 2], [3, 4]]);
print_r($arr->toArray());
// Output: [[1, 2], [3, 4]]

$scalar = NDArray::array([42])->squeeze();
var_dump($scalar->toArray());  // int(42)
```
---

## toScalar()

Convert 0-dimensional array to scalar.

```php
public function toScalar(): bool|float|int
```

Returns float, int, or bool depending on the array's dtype. Throws if the array is not 0-dimensional.

### Parameters

None.

### Returns

- `bool|float|int` - Single scalar value.

### Examples

```php
// 0D array from squeezing a single-element array
$arr = NDArray::array([3.14])->squeeze();
echo $arr->toScalar();  // 3.14
```

---

## toBytes()

Return raw bytes of the array/view in C-order as a binary string.

```php
public function toBytes(): string
```

Returns the raw binary representation in little-endian format. Useful for serialization, file I/O, or passing to other systems that expect binary data.

### Parameters

None.

### Returns

- `string` - Raw binary representation of the array data in little-endian format.

### Examples

```php
$arr = NDArray::array([1.0, 2.0, 3.0], DType::Float64);
$bytes = $arr->toBytes();
// Length = 3 * 8 = 24 bytes for Float64

// Save to file
file_put_contents('data.bin', $bytes);
```

---

## toBuffer()

Export NDArray data to a C buffer for FFI interoperability.

```php
public function toBuffer(?CData $buffer = null, int $start = 0, ?int $len = null): CData
```

Copies flattened C-order data into a C buffer. If no buffer is provided, allocates a new one with the appropriate type. The returned CData is owned by PHP's FFI and will be garbage collected when no longer referenced.

### Parameters

| Name | Type | Description |
|------|------|-------------|
| `$buffer` | `CData\|null` | Optional destination typed C buffer. If null, a new buffer is allocated. |
| `$start` | `int` | Starting element offset (0-indexed). Default: 0 |
| `$len` | `int\|null` | Number of elements to copy. Default: null (copy remaining elements from start) |

### Returns

- `CData` - The buffer containing the copied data (either provided or newly allocated).

### Examples

```php
$arr = NDArray::array([1.0, 2.0, 3.0, 4.0, 5.0]);
$ffi = \PhpMlKit\NDArray\FFI\Lib::get();

// Allocate and copy all elements (new buffer)
$buffer = $arr->toBuffer();
// $buffer is CData (double[5])

// Copy into existing buffer
$existingBuffer = $ffi->new('double[5]');
$buffer = $arr->toBuffer($existingBuffer);
// $buffer === $existingBuffer

// Copy from offset
$buffer = $arr->toBuffer(null, 2);  // Start at index 2
// $buffer contains elements 2, 3, 4 (indices 2, 3, 4)

// Copy with explicit length
$buffer = $arr->toBuffer(null, 1, 2);  // Start at 1, copy 2 elements
// $buffer contains elements 1, 2 (indices 1, 2)
```

### Important Notes

**Buffer Lifetime**: When `toBuffer()` allocates a buffer (no `$buffer` argument), the returned `CData` is managed by PHP's FFI. It will be garbage collected when no longer referenced. If you need the data to persist beyond the current scope, copy it to a location you control.

**Type Safety**: The buffer must match the array's dtype exactly. Passing a `float*` buffer for a `Float64` array will result in incorrect data.

---

## intoBuffer() (Deprecated)

::: warning Deprecated
`intoBuffer()` is deprecated and will be removed in a future version. Use `toBuffer()` instead.
:::

Copy flattened C-order data into a caller-allocated C buffer.

```php
public function intoBuffer(CData $buffer, int $start = 0, ?int $len = null): int
```

This method is functionally equivalent to calling `toBuffer($buffer, $start, $len)` and discarding the return value. It returns the number of elements copied instead of the buffer.

### Migration Guide

Replace:
```php
$n = $arr->intoBuffer($buffer, 0, 100);
```

With:
```php
$buffer = $arr->toBuffer($buffer, 0, 100);
// $buffer now contains the data
```

---

## Summary Table

| Method | Output Format | Use Case |
|--------|---------------|----------|
| `toArray()` | Nested PHP array | Export to PHP code |
| `toScalar()` | Single value | Extract 0D array value |
| `toBytes()` | Binary string | Binary serialization, file I/O |
| `toBuffer()` | FFI C buffer | Low-level FFI interop |

---

## Next Steps

- [Array Creation](/api/array-creation) - Converting from PHP arrays and binary data
- [NDArray Class](/api/ndarray-class) - Array properties and metadata
