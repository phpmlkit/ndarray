# Array Import and Export

Reference for converting arrays to and from other formats.

These methods allow you to convert NDArray objects to PHP native types, raw bytes, or other formats for interoperability.

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

Return raw bytes of the array/view in C-order.

```php
public function toBytes(): string
```

### Parameters

None.

### Returns

- `string` - Raw binary representation of the array data.

### Examples

```php
$arr = NDArray::array([1.0, 2.0, 3.0], DType::Float64);
$bytes = $arr->toBytes();
// Length = 3 * 8 = 24 bytes for Float64
```

---

## intoBuffer()

Copy flattened C-order data into a caller-allocated C buffer.

```php
public function intoBuffer(CData $buffer, int $start = 0, ?int $len = null): int
```

### Parameters

| Name | Type | Description |
|------|------|-------------|
| `$buffer` | `CData` | Destination typed C buffer (FFI) |
| `$start` | `int` | Starting element offset (0-indexed). Default: 0 |
| `$len` | `int\|null` | Number of elements to copy. Default: null (copy to end) |

### Returns

- `int` - Number of elements copied.

### Examples

```php
$arr = NDArray::array([1.0, 2.0, 3.0, 4.0, 5.0]);
$ffi = \PhpMlKit\NDArray\FFI\Lib::get();
$buffer = $ffi->new('double[5]');

// Copy all elements
$n = $arr->intoBuffer($buffer);
// $n === 5

// Copy from offset
$buffer = $ffi->new('double[3]');
$n = $arr->intoBuffer($buffer, 2);  // Start at index 2
// $n === 3 (elements 2, 3, 4)

// Copy with explicit length
$buffer = $ffi->new('double[2]');
$n = $arr->intoBuffer($buffer, 1, 2);  // Start at 1, copy 2 elements
// $n === 2 (elements 1, 2)
```

---

## Summary Table

| Method | Output Format | Use Case |
|--------|---------------|----------|
| `toArray()` | Nested PHP array | Export to PHP code |
| `toScalar()` | Single value | Extract 0D array value |
| `toBytes()` | Binary string | Binary serialization |
| `intoBuffer()` | FFI C buffer | Low-level FFI interop |

---

## Next Steps

- [Array Creation](/api/array-creation) - Converting from PHP arrays
- [NDArray Class](/api/ndarray-class) - Array properties and metadata