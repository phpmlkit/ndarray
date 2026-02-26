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
public function intoBuffer(CData $buffer, ?int $maxElements = null): int
```

### Parameters

| Name | Type | Description |
|------|------|-------------|
| `$buffer` | `CData` | Destination typed C buffer (FFI) |
| `$maxElements` | `int\|null` | Maximum elements the destination can hold. Default: full size |

### Returns

- `int` - Number of elements copied.

### Examples

```php
$arr = NDArray::array([1.0, 2.0, 3.0]);
$ffi = \PhpMlKit\NDArray\FFI\Lib::get();
$buffer = $ffi->new('double[3]');
$n = $arr->intoBuffer($buffer);
// $n === 3
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