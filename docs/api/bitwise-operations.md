# Bitwise Operations

Reference for bitwise operations on integer and boolean arrays.

These methods perform bitwise operations element-wise. They work with integer types (Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64) and Bool types.

For boolean arrays, bitwise operations act as logical operations:
- `bitand()` = logical AND (returns true only if both are true)
- `bitor()` = logical OR (returns true if either is true)
- `bitxor()` = logical XOR (returns true if exactly one is true)

::: tip
For logical operations that work with any type and always return booleans, see [Logic Functions](/api/logic-functions).
:::

---

## bitand()

```php
public function bitand(int|NDArray $other): NDArray
```

Bitwise AND with another array or scalar.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$other` | `int\|NDArray` | Array or scalar. |

### Returns

- `NDArray` - New array with bitwise AND result.

### Examples

**Integer arrays:**
```php
$a = NDArray::array([0b1100, 0b1010, 0b1111]);
$b = NDArray::array([0b1010, 0b1100, 0b1010]);
$result = $a->bitand($b);
print_r($result->toArray());
// Output: [8, 8, 10]  // 0b1000, 0b1000, 0b1010
```

**Boolean arrays (acts as logical AND):**
```php
$mask1 = NDArray::array([true, false, true]);
$mask2 = NDArray::array([true, true, false]);
$result = $mask1->bitand($mask2);
print_r($result->toArray());
// Output: [true, false, false]
```

---

## bitor()

```php
public function bitor(int|NDArray $other): NDArray
```

Bitwise OR with another array or scalar.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$other` | `int\|NDArray` | Array or scalar. |

### Returns

- `NDArray` - New array with bitwise OR result.

### Examples

```php
$a = NDArray::array([0b1100, 0b1010, 0b1111]);
$b = NDArray::array([0b1010, 0b1100, 0b1010]);
$result = $a->bitor($b);
print_r($result->toArray());
// Output: [14, 14, 15]  // 0b1110, 0b1110, 0b1111
```

---

## bitxor()

```php
public function bitxor(int|NDArray $other): NDArray
```

Bitwise XOR with another array or scalar.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$other` | `int\|NDArray` | Array or scalar. |

### Returns

- `NDArray` - New array with bitwise XOR result.

### Examples

```php
$a = NDArray::array([0b1100, 0b1010, 0b1111]);
$b = NDArray::array([0b1010, 0b1100, 0b1010]);
$result = $a->bitxor($b);
print_r($result->toArray());
// Output: [6, 6, 5]  // 0b0110, 0b0110, 0b0101
```

---

## leftShift()

```php
public function leftShift(int|NDArray $other): NDArray
```

Left shift by another array or scalar.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$other` | `int\|NDArray` | Array or scalar (number of bits to shift). |

### Returns

- `NDArray` - New array with left-shifted values.

### Examples

```php
$arr = NDArray::array([1, 2, 4, 8]);
$result = $arr->leftShift(1);
print_r($result->toArray());
// Output: [2, 4, 8, 16]
```

---

## rightShift()

```php
public function rightShift(int|NDArray $other): NDArray
```

Right shift by another array or scalar.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$other` | `int\|NDArray` | Array or scalar (number of bits to shift). |

### Returns

- `NDArray` - New array with right-shifted values.

### Examples

```php
$arr = NDArray::array([8, 16, 32, 64]);
$result = $arr->rightShift(2);
print_r($result->toArray());
// Output: [2, 4, 8, 16]
```

---

## Summary Table

| Method | Operation | Example |
|--------|-----------|---------|
| `bitand()` | Bitwise AND | `0b1100 & 0b1010` → `0b1000` |
| `bitor()` | Bitwise OR | `0b1100 | 0b1010` → `0b1110` |
| `bitxor()` | Bitwise XOR | `0b1100 ^ 0b1010` → `0b0110` |
| `leftShift()` | Left shift | `4 << 1` → `8` |
| `rightShift()` | Right shift | `16 >> 2` → `4` |

---

## Next Steps

- [Logic Functions](/api/logic-functions)
- [Mathematical Functions](/api/mathematical-functions)