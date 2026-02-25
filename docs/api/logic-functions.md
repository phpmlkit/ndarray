# Logic Functions

Reference for boolean and comparison operations.

These methods return boolean arrays based on element-wise comparisons between arrays or between an array and a scalar value.

---

## eq()

```php
public function eq(float|int|NDArray $other): NDArray
```

Element-wise equal comparison.

Returns a boolean array where each element is true if the corresponding elements are equal.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$other` | `float\|int\|NDArray` | Array or scalar to compare. |

### Returns

- `NDArray` - Boolean array with element-wise comparison results.

### Examples

```php
$a = NDArray::array([1, 2, 3, 4]);
$b = NDArray::array([1, 2, 0, 4]);
$result = $a->eq($b);
print_r($result->toArray());
// Output: [true, true, false, true]

// Scalar comparison
$result = $a->eq(2);
print_r($result->toArray());
// Output: [false, true, false, false]
```

---

## ne()

```php
public function ne(float|int|NDArray $other): NDArray
```

Element-wise not-equal comparison.

Returns a boolean array where each element is true if the corresponding elements are not equal.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$other` | `float\|int\|NDArray` | Array or scalar to compare. |

### Returns

- `NDArray` - Boolean array with element-wise comparison results.

### Examples

```php
$a = NDArray::array([1, 2, 3, 4]);
$b = NDArray::array([1, 2, 0, 4]);
$result = $a->ne($b);
print_r($result->toArray());
// Output: [false, false, true, false]

// Scalar comparison
$result = $a->ne(2);
print_r($result->toArray());
// Output: [true, false, true, true]
```

---

## gt()

```php
public function gt(float|int|NDArray $other): NDArray
```

Element-wise greater-than comparison.

Returns a boolean array where each element is true if this element is greater than the corresponding element in `$other`.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$other` | `float\|int\|NDArray` | Array or scalar to compare. |

### Returns

- `NDArray` - Boolean array with element-wise comparison results.

### Examples

```php
$a = NDArray::array([1, 5, 3, 8]);
$b = NDArray::array([2, 4, 3, 7]);
$result = $a->gt($b);
print_r($result->toArray());
// Output: [false, true, false, true]

// Scalar comparison
$result = $a->gt(3);
print_r($result->toArray());
// Output: [false, true, false, true]
```

---

## gte()

```php
public function gte(float|int|NDArray $other): NDArray
```

Element-wise greater-or-equal comparison.

Returns a boolean array where each element is true if this element is greater than or equal to the corresponding element in `$other`.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$other` | `float\|int\|NDArray` | Array or scalar to compare. |

### Returns

- `NDArray` - Boolean array with element-wise comparison results.

### Examples

```php
$a = NDArray::array([1, 5, 3, 8]);
$b = NDArray::array([2, 4, 3, 7]);
$result = $a->gte($b);
print_r($result->toArray());
// Output: [false, true, true, true]

// Scalar comparison
$result = $a->gte(3);
print_r($result->toArray());
// Output: [false, true, true, true]
```

---

## lt()

```php
public function lt(float|int|NDArray $other): NDArray
```

Element-wise less-than comparison.

Returns a boolean array where each element is true if this element is less than the corresponding element in `$other`.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$other` | `float\|int\|NDArray` | Array or scalar to compare. |

### Returns

- `NDArray` - Boolean array with element-wise comparison results.

### Examples

```php
$a = NDArray::array([1, 5, 3, 8]);
$b = NDArray::array([2, 4, 3, 7]);
$result = $a->lt($b);
print_r($result->toArray());
// Output: [true, false, false, false]

// Scalar comparison
$result = $a->lt(4);
print_r($result->toArray());
// Output: [true, false, true, false]
```

---

## lte()

```php
public function lte(float|int|NDArray $other): NDArray
```

Element-wise less-or-equal comparison.

Returns a boolean array where each element is true if this element is less than or equal to the corresponding element in `$other`.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$other` | `float\|int\|NDArray` | Array or scalar to compare. |

### Returns

- `NDArray` - Boolean array with element-wise comparison results.

### Examples

```php
$a = NDArray::array([1, 5, 3, 8]);
$b = NDArray::array([2, 4, 3, 7]);
$result = $a->lte($b);
print_r($result->toArray());
// Output: [true, false, true, false]

// Scalar comparison
$result = $a->lte(4);
print_r($result->toArray());
// Output: [true, false, true, false]
```

---

## and()

```php
public function and(NDArray $other): NDArray
```

Element-wise logical AND.

Converts both arrays to boolean (truthy check), then computes AND. Always returns a boolean array.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$other` | `NDArray` | Array to AND with. |

### Returns

- `NDArray` - Boolean array with logical AND results.

### Examples

```php
// Boolean arrays
$a = NDArray::array([true, false, true]);
$b = NDArray::array([true, true, false]);
$result = $a->and($b);
print_r($result->toArray());
// Output: [true, false, false]

// Integer arrays (truthy/falsy)
$a = NDArray::array([5, 0, 3]);
$b = NDArray::array([2, 1, 0]);
$result = $a->and($b);
print_r($result->toArray());
// Output: [true, false, false]
```

---

## or()

```php
public function or(NDArray $other): NDArray
```

Element-wise logical OR.

Converts both arrays to boolean (truthy check), then computes OR. Always returns a boolean array.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$other` | `NDArray` | Array to OR with. |

### Returns

- `NDArray` - Boolean array with logical OR results.

### Examples

```php
// Boolean arrays
$a = NDArray::array([true, false, true]);
$b = NDArray::array([false, true, false]);
$result = $a->or($b);
print_r($result->toArray());
// Output: [true, true, true]

// Integer arrays (truthy/falsy)
$a = NDArray::array([5, 0, 0]);
$b = NDArray::array([0, 1, 0]);
$result = $a->or($b);
print_r($result->toArray());
// Output: [true, true, false]
```

---

## not()

```php
public function not(): NDArray
```

Element-wise logical NOT.

Converts array to boolean (truthy check), then computes NOT. Always returns a boolean array.

### Returns

- `NDArray` - Boolean array with logical NOT results.

### Examples

```php
// Boolean array
$arr = NDArray::array([true, false, true]);
$result = $arr->not();
print_r($result->toArray());
// Output: [false, true, false]

// Integer array (truthy/falsy)
$arr = NDArray::array([5, 0, 3]);
$result = $arr->not();
print_r($result->toArray());
// Output: [false, true, false]
```

---

## xor()

```php
public function xor(NDArray $other): NDArray
```

Element-wise logical XOR.

Converts both arrays to boolean (truthy check), then computes XOR. Always returns a boolean array.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$other` | `NDArray` | Array to XOR with. |

### Returns

- `NDArray` - Boolean array with logical XOR results.

### Examples

```php
// Boolean arrays
$a = NDArray::array([true, false, true, false]);
$b = NDArray::array([true, true, false, false]);
$result = $a->xor($b);
print_r($result->toArray());
// Output: [false, true, true, false]

// Integer arrays (truthy/falsy)
$a = NDArray::array([5, 0, 3, 0]);
$b = NDArray::array([2, 1, 0, 0]);
$result = $a->xor($b);
print_r($result->toArray());
// Output: [false, true, true, false]
```

---

## Summary Table

### Comparison Operations

| Method | Operation | Example |
|--------|-----------|---------|
| `eq()` | Equal | `[1, 2] == [1, 3]` → `[true, false]` |
| `ne()` | Not equal | `[1, 2] != [1, 3]` → `[false, true]` |
| `gt()` | Greater than | `[1, 2] > [1, 1]` → `[false, true]` |
| `gte()` | Greater or equal | `[1, 2] >= [1, 2]` → `[true, true]` |
| `lt()` | Less than | `[1, 2] < [2, 2]` → `[true, false]` |
| `lte()` | Less or equal | `[1, 2] <= [1, 2]` → `[true, true]` |

### Logical Operations

| Method | Operation | Input Types | Returns |
|--------|-----------|-------------|---------|
| `and()` | Logical AND | Any | Bool |
| `or()` | Logical OR | Any | Bool |
| `not()` | Logical NOT | Any | Bool |
| `xor()` | Logical XOR | Any | Bool |

---

## Next Steps

- [Mathematical Functions](/api/mathematical-functions)
- [Bitwise Operations](/api/bitwise-operations)
- [Indexing Routines](/api/indexing-routines)