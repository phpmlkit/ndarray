# Printing NDArrays

Controlling how arrays are displayed when used as strings.

NDArray implements PHP's `Stringable` interface, meaning you can use arrays directly in any context that expects a string—echo, print, string concatenation, or functions like `sprintf()`.

## Basic Usage

The simplest way to see an array's contents:

```php
$arr = NDArray::array([1, 2, 3, 4, 5]);
echo $arr;
```

Output:
```
array(5)
[1 2 3 4 5]
```

Every array display includes a header showing its shape, followed by the actual data.

### Echo and Print

```php
$arr = NDArray::array([[1, 2], [3, 4]]);

echo $arr;
print $arr;
```

Both `echo` and `print` produce the same output. The array header shows dimensions clearly:

```
array(2, 2)
[
 [1 2]
 [3 4]
]
```

### String Interpolation

Use arrays in string contexts:

```php
$arr = NDArray::array([10, 20, 30]);

// Direct interpolation
echo "Values: $arr";
// Values: array(3)
// [10 20 30]

// With sprintf
$output = sprintf("The array contains: %s", $arr);
// "The array contains: array(3)\n[10 20 30]"
```

### String Concatenation

```php
$a = NDArray::array([1, 2]);
$b = NDArray::array([3, 4]);

$result = "First: " . $a . "\nSecond: " . $b;
echo $result;
```

Output:
```
First: array(2)
[1 2]
Second: array(2)
[3 4]
```

## Output by Dimension

The display format adapts based on the number of dimensions.

### 0D Arrays (Scalars)

```php
$scalar = NDArray::array(42);
echo $scalar;
```

Output:
```
array(0)
42
```

### 1D Arrays

```php
$vec = NDArray::array([1, 2, 3, 4, 5]);
echo $vec;
```

Output:
```
array(5)
[1 2 3 4 5]
```

### 2D Arrays

```php
$matrix = NDArray::array([
    [1, 2, 3],
    [4, 5, 6]
]);
echo $matrix;
```

Output:
```
array(2, 3)
[
 [1 2 3]
 [4 5 6]
]
```

### 3D Arrays

```php
$tensor = NDArray::array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
]);
echo $tensor;
```

Output:
```
array(2, 2, 2)
[
  [1 2]
  [3 4]

  [5 6]
  [7 8]
]
```

### 4D and Higher

Higher-dimensional arrays display with nested brackets:

```php
$data = NDArray::random([2, 2, 3, 4]);
echo $data;
```

Output:
```
array(2, 2, 3, 4)
[
  [
    [
      [0.123 0.456 0.789 0.234]
      [0.567 0.890 0.123 0.456]
      [0.789 0.234 0.567 0.890]]

    [
      [0.901 0.234 0.567 0.890]
      [0.123 0.456 0.789 0.234]
      [0.456 0.789 0.901 0.123]
    ]
  ]

  [
    [
      [0.234 0.567 0.890 0.123]
      [0.456 0.789 0.234 0.567]
      [0.890 0.123 0.456 0.789]]

    [
      [0.789 0.234 0.567 0.890]
      [0.123 0.456 0.789 0.234]
      [0.567 0.890 0.123 0.456]
    ]
  ]
]
```

## Print Options

Control array formatting globally using print options. These settings affect all subsequent array displays in your application.

### setPrintOptions()

Configure how arrays are displayed:

```php
NDArray::setPrintOptions(
    threshold: 1000,  // Max elements before truncation
    edgeitems: 3,      // Items to show at edges when truncating
    precision: 8       // Decimal places for floats
);
```

**Parameters:**
- `threshold` - Maximum number of elements before array is truncated. Default: 1000
- `edgeitems` - Number of items to show at each edge when truncating. Default: 3
- `precision` - Number of decimal places for floating-point numbers. Default: 8

### getPrintOptions()

Retrieve current print settings:

```php
$options = NDArray::getPrintOptions();
print_r($options);
// [
//     'threshold' => 1000,
//     'edgeitems' => 3,
//     'precision' => 8
// ]
```

### resetPrintOptions()

Restore defaults:

```php
NDArray::resetPrintOptions();
// Now uses: threshold=1000, edgeitems=3, precision=8
```

## Next Steps

- **[Operations](/guide/fundamentals/operations)** - Learn about array computations
- **[Indexing and Slicing](/guide/fundamentals/indexing-and-slicing)** - Access array elements
- **[API Reference](/api/ndarray-class)** - Complete method documentation
