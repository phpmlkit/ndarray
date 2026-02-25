# Indexing and Slicing

Accessing and modifying array elements through indexing and slicing.

## Basic Indexing

Access individual elements using integer indices:

```php
$arr = NDArray::array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]);

// Get element at row 1, column 2
$value = $arr->get(1, 2);  // Returns 6
```

**Multi-dimensional Access Patterns:**

```php
$tensor = NDArray::array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
]);

// Access: layer 0, row 1, column 0
$val = $tensor->get(0, 1, 0);  // Returns 3
```

**ArrayAccess Interface:**

You can also use array syntax:

```php
$arr = NDArray::array([10, 20, 30, 40]);

echo $arr[2];  // 30

// Note: For multi-dimensional, use comma notation
$matrix = NDArray::array([[1, 2], [3, 4]]);
echo $matrix['1,0'];  // 3 (row 1, column 0)
```

**Negative Indices:**

Count from the end using negative numbers:

```php
$arr = NDArray::array([10, 20, 30, 40, 50]);

echo $arr[-1];  // 50 (last element)
echo $arr[-2];  // 40 (second to last)
```

## Basic Slicing

Extract portions of arrays using slice notation.

**Slice Syntax:** `start:stop:step`

```
Array: [0] [1] [2] [3] [4] [5] [6] [7]
        ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑
        0   1   2   3   4   5   6   7  ← Positive indices
       -8  -7  -6  -5  -4  -3  -2  -1  ← Negative indices
```

**Common Slice Patterns:**

```php
$arr = NDArray::array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

// Elements 2 through 5 (exclusive of 5)
$slice = $arr->slice(['2:5']);        // [2, 3, 4]

// From start to index 5
$slice = $arr->slice([':5']);         // [0, 1, 2, 3, 4]

// From index 5 to end
$slice = $arr->slice(['5:']);         // [5, 6, 7, 8, 9]

// All elements (copy)
$slice = $arr->slice([':']);          // [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

// Every second element
$slice = $arr->slice(['::2']);        // [0, 2, 4, 6, 8]

// Reverse
$slice = $arr->slice(['::-1']);       // [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

// Last 3 elements
$slice = $arr->slice(['-3:']);        // [7, 8, 9]
```

## Multi-dimensional Slicing

Slice along each dimension independently:

```php
$matrix = NDArray::array([
    [1,  2,  3,  4],
    [5,  6,  7,  8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
]);

// Rows 0-1, Columns 1-3
$slice = $matrix->slice(['0:2', '1:4']);
// [[2, 3, 4],
//  [6, 7, 8]]

// All rows, columns 1-2
$slice = $matrix->slice([':', '1:3']);
// [[2, 3],
//  [6, 7],
//  [10, 11],
//  [14, 15]]

// Every other row
$slice = $matrix->slice(['::2', ':']);
// [[1, 2, 3, 4],
//  [9, 10, 11, 12]]
```

**Visual Guide:**

```
Original Matrix [4, 4]:
┌───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │ ← Row 0
├───┼───┼───┼───┤
│ 5 │ 6 │ 7 │ 8 │ ← Row 1
├───┼───┼───┼───┤
│ 9 │10 │11 │12 │ ← Row 2
├───┼───┼───┼───┤
│13 │14 │15 │16 │ ← Row 3
└───┴───┴───┴───┘
  ↑   ↑   ↑   ↑
 Col 0 1   2   3

Slice ['1:3', '1:3'] (Rows 1-2, Cols 1-2):
┌───┬───┐
│ 6 │ 7 │
├───┼───┤
│10 │11 │
└───┴───┘
```

## Boolean Indexing (Masking)

Select elements using boolean arrays:

```php
$arr = NDArray::array([1, 2, 3, 4, 5, 6]);

// Create a mask
$mask = $arr->gt(3);  // [false, false, false, true, true, true]

// Select elements where mask is true
$result = NDArray::where($mask, $arr, NDArray::zeros([6]));
// [0, 0, 0, 4, 5, 6]

// More complex conditions
$arr = NDArray::array([10, 25, 30, 45, 50, 65]);

// Find values between 30 and 50 (inclusive)
$gte30 = $arr->gte(30);
$lte50 = $arr->lte(50);
$mask = $gte30->and($lte50);
$result = NDArray::where($mask, $arr, NDArray::zeros([6]));
// [0, 0, 30, 45, 50, 0]
```

## Integer Array Indexing

Select specific elements by their indices:

```php
$arr = NDArray::array([10, 20, 30, 40, 50]);

// Take elements at specific indices
$indices = NDArray::array([0, 2, 4]);
$result = $arr->take($indices);
// [10, 30, 50]

// Multi-dimensional indexing
$matrix = NDArray::array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]);

// Take rows 0 and 2
$rows = NDArray::array([0, 2]);
$result = $matrix->take($rows, axis: 0);
// [[1, 2, 3],
//  [7, 8, 9]]
```

## Setting Values

Modify elements in place:

```php
$arr = NDArray::array([1, 2, 3, 4, 5]);

// Set single element
$arr->setAt(2, 999);
echo $arr->toArray();  // [1, 2, 999, 4, 5]

// Set with slicing
$arr = NDArray::array([1, 2, 3, 4, 5, 6]);
$arr->put(['1:4'], NDArray::array([10, 20, 30]));
echo $arr->toArray();  // [1, 10, 20, 30, 5, 6]
```

## Views vs Copies

**Slices return views** (share memory):

```php
$original = NDArray::array([1, 2, 3, 4, 5]);
$view = $original->slice(['1:4']);

$view->setAt(0, 999);
echo $original->toArray();  // [1, 999, 3, 4, 5] ← Original changed!
```

**Integer indexing returns copies:**

```php
$original = NDArray::array([1, 2, 3, 4, 5]);
$copy = $original->take(NDArray::array([1, 2, 3]));

$copy->setAt(0, 999);
echo $original->toArray();  // [1, 2, 3, 4, 5] ← Original unchanged
```

See [Views vs Copies](/guide/fundamentals/views-vs-copies) for complete details.

## Next Steps

- **[Broadcasting](/guide/fundamentals/broadcasting)** - Operations between different shapes
- **[Operations](/guide/fundamentals/operations)** - Working with array data
- **[API Reference: Indexing Routines](/api/indexing-routines)** - Complete method reference
