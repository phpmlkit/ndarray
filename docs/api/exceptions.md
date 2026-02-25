# Exceptions

NDArray throws specific exceptions for different error conditions.

## Exception Hierarchy

```
NDArrayException (base)
├── ShapeException
├── IndexException
├── DTypeException
├── AllocationException
├── MathException
└── PanicException
```

## NDArrayException

Base exception class for all NDArray errors.

```php
use PhpMlKit\NDArray\Exceptions\NDArrayException;

try {
    $arr = NDArray::array([]);
} catch (NDArrayException $e) {
    echo $e->getMessage();
}
```

---

## ShapeException

Thrown when array shapes are incompatible.

```php
use PhpMlKit\NDArray\Exceptions\ShapeException;

$a = NDArray::zeros([3, 3]);
$b = NDArray::zeros([4, 4]);

try {
    $c = $a->add($b);  // Throws ShapeException
} catch (ShapeException $e) {
    echo "Shape mismatch: " . $e->getMessage();
}
```

**Common causes:**
- Incompatible shapes for arithmetic operations
- Invalid reshape dimensions
- Broadcasting failures

---

## IndexException

Thrown for invalid indexing operations.

```php
use PhpMlKit\NDArray\Exceptions\IndexException;

$arr = NDArray::array([1, 2, 3]);

try {
    $val = $arr->get(10);  // Throws IndexException
} catch (IndexException $e) {
    echo "Invalid index: " . $e->getMessage();
}
```

**Common causes:**
- Index out of bounds
- Wrong number of indices
- Invalid slice syntax

---

## DTypeException

Thrown for data type errors.

```php
use PhpMlKit\NDArray\Exceptions\DTypeException;
```

**Common causes:**
- Unsupported type conversions
- Type mismatch in operations

---

## AllocationException

Thrown when memory allocation fails.

```php
use PhpMlKit\NDArray\Exceptions\AllocationException;

try {
    $arr = NDArray::zeros([1000000, 1000000]);  // May throw
} catch (AllocationException $e) {
    echo "Out of memory: " . $e->getMessage();
}
```

**Common causes:**
- Requesting too much memory
- System memory exhausted

---

## MathException

Thrown for mathematical errors.

```php
use PhpMlKit\NDArray\Exceptions\MathException;
```

**Common causes:**
- Division by zero
- Domain errors (e.g., sqrt of negative)

---

## PanicException

Thrown when Rust panics (unexpected error).

```php
use PhpMlKit\NDArray\Exceptions\PanicException;
```

::: danger
This indicates an internal error. Please report bugs if you encounter this.
:::

---

## Handling Exceptions

### Basic Try-Catch

```php
use PhpMlKit\NDArray\NDArray;
use PhpMlKit\NDArray\Exceptions\NDArrayException;

try {
    $arr = NDArray::array($data);
    $result = $arr->resolve([3, 3]);
} catch (NDArrayException $e) {
    // Handle any NDArray error
    echo "Error: " . $e->getMessage();
}
```

### Specific Exception Handling

```php
use PhpMlKit\NDArray\Exceptions\ShapeException;
use PhpMlKit\NDArray\Exceptions\IndexException;

try {
    // Operation that might fail
    $result = processArray($arr);
} catch (ShapeException $e) {
    // Handle shape errors specifically
    echo "Shape error: " . $e->getMessage();
} catch (IndexException $e) {
    // Handle index errors
    echo "Index error: " . $e->getMessage();
} catch (NDArrayException $e) {
    // Handle other NDArray errors
    echo "NDArray error: " . $e->getMessage();
}
```

### Validation Before Operations

```php
function safeAdd(NDArray $a, NDArray $b): ?NDArray {
    // Check shapes before operation
    if ($a->shape() !== $b->shape()) {
        echo "Warning: Shape mismatch\n";
        return null;
    }
    
    try {
        return $a->add($b);
    } catch (NDArrayException $e) {
        echo "Error: " . $e->getMessage();
        return null;
    }
}
```

---

## Summary Table

| Exception | When Thrown | Catch When |
|-----------|-------------|------------|
| `NDArrayException` | Base for all errors | General error handling |
| `ShapeException` | Shape mismatch | Arithmetic, reshape operations |
| `IndexException` | Invalid index | Indexing, slicing operations |
| `DTypeException` | Type error | Type conversions |
| `AllocationException` | Memory error | Large array creation |
| `MathException` | Math error | Division, sqrt, log operations |
| `PanicException` | Rust panic | Report as bug |

---

## Next Steps

- [NDArray Class](/api/ndarray-class)
- [API Reference Index](/api/)
