# PHP NDArray Library Specification

## 1. Project Overview

### 1.1 Project Name
**NDArray PHP** - A high-performance N-dimensional array library for PHP, backed by Rust

### 1.2 Purpose
To provide PHP developers with a NumPy-like interface for efficient numerical computing, scientific computing, and data manipulation, leveraging Rust's performance and safety through FFI.

### 1.3 Core Goals
- **Performance**: Near-native performance through Rust implementation
- **Familiarity**: NumPy-inspired API for easy adoption
- **Safety**: Proper memory management across FFI boundary
- **Flexibility**: Support for various data types, memory layouts, and operations
- **Optional BLAS**: High-performance linear algebra through optional BLAS integration

---

## 2. Core Array Types and Properties

### 2.1 NDArray Class (REQ-2.1)
**Priority**: CRITICAL

The primary array type representing N-dimensional arrays.

**Requirements**:
- 2.1.1: Support for arbitrary dimensions (1D, 2D, 3D, ..., ND)
- 2.1.2: Support for dynamic dimension sizing
- 2.1.3: Efficient memory layout (row-major by default)
- 2.1.4: Type-safe element access
- 2.1.5: Automatic reference counting for memory management

### 2.2 Supported Data Types (REQ-2.2)
**Priority**: CRITICAL

**Requirements**:
- 2.2.1: `int8`, `int16`, `int32`, `int64` - Signed integers
- 2.2.2: `uint8`, `uint16`, `uint32`, `uint64` - Unsigned integers
- 2.2.3: `float32`, `float64` - Floating-point numbers
- 2.2.4: `bool` - Boolean values
- 2.2.5: `complex64`, `complex128` - Complex numbers (future)

### 2.3 Array Properties (REQ-2.3)
**Priority**: CRITICAL

**Requirements**:
- 2.3.1: `shape`: Tuple/array representing dimensions
- 2.3.2: `ndim`: Number of dimensions
- 2.3.3: `size`: Total number of elements
- 2.3.4: `dtype`: Data type of elements
- 2.3.5: `strides`: Byte-steps in each dimension
- 2.3.6: `itemsize`: Size of each element in bytes
- 2.3.7: `nbytes`: Total bytes consumed
- 2.3.8: `flags`: Memory layout flags (C_CONTIGUOUS, F_CONTIGUOUS, etc.)

---

## 3. Array Creation

### 3.1 Basic Creation Functions (REQ-3.1)
**Priority**: CRITICAL

**Requirements**:
- 3.1.1: `NDArray::array($data, $dtype = null)` - From PHP array
- 3.1.2: `NDArray::zeros($shape, $dtype = 'float64')` - Array of zeros
- 3.1.3: `NDArray::ones($shape, $dtype = 'float64')` - Array of ones
- 3.1.4: `NDArray::full($shape, $fill_value, $dtype = null)` - Filled array
- 3.1.5: `NDArray::empty($shape, $dtype = 'float64')` - Uninitialized array
- 3.1.6: `NDArray::eye($n, $m = null, $k = 0, $dtype = 'float64')` - Identity matrix

### 3.2 Range and Sequence Creation (REQ-3.2)
**Priority**: HIGH

**Requirements**:
- 3.2.1: `NDArray::arange($start, $stop = null, $step = 1, $dtype = null)` - Evenly spaced values
- 3.2.2: `NDArray::linspace($start, $stop, $num = 50, $endpoint = true, $dtype = null)` - Linear spacing
- 3.2.3: `NDArray::logspace($start, $stop, $num = 50, $base = 10.0, $dtype = null)` - Logarithmic spacing
- 3.2.4: `NDArray::geomspace($start, $stop, $num = 50, $dtype = null)` - Geometric spacing

### 3.3 Random Array Creation (REQ-3.3)
**Priority**: HIGH

**Requirements**:
- 3.3.1: `NDArray::random($shape, $dtype = 'float64')` - Uniform [0, 1)
- 3.3.2: `NDArray::randomInt($low, $high, $shape, $dtype = 'int64')` - Random integers
- 3.3.3: `NDArray::randn($shape, $dtype = 'float64')` - Standard normal distribution
- 3.3.4: `NDArray::normal($mean, $std, $shape, $dtype = 'float64')` - Normal distribution
- 3.3.5: `NDArray::uniform($low, $high, $shape, $dtype = 'float64')` - Uniform distribution

### 3.4 Like Functions (REQ-3.4)
**Priority**: MEDIUM

**Requirements**:
- 3.4.1: `$array->zerosLike()` - Zeros with same shape/dtype
- 3.4.2: `$array->onesLike()` - Ones with same shape/dtype
- 3.4.3: `$array->fullLike($value)` - Filled with same shape/dtype
- 3.4.4: `$array->emptyLike()` - Empty with same shape/dtype

---

## 4. Indexing and Slicing

### 4.1 Basic Indexing (REQ-4.1)
**Priority**: CRITICAL

**Requirements**:
- 4.1.1: Integer indexing: `$array[0]`, `$array[1, 2]`
- 4.1.2: Negative indexing: `$array[-1]`
- 4.1.3: Multi-dimensional indexing: `$array[0, 1, 2]`
- 4.1.4: Return scalar for single element, NDArray for sub-array

### 4.2 Slicing (REQ-4.2)
**Priority**: CRITICAL

**Requirements**:
- 4.2.1: Basic slice: `$array->slice([':'])` or `$array['0:5']`
- 4.2.2: Step slicing: `$array->slice(['::2'])`
- 4.2.3: Multi-dimensional slicing: `$array->slice([':', '0:5'])`
- 4.2.4: Negative indices in slices: `$array->slice(['-5:'])`
- 4.2.5: Ellipsis support: `$array->slice(['...', 0])`

### 4.3 Advanced Indexing (REQ-4.3)
**Priority**: MEDIUM

**Requirements**:
- 4.3.1: Boolean indexing: `$array->at($bool_array)`
- 4.3.2: Integer array indexing: `$array->at([0, 2, 4])`
- 4.3.3: Fancy indexing with multiple arrays
- 4.3.4: Return views where possible, copies when necessary

### 4.4 Assignment (REQ-4.4)
**Priority**: CRITICAL

**Requirements**:
- 4.4.1: Scalar assignment: `$array[0] = 5`
- 4.4.2: Array assignment: `$array['0:5'] = $other_array`
- 4.4.3: Broadcast assignment: `$array[':', 0] = 5`
- 4.4.4: Boolean mask assignment: `$array->set($mask, $value)`

---

## 5. Array Views and Copies

### 5.1 View Creation (REQ-5.1)
**Priority**: HIGH

**Requirements**:
- 5.1.1: Slicing operations return views by default
- 5.1.2: `$array->view()` - Create explicit view
- 5.1.3: Views share data with parent array
- 5.1.4: Modifications to views affect parent
- 5.1.5: Proper reference counting for view lifetime

### 5.2 Copy Operations (REQ-5.2)
**Priority**: HIGH

**Requirements**:
- 5.2.1: `$array->copy()` - Deep copy
- 5.2.2: `$array->astype($dtype)` - Copy with type conversion
- 5.2.3: Copies are independent of parent
- 5.2.4: Contiguous memory for copies

### 5.3 Memory Layout Flags (REQ-5.3)
**Priority**: MEDIUM

**Requirements**:
- 5.3.1: `isContiguous()` - Check if C-contiguous
- 5.3.2: `isFortranContiguous()` - Check if Fortran-contiguous
- 5.3.3: `asContiguousArray()` - Return C-contiguous copy
- 5.3.4: `asFortranArray()` - Return Fortran-contiguous copy

---

## 6. Shape Manipulation

### 6.1 Reshaping (REQ-6.1)
**Priority**: CRITICAL

**Requirements**:
- 6.1.1: `$array->reshape($new_shape)` - Change shape (view if possible)
- 6.1.2: Support for `-1` in shape (infer dimension)
- 6.1.3: `$array->flatten()` - 1D view/copy
- 6.1.4: `$array->ravel()` - 1D view if possible
- 6.1.5: Maintain element count constraint

### 6.2 Transposition (REQ-6.2)
**Priority**: HIGH

**Requirements**:
- 6.2.1: `$array->transpose($axes = null)` - Permute dimensions
- 6.2.2: `$array->T` property - 2D transpose shorthand
- 6.2.3: `$array->swapaxes($axis1, $axis2)` - Swap two axes
- 6.2.4: `$array->moveaxis($source, $destination)` - Move axis position

### 6.3 Dimension Manipulation (REQ-6.3)
**Priority**: HIGH

**Requirements**:
- 6.3.1: `$array->squeeze($axis = null)` - Remove single-dimensional axes
- 6.3.2: `$array->expandDims($axis)` - Add dimension
- 6.3.3: `NDArray::newaxis` constant for adding dimensions
- 6.3.4: `$array->broadcast_to($shape)` - Broadcast to shape

### 6.4 Joining and Splitting (REQ-6.4)
**Priority**: MEDIUM

**Requirements**:
- 6.4.1: `NDArray::concatenate($arrays, $axis = 0)` - Join arrays
- 6.4.2: `NDArray::stack($arrays, $axis = 0)` - Stack arrays along new axis
- 6.4.3: `NDArray::vstack($arrays)` - Vertical stack
- 6.4.4: `NDArray::hstack($arrays)` - Horizontal stack
- 6.4.5: `$array->split($indices_or_sections, $axis = 0)` - Split array
- 6.4.6: `$array->vsplit($indices)` - Vertical split
- 6.4.7: `$array->hsplit($indices)` - Horizontal split

---

## 7. Mathematical Operations

### 7.1 Element-wise Arithmetic (REQ-7.1)
**Priority**: CRITICAL

**Requirements**:
- 7.1.1: Addition: `$a + $b`, `$a->add($b)`
- 7.1.2: Subtraction: `$a - $b`, `$a->subtract($b)`
- 7.1.3: Multiplication: `$a * $b`, `$a->multiply($b)`
- 7.1.4: Division: `$a / $b`, `$a->divide($b)`
- 7.1.5: Floor division: `$a->floorDivide($b)`
- 7.1.6: Modulo: `$a % $b`, `$a->mod($b)`
- 7.1.7: Power: `$a->power($b)`, `$a ** $b`
- 7.1.8: Negation: `-$a`, `$a->negative()`

### 7.2 Element-wise Comparison (REQ-7.2)
**Priority**: HIGH

**Requirements**:
- 7.2.1: Equal: `$a == $b`, `$a->equal($b)`
- 7.2.2: Not equal: `$a != $b`, `$a->notEqual($b)`
- 7.2.3: Greater: `$a > $b`, `$a->greater($b)`
- 7.2.4: Greater or equal: `$a >= $b`, `$a->greaterEqual($b)`
- 7.2.5: Less: `$a < $b`, `$a->less($b)`
- 7.2.6: Less or equal: `$a <= $b`, `$a->lessEqual($b)`

### 7.3 Element-wise Mathematical Functions (REQ-7.3)
**Priority**: HIGH

**Requirements**:
- 7.3.1: `$array->abs()` - Absolute value
- 7.3.2: `$array->sqrt()` - Square root
- 7.3.3: `$array->exp()` - Exponential
- 7.3.4: `$array->log()` - Natural logarithm
- 7.3.5: `$array->log10()` - Base-10 logarithm
- 7.3.6: `$array->log2()` - Base-2 logarithm
- 7.3.7: `$array->sin()`, `cos()`, `tan()` - Trigonometric
- 7.3.8: `$array->arcsin()`, `arccos()`, `arctan()` - Inverse trigonometric
- 7.3.9: `$array->sinh()`, `cosh()`, `tanh()` - Hyperbolic
- 7.3.10: `$array->floor()`, `ceil()`, `round()` - Rounding
- 7.3.11: `$array->sign()` - Sign function
- 7.3.12: `$array->clip($min, $max)` - Clip values

### 7.4 Broadcasting (REQ-7.4)
**Priority**: CRITICAL

**Requirements**:
- 7.4.1: Automatic broadcasting for binary operations
- 7.4.2: Broadcasting rules compatible with NumPy
- 7.4.3: Support for scalar broadcasting
- 7.4.4: Error on incompatible shapes
- 7.4.5: `$array->broadcastTo($shape)` for explicit broadcasting

---

## 8. Reduction Operations

### 8.1 Statistical Reductions (REQ-8.1)
**Priority**: HIGH

**Requirements**:
- 8.1.1: `$array->sum($axis = null, $keepdims = false)` - Sum
- 8.1.2: `$array->mean($axis = null, $keepdims = false)` - Mean
- 8.1.3: `$array->std($axis = null, $ddof = 0, $keepdims = false)` - Standard deviation
- 8.1.4: `$array->var($axis = null, $ddof = 0, $keepdims = false)` - Variance
- 8.1.5: `$array->min($axis = null, $keepdims = false)` - Minimum
- 8.1.6: `$array->max($axis = null, $keepdims = false)` - Maximum
- 8.1.7: `$array->median($axis = null, $keepdims = false)` - Median
- 8.1.8: `$array->quantile($q, $axis = null, $keepdims = false)` - Quantile

### 8.2 Logical Reductions (REQ-8.2)
**Priority**: MEDIUM

**Requirements**:
- 8.2.1: `$array->all($axis = null, $keepdims = false)` - All true
- 8.2.2: `$array->any($axis = null, $keepdims = false)` - Any true

### 8.3 Index-based Operations (REQ-8.3)
**Priority**: MEDIUM

**Requirements**:
- 8.3.1: `$array->argmin($axis = null)` - Index of minimum
- 8.3.2: `$array->argmax($axis = null)` - Index of maximum
- 8.3.3: `$array->argsort($axis = -1)` - Indices that would sort
- 8.3.4: `$array->nonzero()` - Indices of non-zero elements

### 8.4 Cumulative Operations (REQ-8.4)
**Priority**: MEDIUM

**Requirements**:
- 8.4.1: `$array->cumsum($axis = null)` - Cumulative sum
- 8.4.2: `$array->cumprod($axis = null)` - Cumulative product

---

## 9. Linear Algebra

### 9.1 Matrix Operations (REQ-9.1)
**Priority**: HIGH

**Requirements**:
- 9.1.1: `$a->dot($b)` - Dot product / matrix multiplication
- 9.1.2: `$a->matmul($b)` - Matrix multiplication (@ operator support)
- 9.1.3: `$array->trace()` - Sum of diagonal elements
- 9.1.4: `$array->diagonal($offset = 0)` - Extract diagonal

### 9.2 Matrix Decompositions (REQ-9.2)
**Priority**: MEDIUM (requires BLAS)

**Requirements**:
- 9.2.1: `$array->svd($full_matrices = true)` - Singular value decomposition
- 9.2.2: `$array->qr()` - QR decomposition
- 9.2.3: `$array->cholesky()` - Cholesky decomposition
- 9.2.4: `$array->eig()` - Eigenvalue decomposition
- 9.2.5: `$array->lu()` - LU decomposition

### 9.3 Matrix Properties (REQ-9.3)
**Priority**: MEDIUM (requires BLAS)

**Requirements**:
- 9.3.1: `$array->det()` - Determinant
- 9.3.2: `$array->inv()` - Matrix inverse
- 9.3.3: `$array->pinv()` - Pseudo-inverse
- 9.3.4: `$array->norm($ord = null, $axis = null)` - Matrix/vector norm
- 9.3.5: `$array->cond($p = null)` - Condition number
- 9.3.6: `$array->rank()` - Matrix rank

### 9.4 Solving Linear Systems (REQ-9.4)
**Priority**: MEDIUM (requires BLAS)

**Requirements**:
- 9.4.1: `NDArray::solve($a, $b)` - Solve linear equations
- 9.4.2: `NDArray::lstsq($a, $b)` - Least squares solution

---

## 10. Iteration and Application

### 10.1 Iterator Support (REQ-10.1)
**Priority**: HIGH

**Requirements**:
- 10.1.1: Implement PHP `Iterator` interface for 1D iteration
- 10.1.2: `$array->nditer()` - N-dimensional iterator
- 10.1.3: Iterate over first axis by default
- 10.1.4: Support for `foreach` loops
- 10.1.5: Efficient iteration without copying

### 10.2 Functional Operations (REQ-10.2)
**Priority**: MEDIUM

**Requirements**:
- 10.2.1: `$array->apply($callable)` - Apply function element-wise
- 10.2.2: `$array->map($callable)` - Map function (returns new array)
- 10.2.3: `$array->reduce($callable, $initial = null)` - Reduce operation

---

## 11. Type Conversion and I/O

### 11.1 Type Conversion (REQ-11.1)
**Priority**: HIGH

**Requirements**:
- 11.1.1: `$array->astype($dtype)` - Convert data type
- 11.1.2: `$array->toArray()` - Convert to PHP array
- 11.1.3: `$array->toList()` - Convert to PHP list (alias for toArray)
- 11.1.4: `$array->toScalar()` - Convert 0-d array to scalar
- 11.1.5: Automatic type inference from PHP arrays

### 11.2 Serialization (REQ-11.2)
**Priority**: MEDIUM

**Requirements**:
- 11.2.1: `$array->save($filename)` - Save to binary file (.npy format)
- 11.2.2: `NDArray::load($filename)` - Load from binary file
- 11.2.3: `$array->toString()` - String representation
- 11.2.4: `$array->toJson()` - JSON serialization
- 11.2.5: Support for `serialize()` / `unserialize()`

---

## 12. BLAS Integration

### 12.1 Optional BLAS Backend (REQ-12.1)
**Priority**: MEDIUM

**Requirements**:
- 12.1.1: Configurable at runtime (not compile-time requirement)
- 12.1.2: Graceful fallback to pure Rust implementation
- 12.1.3: Support for OpenBLAS, Intel MKL, Apple Accelerate
- 12.1.4: Configuration via: `NDArray::enableBLAS($backend = 'openblas')`
- 12.1.5: Query BLAS status: `NDArray::blasEnabled()`

### 12.2 BLAS-Accelerated Operations (REQ-12.2)
**Priority**: MEDIUM

**Requirements**:
- 12.2.1: Matrix multiplication (GEMM)
- 12.2.2: Dot products (DOT)
- 12.2.3: Matrix-vector operations (GEMV)
- 12.2.4: Linear system solving
- 12.2.5: Eigenvalue/SVD decompositions

---

## 13. Memory Management

### 13.1 Memory Allocation (REQ-13.1)
**Priority**: CRITICAL

**Requirements**:
- 13.1.1: Rust handles all memory allocation
- 13.1.2: PHP holds opaque pointer to Rust data
- 13.1.3: Reference counting for shared data (views)
- 13.1.4: Automatic cleanup on PHP object destruction
- 13.1.5: Manual cleanup via `$array->dispose()` if needed

### 13.2 Memory Safety (REQ-13.2)
**Priority**: CRITICAL

**Requirements**:
- 13.2.1: No use-after-free errors
- 13.2.2: No double-free errors
- 13.2.3: No memory leaks
- 13.2.4: Thread-safe reference counting
- 13.2.5: Proper exception handling across FFI boundary

### 13.3 Memory Optimization (REQ-13.3)
**Priority**: MEDIUM

**Requirements**:
- 13.3.1: Copy-on-write where applicable
- 13.3.2: In-place operations where possible
- 13.3.3: View creation instead of copying
- 13.3.4: Memory pooling for small allocations (optional)

---

## 14. Error Handling

### 14.1 Exception Types (REQ-14.1)
**Priority**: HIGH

**Requirements**:
- 14.1.1: `NDArrayException` - Base exception class
- 14.1.2: `ShapeException` - Shape mismatch errors
- 14.1.3: `DTypeException` - Data type errors
- 14.1.4: `IndexException` - Indexing errors
- 14.1.5: `BroadcastException` - Broadcasting errors
- 14.1.6: `LinearAlgebraException` - Linear algebra errors

### 14.2 Error Propagation (REQ-14.2)
**Priority**: HIGH

**Requirements**:
- 14.2.1: Rust panics converted to PHP exceptions
- 14.2.2: Clear error messages with context
- 14.2.3: Stack traces preserved where possible
- 14.2.4: No silent failures

---

## 15. Performance Requirements

### 15.1 Benchmarks (REQ-15.1)
**Priority**: HIGH

**Requirements**:
- 15.1.1: Matrix multiplication: 10x faster than pure PHP
- 15.1.2: Element-wise operations: 5-10x faster than pure PHP
- 15.1.3: Reductions: 10x faster than pure PHP
- 15.1.4: FFI overhead < 5% for large arrays (>1000 elements)
- 15.1.5: Memory usage comparable to NumPy

### 15.2 Optimization Strategies (REQ-15.2)
**Priority**: HIGH

**Requirements**:
- 15.2.1: SIMD vectorization in Rust
- 15.2.2: Multi-threading for large operations (configurable)
- 15.2.3: Cache-friendly memory access patterns
- 15.2.4: Minimize FFI crossings
- 15.2.5: Batch operations in Rust layer

---

## 16. Testing and Quality

### 16.1 Test Coverage (REQ-16.1)
**Priority**: HIGH

**Requirements**:
- 16.1.1: Unit tests for all public methods
- 16.1.2: Integration tests for complex workflows
- 16.1.3: Property-based testing for mathematical correctness
- 16.1.4: Benchmark suite for performance regression
- 16.1.5: Memory leak detection tests

### 16.2 Compatibility Testing (REQ-16.2)
**Priority**: MEDIUM

**Requirements**:
- 16.2.1: NumPy compatibility tests (behavior comparison)
- 16.2.2: Cross-platform testing (Linux, macOS, Windows)
- 16.2.3: PHP version testing (8.1, 8.2, 8.3+)
- 16.2.4: Different BLAS backends

---

## 17. Documentation

### 17.1 API Documentation (REQ-17.1)
**Priority**: HIGH

**Requirements**:
- 17.1.1: PHPDoc annotations for all public APIs
- 17.1.2: User guide with examples
- 17.1.3: Migration guide from NumPy
- 17.1.4: Performance tuning guide
- 17.1.5: API reference (auto-generated)

### 17.2 Examples (REQ-17.2)
**Priority**: MEDIUM

**Requirements**:
- 17.2.1: Quick start tutorial
- 17.2.2: Common operations cookbook
- 17.2.3: Performance comparison examples
- 17.2.4: Advanced usage patterns
- 17.2.5: Integration examples (with Laravel, Symfony, etc.)

---

## 18. Installation and Distribution

### 18.1 Package Distribution (REQ-18.1)
**Priority**: HIGH

**Requirements**:
- 18.1.1: Composer package
- 18.1.2: Pre-built binaries for common platforms
- 18.1.3: Build instructions for source compilation
- 18.1.4: Docker images for development

### 18.2 Platform Support (REQ-18.2)
**Priority**: HIGH

**Requirements**:
- 18.2.1: Linux (x86_64, ARM64)
- 18.2.2: macOS (x86_64, Apple Silicon)
- 18.2.3: Windows (x86_64)
- 18.2.4: PHP 8.1+ with FFI extension enabled

---

## 19. Future Enhancements (Optional)

### 19.1 Phase 2 Features (REQ-19.1)
**Priority**: LOW

- 19.1.1: GPU acceleration via CUDA/ROCm
- 19.1.2: Sparse matrix support
- 19.1.3: Complex number support
- 19.1.4: FFT operations
- 19.1.5: Image processing utilities
- 19.1.6: Pandas-like DataFrame functionality

### 19.2 Integration Features (REQ-19.2)
**Priority**: LOW

- 19.2.1: Laravel facade
- 19.2.2: Symfony bundle
- 19.2.3: CSV/Excel import utilities
- 19.2.4: Database query result conversion

---

## 20. Success Criteria

### 20.1 Functional Criteria
- All CRITICAL priority requirements implemented
- 90%+ HIGH priority requirements implemented
- Passes all NumPy compatibility tests
- Zero critical bugs in core operations

### 20.2 Performance Criteria
- Meets or exceeds benchmark targets (REQ-15.1)
- Memory usage within 10% of NumPy
- FFI overhead negligible for large arrays

### 20.3 Quality Criteria
- >90% code coverage
- No memory leaks detected
- Zero use-after-free or double-free errors
- Clean Valgrind reports

### 20.4 Usability Criteria
- Complete API documentation
- 10+ example projects
- Migration guide from NumPy
- Active community engagement

---

## Appendix A: NumPy Compatibility Matrix

| Feature | NumPy Equivalent | Priority | Status |
|---------|------------------|----------|--------|
| Array creation | `np.array()` | CRITICAL | Planned |
| Slicing | `arr[0:5]` | CRITICAL | Planned |
| Broadcasting | Automatic | CRITICAL | Planned |
| Matrix mult | `@` operator | HIGH | Planned |
| Reductions | `sum()`, `mean()` | HIGH | Planned |
| SVD | `np.linalg.svd()` | MEDIUM | Planned |
| FFT | `np.fft.fft()` | LOW | Future |

---

## Appendix B: Glossary

- **FFI**: Foreign Function Interface - allows PHP to call Rust code
- **BLAS**: Basic Linear Algebra Subprograms - optimized linear algebra routines
- **View**: Array that shares data with another array
- **Broadcasting**: Automatic shape alignment for operations
- **Contiguous**: Memory layout where elements are stored sequentially
- **Stride**: Number of bytes to step in each dimension

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-09 | Initial specification |

