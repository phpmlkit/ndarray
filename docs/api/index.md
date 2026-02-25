# API Reference

Complete reference for all NDArray classes and methods.

## Core

- [NDArray Class](/api/ndarray-class) - Main array class with properties
- [Array Creation](/api/array-creation) - Array creation methods

## Array Manipulation

- [Array Manipulation](/api/array-manipulation) - Reshape, transpose, stacking, splitting
- [Indexing Routines](/api/indexing-routines) - Access and modify elements
- [Array Import and Export](/api/array-import-export) - Converting to/from other formats

## Mathematical Operations

- [Mathematical Functions](/api/mathematical-functions) - Arithmetic, trigonometry, exponentials, rounding
- [Logic Functions](/api/logic-functions) - Comparison operations (eq, gt, lt, etc.)
- [Bitwise Operations](/api/bitwise-operations) - Bitwise AND, OR, XOR, shifts
- [Statistics](/api/statistics) - Sum, mean, variance, min, max
- [Sorting, Searching, and Counting](/api/sorting-searching) - Sort, argsort, argmin, argmax
- [Linear Algebra](/api/linear-algebra) - Matrix operations

## Other

- [Exceptions](/api/exceptions) - Error handling

## Quick Reference

### Most Common Methods

**Creation:**
- `NDArray::array()` - From PHP array
- `NDArray::zeros()` - Array of zeros
- `NDArray::ones()` - Array of ones
- `NDArray::random()` - Random values

**Properties:**
- `shape()` - Get dimensions
- `dtype()` - Get data type
- `size()` - Total elements

**Math:**
- `add()`, `subtract()`, `multiply()`, `divide()`
- `sum()`, `mean()`, `std()`, `min()`, `max()`

**Shape:**
- `reshape()`, `transpose()`, `flatten()`
- `slice()` - Extract subarrays

## Next Steps

- [Getting Started](/guide/getting-started/installation)
- [Quick Start](/guide/getting-started/quick-start)