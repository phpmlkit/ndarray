# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v1.0.0 - 2026-03-01

### Added

- **Initial release of NDArray PHP** - High-performance N-dimensional arrays for PHP powered by Rust via FFI

#### Array Creation

- `NDArray::array()` - Create from PHP nested arrays
- `NDArray::zeros()`, `NDArray::ones()`, `NDArray::full()` - Fill value factories
- `NDArray::empty()` - Create empty arrays for preallocation
- `NDArray::fromBuffer()` - Import from external FFI buffers (e.g., audio/image libraries)
- `NDArray::zerosLike()`, `NDArray::onesLike()`, `NDArray::fullLike()` - Shape-based factories
- `NDArray::eye()` - Identity matrix creation
- `NDArray::arange()`, `NDArray::linspace()`, `NDArray::logspace()`, `NDArray::geomspace()` - Sequence generation
- `NDArray::random()`, `NDArray::randomInt()`, `NDArray::randn()` - Random number generation
- `NDArray::normal()`, `NDArray::uniform()` - Statistical distributions

#### Mathematical Operations

- Element-wise arithmetic: `add()`, `subtract()`, `multiply()`, `divide()`, `rem()`, `mod()`
- Unary operations: `abs()`, `negative()`, `sqrt()`, `exp()`, `log()`, `ln()`
- Trigonometric: `sin()`, `cos()`, `tan()`, `sinh()`, `cosh()`, `tanh()`, `asin()`, `acos()`, `atan()`
- Advanced math: `cbrt()`, `ceil()`, `exp2()`, `floor()`, `log2()`, `log10()`, `pow2()`, `round()`, `signum()`, `recip()`, `ln1p()`, `toDegrees()`, `toRadians()`
- Power operations: `powi()`, `powf()`, `hypot()`
- Clamping: `clamp()`, `clip()`
- Neural network helpers: `sigmoid()`, `softmax()`

#### Array Manipulation

- Shape operations: `reshape()`, `transpose()`, `swapaxes()`, `permute()`, `merge()`
- Dimension operations: `flatten()`, `ravel()`, `squeeze()`, `expandDims()`, `insert()`
- Axis operations: `flip()`, `pad()`
- Repetition: `tile()`, `repeat()`

#### Slicing and Indexing

- `slice()` - Advanced slicing with step support
- ArrayAccess: `$arr[0]`, `$arr['0:5']`, `$arr['0,1']`
- Indexing methods: `get()`, `set()`, `getAt()`, `setAt()`
- Advanced indexing: `take()`, `takeAlongAxis()`, `put()`, `putAlongAxis()`, `scatterAdd()`

#### Reductions and Statistics

- Basic: `sum()`, `mean()`, `product()`, `min()`, `max()`
- Variance/Std: `var()`, `std()` with ddof support
- Argmin/Argmax: `argmin()`, `argmax()`
- Cumulative: `cumsum()`, `cumprod()`
- Sorting: `sort()`, `argsort()`
- Search: `topk()`, `bincount()`

#### Linear Algebra

- `dot()` - Dot product
- `matmul()` - Matrix multiplication
- `diagonal()` - Extract diagonal
- `trace()` - Matrix trace
- `norm()` - Vector/matrix norms (L1, L2, etc.)

#### Stacking and Splitting

- `concatenate()`, `stack()`, `vstack()`, `hstack()` - Joining arrays
- `split()`, `vsplit()`, `hsplit()` - Splitting arrays

#### Comparison and Logical Operations

- Comparison: `eq()`, `ne()`, `gt()`, `gte()`, `lt()`, `lte()`
- Logical: `and()`, `or()`, `not()`, `xor()`

#### Bitwise Operations

- `bitand()`, `bitor()`, `bitxor()`, `leftShift()`, `rightShift()`

#### Type Conversion and Export

- `toArray()` - Convert to nested PHP array
- `toScalar()` - Extract 0D array value
- `toBytes()` - Raw binary export
- `intoBuffer()` - Export to FFI C buffer with offset/length support
- `copy()` - Deep copy
- `astype()` - Type casting

#### Views and Memory Management

- Zero-copy views for slicing, indexing, transpose, reshape (when contiguous)
- Automatic memory management via Rust allocation
- View chain tracking for proper lifecycle management
- `isView()`, `isContiguous()` introspection

#### Utilities

- `flat()` - Flat iterator for memory-efficient iteration
- Printing: configurable display options, `__toString()`
- `where()` - Conditional array selection

### Technical Highlights

- **Zero-copy views** - Slicing, indexing, and many operations return views sharing memory with parent arrays
- **Rust backend** - Core computations performed by optimized Rust code with SIMD support
- **FFI architecture** - Minimal PHP/Rust boundary crossing, batch operations when possible
- **Memory safety** - Rust owns all data, PHP holds opaque pointers, automatic cleanup
- **Type safety** - Full type hints, 11 DType variants (Int8-64, UInt8-64, Float32/64, Bool)
- **Comprehensive testing** - 927 tests, 62,353 assertions covering all functionality
- **Extensive documentation** - API reference, guides, and examples

**Full Changelog**: https://github.com/phpmlkit/ndarray/commits/1.0.0
