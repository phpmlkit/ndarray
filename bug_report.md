# NDArray View Strides Bug — Investigation Prompt

## What's Happening

NDArray has a zero-copy view system where operations like `transpose()`, `slice()`, `swapaxes()`, `squeeze()`, `insertaxis()`, `reshape()` (on contiguous arrays), `ravel()` (on contiguous arrays), and partial `get()` return PHP-side views. A view reuses the **same Rust data handle** and only adjusts PHP-side metadata — shape, strides, and offset. No Rust allocation occurs.

These views work correctly for PHP-side access (`getAt()`, `get()`). The `logicalFlatToStorageIndex` method correctly maps logical flat indices to storage flat indices using the view's shape, strides, and offset.

**The bug:** When a view (with non-standard strides) is passed to ANY Rust operation (`clamp`, `log`, `log10`, `exp`, `add`, `multiply`, `sqrt`, `abs`, `round`, etc.), the Rust function extracts an `ArrayViewD` from the metadata and operates on it. The Rust operations produce output in **raw data buffer order** (original memory layout), ignoring the view's strides. PHP then reads the output using the logical shape — but the data at each position is wrong because it was populated in the wrong order.

## Reproduction

Take a contiguous `[3001, 80]` NDArray. Call `.transpose()` to get a `[80, 3001]` view with strides `[1, 80]`. Call `.clamp(min, max)` or `.log10()` on it. `getAt(1)` returns the value from raw position 1 (which is `original[0, 1]` = `mel1_frame0`) instead of logical position 1 (which should be `view[0, 1]` = `original[1, 0]` = `mel0_frame1`).

The view's `get(0, 1)` correctly returns `mel0_frame1`. But after any Rust operation, `getAt(1)` returns `mel1_frame0` — the raw-data-order value.

## Affected View Operations

All of these return PHP-side views (same Rust handle, adjusted metadata):

| Always a View | Conditionally a View (when contiguous) |
|--------------|---------------------------------------|
| `squeeze()` | `reshape()` |
| `insertaxis()` / `expandDims()` | `transpose()` |
| `swapaxes()` | `ravel()` |
| `mergeaxes()` | |
| `slice()` | |
| `get()` with partial indices | |
| `split()` / `vsplit()` / `hsplit()`* | |

> *split() calls Rust for metadata computation but returns views sharing the same handle.

## Affected Rust Operations

All of these call `extract_view_f64` (or equivalent), get an `ArrayViewD`, operate on it, and return wrong results when the view has non-standard strides:

- `clamp()` — `rust/src/ffi/misc/clamp.rs`
- `log()`, `log10()`, `exp()`, `sqrt()` — `rust/src/ffi/math/`
- `add_scalar`, `multiply_scalar` — `rust/src/ffi/arithmetic/`
- `abs()`, `round()`, `ceil()`, `floor()` — `rust/src/ffi/math/`
- `neg()` — `rust/src/ffi/arithmetic/`
- Basically every function that calls `extract_view_*` and then applies an ndarray method

## Root Cause Hypothesis

The view extraction in Rust (`extract_view_f64` in `rust/src/macros/view.rs` line 24) correctly creates an `ArrayViewD` using `from_shape_ptr(shape, strides, ptr)`. The view itself is logically correct — `view[0, 1]` reads from `ptr[0*stride0 + 1*stride1]` which for transposed strides `[1, 80]` = `ptr[80]`, the correct element.

But ndarray `0.17.2`'s methods like `clamp()`, `log10()`, `mapv()`, etc. appear to NOT respect the view's strides when producing output arrays. They iterate over the raw data buffer in its original contiguous order, ignoring the strides.

## Verification Already Done

Empirical tests in the TransformersPHP test suite confirmed:

1. `getAt()` on a transposed view → correct (PHP-side, respects strides)
2. `get(0,1)` on a transposed view → correct (PHP-side, respects strides)
3. `clamp()` on a transposed view → wrong (Rust, ignores strides)
4. `log10()` on a transposed view → wrong (Rust, ignores strides)
5. `add()` on a transposed view → wrong (Rust, ignores strides)
6. `multiply()` on a transposed view → wrong (Rust, ignores strides)
7. `clamp()` on a non-transposed (contiguous) array → correct

## Where to Look

1. **`rust/src/macros/view.rs`** — `define_extract_view` macro (line 5). Creates `ArrayViewD` with custom strides. Verify the view is constructed correctly.

2. **`rust/src/ffi/misc/clamp.rs`** — The clamp FFI function. Modify to call `view.to_owned().clamp()` instead of `view.clamp()` to materialize the view before operating.

3. **`rust/src/ffi/math/`** — All math operations (log, log10, exp, sqrt, etc.). Same pattern — extract view, call ndarray method. These are also affected.

4. **`rust/src/ffi/arithmetic/`** — Scalar add/multiply/divide operations. Also affected.

## Potential Fix Paths

**Path A (per-operation):** In each Rust FFI function, call `view.to_owned()` before the operation. This materializes the view into a contiguous copy, then the operation works correctly. Downside: modifies every operation file.

**Path B (view extraction):** In the `define_extract_view` macro or similar, return `ArrayD` (owned, contiguous) instead of `ArrayViewD`. This forces materialization for all operations. Downside: loses zero-copy benefits for operations that DO respect strides.

**Path C (PHP side):** Make `transpose()` and other view-returning methods always go through Rust to materialize the transposition. Downside: one extra FFI call per transpose.

**Path D (upstream):** Determine if this is a bug in ndarray 0.17.2's `mapv`/`clamp`/`log10` implementations. If so, upgrade ndarray or patch it. The `mapv` method should iterate in logical order using the view's strides.

## Test Cases to Confirm

```php
use PhpMlKit\NDArray\NDArray;
use PHPUnit\Event\Runtime\PHP;

require_once 'vendor/autoload.php';

// 1. Create a [3, 2] NDArray: [[1,2], [3,4], [5,6]]
$arr = NDArray::array([[1,2],[3,4],[5,6]]);

// 2. Transpose → [2, 3]: [[1,3,5], [2,4,6]]
$tr = $arr->transpose();

// 3. Clamp -> [[1,2,3], [4,5,6]] instead of [[1,3,5], [2,4,6]]
$clamped = $tr->clamp(-INF, INF);

echo "Transpose: $tr" . PHP_EOL;
echo "Clamped: $clamped" . PHP_EOL;
```
