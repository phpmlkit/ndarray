## Purpose

This file defines how AI coding agents must work inside this repository.

The project implements a high-performance, NumPy-inspired NDArray library for PHP, backed by Rust via FFI.
Correctness, memory safety, and API parity matter more than speed of implementation.

Agents must treat this as a systems project, not a typical PHP library.

⸻

## Canonical Project Documents

Agents must consult these files before making architectural or behavioral decisions:
- SPEC.md
  → Defines what the library must do, feature priorities, API guarantees, and success criteria.
  → Always check this when:
	- adding or modifying public APIs
	- implementing new operations
	- deciding between view vs copy behavior
	- handling edge cases (broadcasting, slicing, dtype rules)

If there is a conflict between intuition and these documents, the documents win.

⸻

## External References

- Rust ndarray crate documentation: https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html
  → The core Rust library this project wraps
  → Consult for understanding ArrayBase, views, slicing, and memory layout

⸻

## High-Level Agent Responsibilities

Agents working on this repo must:

1. Preserve NumPy-compatible semantics where specified.
2. Minimize FFI crossings and avoid chatty PHP↔Rust calls.
3. Treat memory safety as non-negotiable.
4. Prefer views over copies unless explicitly unsafe or forbidden.
5. Keep the PHP API idiomatic, even when backed by complex Rust logic.
6. Avoid introducing behavior that diverges subtly from the spec.

⸻

## What This Project Is Not

Agents must not assume:

- This is a pure PHP numerical library
- PHP should manage memory for NDArray buffers
- Copying data is cheap
- Rust panics are acceptable
- Convenience > correctness

This is a low-level numerical system exposed through a high-level PHP API.

⸻

## PHP ↔ Rust FFI Rules (Critical)

Agents must follow these rules exactly:

### Ownership & Memory

- Rust owns all NDArray memory
- PHP holds opaque pointers only
- PHP must never:
- free Rust memory manually
- clone raw buffers
- assume pointer validity beyond documented lifetimes

### Views

- Slicing returns views by default
- Views share memory
- Reference counting must be updated correctly in Rust
- Never fake a view in PHP by copying

### Errors

- Rust panics must be converted to PHP exceptions
- Silent failures are forbidden
- Error messages should include:
	- operation name
	- shape/dtype context where possible

### Binding Maintenance (Type Intelligence)

- Whenever a new FFI function is exported in Rust (`extern "C"`), you **MUST** update `src/FFI/Bindings.php`.
- Add the method signature to the `Bindings` interface to match the C header.
- This ensures IDEs (VS Code, PHPStorm) and static analyzers provide correct autocomplete and type checking for `$ffi->method_name(...)` calls.
- Failure to do this degrades the developer experience and makes the codebase harder to maintain.

⸻

## API Design Rules

When implementing or modifying APIs:

- Public PHP APIs must:
- Match SPEC.md naming and behavior
- Be discoverable and predictable
- Avoid surprising PHP-isms that break NumPy mental models
- Internal helper methods may exist, but:
- They must not leak into public API
- They must not bypass validation logic
- Operator overloading (+, -, *, @, etc.):
- Must obey broadcasting rules
- Must fail loudly on invalid shapes

⸻

## Shape, Broadcasting, and DType Discipline

Agents must be extremely strict about:

- Shape validation before execution
- Broadcasting compatibility checks
- DType inference rules

Never:
- Auto-reshape silently
- Truncate data
- Coerce dtypes without explicit rules

If behavior is unclear:
1. Check SPEC.md
2. Compare with NumPy
3. Choose the stricter, safer behavior

⸻

## Performance Expectations

Agents should optimize deliberately, not prematurely.

Preferred optimizations:

- Batch operations in Rust
- Use contiguous memory when required
- Defer copies unless unavoidable
- Prefer SIMD and parallelism in Rust, not PHP loops

Avoid:

- PHP-side element loops
- Per-element FFI calls
- Micro-optimizations that complicate correctness

If an optimization changes semantics, do not apply it.

⸻

## Testing Obligations

When adding or changing behavior, agents must:

- Add or update PHP unit tests
- Add Rust tests if logic lives in Rust
- Consider:
- shape edge cases
- zero-dim arrays
- broadcasting failures
- view vs copy behavior
- memory lifetime scenarios

Performance-sensitive changes should include benchmarks when feasible.

⸻

## Common Mistakes to Avoid

Agents must not:

- Implement NumPy-like APIs that behave almost the same
- Return copies where views are required
- Assume PHP GC will handle Rust resources
- Introduce hidden global state in FFI
- Hard-require BLAS
- Break cross-platform builds
- Add features not described in SPEC.md without clear justification

⸻

## When in Doubt

If uncertain about:

- Semantics → Check SPEC.md
- Performance → Favor correctness
- Memory → Assume danger and verify
- API design → Favor explicitness

When tradeoffs exist, document the decision clearly in code comments.

⸻

## Code Documentation Rules

Agents must follow these rules for code comments and docblocks:

- Never reference SPEC.md, or any specification documents in code
- Never include requirement IDs (e.g., "REQ-2.2") in source files
- Write docblocks that describe what the code does, not what document it implements
- Comments should be self-contained and understandable without external references
- Use proper PHPDoc for PHP and rustdoc for Rust
- Keep comments concise and implementation-focused

⸻

## Rust View Extraction Patterns

When implementing operations that work with strided array views in Rust:

### Type-Specific Extraction
Use `extract_view_f64`, `extract_view_i64`, etc. from `view_helpers.rs` instead of generic conversion. This avoids unnecessary data copying for native types.

```rust
// Good: Try native type first
if let Some(view) = extract_view_f64(wrapper, offset, shape, strides) {
    return view.sum();  // Zero-copy for f64
}
if let Some(view) = extract_view_i64(wrapper, offset, shape, strides) {
    return view.iter().map(|&x| x as f64).sum();  // Exact integer sum
}
```

### Use Native ndarray Methods
Prefer ndarray's built-in methods over manual implementations:
- `view.sum()` not manual iteration
- `view.mean()` not `sum / n`
- `view.var(ddof)` not manual variance calculation
- `view.product()` not manual fold
- `view.product_axis(axis)` not manual fold_axis

### Float Operations for Integers
For operations requiring float arithmetic (mean, var, std):
- Extract integer view as native type
- Use `.mapv(|x| x as f64)` to convert
- Then call ndarray's float methods

### Mixed Type Operations
Use `extract_view_as_*` functions to handle operations between different types:

```rust
use crate::core::view_helpers::extract_view_as_i32;

// This extracts any input type as i32 (with silent truncation)
if let Some(view) = extract_view_as_i32(wrapper, offset, shape, strides) {
    // view is ArrayD<i32> regardless of input type
}
```

Available functions:
- `extract_view_as_f64`, `extract_view_as_f32` - for float operations
- `extract_view_as_i64`, `extract_view_as_i32`, etc. - for integer operations
- `extract_view_as_u64`, `extract_view_as_u32`, etc. - for unsigned operations

These functions:
- Try native extraction first (zero-copy for matching types)
- Convert from other types via `mapv(|x| x as TargetType)`
- Silently truncate values that don't fit (Rust's `as` behavior)

### File Organization
- Keep operation-specific logic in its own file (e.g., `sum_scalar.rs`)
- `view_helpers.rs` should only contain generic extraction utilities
- Never put operation logic in view_helpers

## Arithmetic Operations Patterns

### Type-Preserving Operations
Use `math_helpers.rs` for type-specific operations:

**Generic ops** (work on all types like abs, pow2):
```rust
use crate::core::math_helpers::unary_op_generic;

let result = unary_op_generic(
    wrapper, offset, shape, strides,
    |x: f64| x.abs(),
    |x: f32| x.abs(),
    |x: i64| x.abs(),
    // ... etc for all types
)?;
```

**Float-only ops** (floor, sqrt, sin, etc.):
```rust
use crate::core::math_helpers::unary_op_float_only;

let result = unary_op_float_only(
    wrapper, offset, shape, strides,
    |x: f64| x.floor(),
    |x: f32| x.floor(),
    "floor",
)?;
```

**Binary ops** with type promotion:
```rust
use crate::core::math_helpers::{
    binary_op_f64, binary_op_i64, promote_dtypes,
};

let out_dtype = promote_dtypes(a.dtype, b.dtype)?;
let result = match out_dtype {
    DType::Float64 => binary_op_f64(a, ..., b, ..., |x, y| x + y),
    DType::Int64 => binary_op_i64(a, ..., b, ..., |x, y| x + y),
    // ... etc
};
```

Binary helpers use `extract_view_as_*` internally, so they automatically handle mixed types:
- `binary_op_i32(Int32 array, Int16 array)` converts Int16 to Int32 automatically
- No need for separate "mixed" helper functions

**Scalar ops** preserving input type:
```rust
use crate::core::math_helpers::scalar_op_generic;

let result = scalar_op_generic(
    wrapper, offset, shape, strides, scalar,
    |a: f64, b: f64| a + b,
    |a: f32, b: f32| a + b,
    |a: i64, b: i64| a + b,
    // ... etc
)?;
```

### Key Principles
- Never convert through f64 unless necessary
- Use closures with explicit type parameters: `|x: f64| x.abs()` not `|x| x.abs()`
- Binary operations promote to higher precision dtype
- Float-only operations error on integer inputs
- Bool operations error (use specific Bool helpers if needed)

## Final Principle

This project aims to be: A serious numerical computing foundation for PHP, not a convenience wrapper.

Agents should act accordingly.