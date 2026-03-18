//! NDArray PHP - Rust library
//!
//! This crate provides the core types, macros, helpers, and optional FFI layer for NDArray PHP.
//!
//! When built with the `ffi` feature, it exposes C-compatible functions for PHP FFI.
//! When used as a dependency, only types, macros, and helpers are included.

#[macro_use]
pub mod macros;

pub mod helpers;
pub mod types;

#[cfg(feature = "ffi")]
pub mod ffi;

pub use helpers::{
    set_last_error, write_output_metadata, ERR_ALLOC, ERR_DTYPE, ERR_GENERIC, ERR_INDEX, ERR_MATH,
    ERR_PANIC, ERR_SHAPE, SUCCESS,
};
pub use macros::*;
pub use types::{ArrayData, ArrayMetadata, DType, DTypeError, NDArrayWrapper, NdArrayHandle};

#[cfg(feature = "ffi")]
pub use ffi::*;
