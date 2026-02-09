//! NDArray PHP - Rust FFI Library
//!
//! This crate provides the Rust backend for the NDArray PHP library.
//! It exposes C-compatible functions for PHP FFI.

// Macros must be declared before modules that use them
#[macro_use]
pub mod core;

pub mod dtype;
pub mod ffi;

// Re-export public types
pub use core::{ArrayData, NDArrayWrapper};
pub use dtype::{DType, DTypeError};
pub use ffi::NdArrayHandle;
