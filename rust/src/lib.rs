//! NDArray PHP - Rust FFI Library
//!
//! This crate provides the Rust backend for the NDArray PHP library.
//! It exposes C-compatible functions for PHP FFI.

pub mod dtype;

pub use dtype::{DType, DTypeError};
