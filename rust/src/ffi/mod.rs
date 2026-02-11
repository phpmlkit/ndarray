//! FFI module for PHP integration.
//!
//! This module exposes Rust functionality to PHP via C-compatible functions.

pub mod arithmetic;
pub mod array;
pub mod generators;
pub mod indexing;
mod types;

pub use types::NdArrayHandle;

// Re-export all FFI functions for lib.rs
pub use arithmetic::*;
pub use array::*;
pub use generators::*;
pub use indexing::*;
