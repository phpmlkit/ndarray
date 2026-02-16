//! FFI module for PHP integration.
//!
//! This module exposes Rust functionality to PHP via C-compatible functions.

pub mod arithmetic;
pub mod array;
pub mod cast;
pub mod generators;
pub mod indexing;
pub mod linalg;
pub mod reductions;
pub mod shape_ops;
mod types;

pub use types::NdArrayHandle;

// Re-export all FFI functions for lib.rs
pub use arithmetic::*;
pub use array::*;
pub use cast::*;
pub use generators::*;
pub use indexing::*;
pub use linalg::*;
pub use reductions::*;
pub use shape_ops::*;
