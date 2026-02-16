//! FFI module for PHP integration.
//!
//! This module exposes Rust functionality to PHP via C-compatible functions.

pub mod arithmetic;
pub mod array;
pub mod bitwise;
pub mod cast;
pub mod comparison;
pub mod generators;
pub mod indexing;
pub mod linalg;
pub mod math;
pub mod reductions;
pub mod shape_ops;
pub mod special;
mod types;

pub use types::NdArrayHandle;

// Re-export all FFI functions for lib.rs
pub use arithmetic::*;
pub use array::*;
pub use bitwise::*;
pub use cast::*;
pub use comparison::*;
pub use generators::*;
pub use indexing::*;
pub use linalg::*;
pub use math::*;
pub use reductions::*;
pub use shape_ops::*;
pub use special::*;
