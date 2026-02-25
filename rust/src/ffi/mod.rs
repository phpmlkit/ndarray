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
pub mod logical;
pub mod math;
pub mod metadata;
mod output_meta;
pub mod reductions;
pub mod shape_ops;
pub mod sorting;
pub mod special;
pub mod stacking;
pub mod to_string;
mod types;

pub use metadata::{HasViewMetadata, ViewMetadata};
pub use output_meta::write_output_metadata;
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
pub use logical::*;
pub use math::*;
pub use reductions::*;
pub use shape_ops::*;
pub use sorting::*;
pub use special::*;
pub use stacking::*;
pub use to_string::ndarray_to_string;
