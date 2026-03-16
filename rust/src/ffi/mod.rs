//! FFI module for PHP integration.
//!
//! This module exposes Rust functionality to PHP via C-compatible functions.

pub mod arithmetic;
pub mod array;
pub mod bitwise;
pub mod comparison;
pub mod generators;
pub mod indexing;
pub mod linalg;
pub mod logical;
pub mod math;
pub mod reductions;
pub mod shape_ops;
pub mod sorting;
pub mod misc;
pub mod stacking;

pub use crate::core::{ArrayMetadata, NdArrayHandle, write_output_metadata};

// Re-export all FFI functions for lib.rs
pub use arithmetic::*;
pub use array::*;
pub use bitwise::*;
pub use comparison::*;
pub use generators::*;
pub use indexing::*;
pub use linalg::*;
pub use logical::*;
pub use math::*;
pub use reductions::*;
pub use shape_ops::*;
pub use sorting::*;
pub use misc::*;
pub use stacking::*;
