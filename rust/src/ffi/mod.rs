//! FFI module for PHP integration.
//!
//! This module exposes Rust functionality to PHP via C-compatible functions.

pub mod arithmetic;
pub mod array;
pub mod bitwise;
pub mod comparison;
pub mod fft;
pub mod generators;
pub mod indexing;
pub mod linalg;
pub mod logical;
pub mod math;
pub mod misc;
pub mod reductions;
pub mod shape_ops;
pub mod sorting;
pub mod stacking;
pub mod windows;

pub use arithmetic::*;
pub use array::*;
pub use bitwise::*;
pub use comparison::*;
pub use fft::*;
pub use generators::*;
pub use indexing::*;
pub use linalg::*;
pub use logical::*;
pub use math::*;
pub use misc::*;
pub use reductions::*;
pub use shape_ops::*;
pub use sorting::*;
pub use stacking::*;
pub use windows::*;
