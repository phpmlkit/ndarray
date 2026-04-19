//! Helper utilities for NDArray PHP.
//!
//! This module provides view extraction helpers and output metadata utilities.

pub mod elementwise_minmax;
pub mod error;
pub mod indexing;
pub mod output;
pub mod scalar;
pub mod view;

pub use indexing::{normalize_axis, normalize_index};

pub use error::{
    set_last_error, ERR_ALLOC, ERR_DTYPE, ERR_GENERIC, ERR_INDEX, ERR_MATH, ERR_PANIC, ERR_SHAPE,
    SUCCESS,
};
pub use output::write_output_metadata;
pub use scalar::*;
pub use view::*;
