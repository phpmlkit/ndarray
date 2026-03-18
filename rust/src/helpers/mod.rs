//! Helper utilities for NDArray PHP.
//!
//! This module provides view extraction helpers and output metadata utilities.

pub mod error;
pub mod indexing;
pub mod output;
pub mod view;

pub use indexing::{normalize_axis, normalize_index};

pub use error::{
    set_last_error, ERR_ALLOC, ERR_DTYPE, ERR_GENERIC, ERR_INDEX, ERR_MATH, ERR_PANIC, ERR_SHAPE,
    SUCCESS,
};
pub use output::write_output_metadata;
pub use view::{
    broadcast_shape, extract_view_as_bool, extract_view_as_f32, extract_view_as_f64,
    extract_view_as_i16, extract_view_as_i32, extract_view_as_i64, extract_view_as_i8,
    extract_view_as_u16, extract_view_as_u32, extract_view_as_u64, extract_view_as_u8,
    extract_view_bool, extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32,
    extract_view_i64, extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64,
    extract_view_u8,
};
