//! Core array implementation modules.
//!
//! This module contains the internal implementation of NDArray types
//! separate from the FFI layer.

mod array_data;
#[macro_use]
mod macros;
pub mod view_helpers;
mod wrapper;

pub use array_data::ArrayData;
pub use wrapper::NDArrayWrapper;
