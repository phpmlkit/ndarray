//! Core array implementation modules.
//!
//! This module contains the internal implementation of NDArray types
//! separate from the FFI layer.

mod array_data;
pub mod dtype;
pub mod error;
mod handle;
#[macro_use]
pub mod macros;
mod metadata;
mod output_meta;
pub mod view_helpers;
mod wrapper;

pub use array_data::ArrayData;
pub use dtype::{DType, DTypeError};
pub use error::{
    set_last_error, ERR_ALLOC, ERR_DTYPE, ERR_GENERIC, ERR_INDEX, ERR_MATH, ERR_PANIC, ERR_SHAPE,
    SUCCESS,
};
pub use handle::NdArrayHandle;
pub use metadata::ArrayMetadata;
pub use output_meta::write_output_metadata;
pub use wrapper::NDArrayWrapper;
