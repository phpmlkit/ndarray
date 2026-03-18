//! Core type definitions for NDArray PHP.
//!
//! This module contains the type definitions used throughout the library.

mod array_data;
pub mod dtype;
mod handle;
mod metadata;
mod pad_mode;
mod sort_kind;
mod wrapper;

pub use array_data::ArrayData;
pub use dtype::{DType, DTypeError};
pub use handle::NdArrayHandle;
pub use metadata::ArrayMetadata;
pub use pad_mode::PadMode;
pub use sort_kind::SortKind;
pub use wrapper::NDArrayWrapper;
