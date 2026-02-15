//! Shape manipulation operations module.
//!
//! Provides operations for reshaping, transposing, flattening, and
//! manipulating array dimensions.

pub mod flatten;
pub mod insert_axis;
pub mod invert_axis;
pub mod merge_axes;
pub mod reshape;
pub mod squeeze;
pub mod swap_axes;
pub mod transpose;

// Re-export all FFI functions
pub use flatten::*;
pub use insert_axis::*;
pub use invert_axis::*;
pub use merge_axes::*;
pub use reshape::*;
pub use squeeze::*;
pub use swap_axes::*;
pub use transpose::*;
