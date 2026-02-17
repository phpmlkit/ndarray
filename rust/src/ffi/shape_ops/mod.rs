//! Shape manipulation operations module.
//!
//! Provides operations for reshaping, transposing, flattening, and
//! manipulating array dimensions.

pub mod flatten;
pub mod helpers;
pub mod insert_axis;
pub mod invert_axis;
pub mod merge_axes;
pub mod pad;
pub mod permute_axes;
pub mod reshape;
pub mod squeeze;
pub mod swap_axes;
pub mod transpose;

// Re-export all FFI functions
pub use flatten::*;
pub use helpers::*;
pub use insert_axis::*;
pub use invert_axis::*;
pub use merge_axes::*;
pub use pad::*;
pub use permute_axes::*;
pub use reshape::*;
pub use squeeze::*;
pub use swap_axes::*;
pub use transpose::*;
