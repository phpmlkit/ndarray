//! Shape manipulation operations module.
//!
//! Provides operations for reshaping, transposing, flattening, and
//! manipulating array dimensions.

pub mod flatten;
pub mod reshape;
pub mod squeeze;
pub mod transpose;

// Re-export all FFI functions
pub use flatten::*;
pub use reshape::*;
pub use squeeze::*;
pub use transpose::*;
