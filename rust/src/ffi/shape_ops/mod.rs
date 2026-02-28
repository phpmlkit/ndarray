//! Shape manipulation operations module.
//!
//! Provides operations for reshaping, transposing, flattening, and
//! manipulating array dimensions.

pub mod flatten;
pub mod flip;
pub mod helpers;
pub mod pad;
pub mod permute;
pub mod repeat;
pub mod reshape;
pub mod tile;
pub mod transpose;

// Re-export all FFI functions
pub use flatten::*;
pub use flip::*;
pub use helpers::*;
pub use pad::*;
pub use permute::*;
pub use repeat::*;
pub use reshape::*;
pub use tile::*;
pub use transpose::*;
