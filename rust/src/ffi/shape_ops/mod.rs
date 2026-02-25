//! Shape manipulation operations module.
//!
//! Provides operations for reshaping, transposing, flattening, and
//! manipulating array dimensions.

pub mod flatten;
pub mod flip;
pub mod helpers;
pub mod insert;
pub mod merge;
pub mod pad;
pub mod permute;
pub mod repeat;
pub mod reshape;
pub mod squeeze;
pub mod swap;
pub mod tile;
pub mod transpose;

// Re-export all FFI functions
pub use flatten::*;
pub use flip::*;
pub use helpers::*;
pub use insert::*;
pub use merge::*;
pub use pad::*;
pub use permute::*;
pub use repeat::*;
pub use reshape::*;
pub use squeeze::*;
pub use swap::*;
pub use tile::*;
pub use transpose::*;
