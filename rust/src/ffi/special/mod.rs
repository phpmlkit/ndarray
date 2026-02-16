//! Special operations module.
//!
//! Provides specialized array operations that don't fit into other categories.

pub mod clamp;

// Re-export all FFI functions
pub use clamp::*;
