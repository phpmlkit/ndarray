//! Element-level indexing FFI functions.
//!
//! Provides type-specific get/set operations for single elements at a flat index.

// Public modules
pub mod assign;
pub mod fill;
pub mod get_element;
pub mod set_element;

// Re-export all public functions for convenient access
pub use assign::*;
pub use fill::*;
pub use get_element::*;
pub use set_element::*;
