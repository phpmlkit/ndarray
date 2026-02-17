//! Element-level indexing FFI functions.
//!
//! Provides type-specific get/set operations for single elements at a flat index.

// Public modules
pub mod assign;
pub mod fill;
pub mod get_element;
pub mod put;
pub mod scatter_add;
pub mod set_element;
pub mod take;
pub mod where_op;

// Re-export all public functions for convenient access
pub use assign::*;
pub use fill::*;
pub use get_element::*;
pub use put::*;
pub use scatter_add::*;
pub use set_element::*;
pub use take::*;
pub use where_op::*;