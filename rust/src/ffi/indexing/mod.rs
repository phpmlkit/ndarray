//! Element-level indexing FFI functions.
//!
//! Provides type-specific get/set operations for single elements at a flat index.

// Public modules
pub mod assign;
pub mod fill;
pub mod get_element;
pub mod put;
pub mod put_along_axis;
pub mod scatter_add;
pub mod set_element;
pub mod take;
pub mod take_along_axis;
pub mod utils;
pub mod where_op;

// Re-export all public functions for convenient access
pub use assign::*;
pub use fill::*;
pub use get_element::*;
pub use put::*;
pub use put_along_axis::*;
pub use scatter_add::*;
pub use set_element::*;
pub use take::*;
pub use take_along_axis::*;
pub use where_op::*;
