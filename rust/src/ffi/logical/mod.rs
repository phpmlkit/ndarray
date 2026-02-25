//! Logical operations module.
//!
//! Provides element-wise logical operations including AND, OR, NOT, and XOR.
//! All operations work on any input type (converts to bool first) and return Bool arrays.

pub mod and;
pub mod not;
pub mod or;
pub mod xor;

// Re-export all FFI functions
pub use and::*;
pub use not::*;
pub use or::*;
pub use xor::*;
