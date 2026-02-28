//! Basic arithmetic operations module.
//!
//! Provides element-wise arithmetic operations with broadcasting support
//! for both array-array and array-scalar operations.

pub mod add;
pub mod div;
pub mod minmax;
pub mod mul;
pub mod rem;
pub mod sub;

// Re-export all FFI functions
pub use add::*;
pub use div::*;
pub use minmax::*;
pub use mul::*;
pub use rem::*;
pub use sub::*;
