//! Basic array operations module.
//!
//! Provides core array functionality: creation, data access, properties, and serialization.

pub mod as_scalar;
pub mod copy;
pub mod create;
pub mod free;
pub mod get_data;

// Re-export all public functions
pub use as_scalar::*;
pub use copy::*;
pub use create::*;
pub use free::*;
pub use get_data::*;
