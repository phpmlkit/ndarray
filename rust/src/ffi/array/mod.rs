//! Basic array operations module.
//!
//! Provides core array functionality: creation, data access, properties, and serialization.

pub mod create;
pub mod data;
pub mod properties;
pub mod scalar;

// Re-export all public functions
pub use create::*;
pub use data::*;
pub use properties::*;
pub use scalar::*;
