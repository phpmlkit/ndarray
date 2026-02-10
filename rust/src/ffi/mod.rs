//! FFI module for PHP integration.
//!
//! This module exposes Rust functionality to PHP via C-compatible functions.

pub mod create;
pub mod data;
pub mod generators;
pub mod properties;
pub mod serialize;
mod serialize_helpers;
mod types;

pub use types::NdArrayHandle;

// Re-export all FFI functions for lib.rs
pub use create::*;
pub use data::*;
pub use generators::*;
pub use properties::*;
pub use serialize::*;
