//! Reduction and aggregation operations module.
//!
//! Provides scalar reductions (sum, mean, min, max, etc.) and
//! axis reductions with keepdims support.

pub(crate) mod helpers;

pub mod argmax;
pub mod argmin;
pub mod bincount;
pub mod cumprod;
pub mod cumsum;
pub mod mean;
pub mod max;
pub mod min;
pub mod product;
pub mod var;
pub mod std;
pub mod sum;

// Re-export all FFI functions
pub use argmax::*;
pub use argmin::*;
pub use bincount::*;
pub use cumprod::*;
pub use cumsum::*;
pub use mean::*;
pub use max::*;
pub use min::*;
pub use product::*;
pub use std::*;
pub use var::*;
pub use sum::*;
