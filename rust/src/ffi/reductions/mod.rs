//! Reduction and aggregation operations module.
//!
//! Provides scalar reductions (sum, mean, min, max, etc.) and
//! axis reductions with keepdims support.

pub(crate) mod helpers;

// Scalar reductions
pub mod argminmax_scalar;
pub mod cumprod;
pub mod cumsum;
pub mod mean_scalar;
pub mod min_max_scalar;
pub mod product_scalar;
pub mod std_var_scalar;
pub mod sum_scalar;

// Axis reductions
pub mod argminmax_axis;
pub mod cumprod_axis;
pub mod cumsum_axis;
pub mod mean_axis;
pub mod min_max_axis;
pub mod product_axis;
pub mod sort;
pub mod std_var_axis;
pub mod sum_axis;

// Re-export all FFI functions
pub use argminmax_axis::*;
pub use argminmax_scalar::*;
pub use cumprod::*;
pub use cumprod_axis::*;
pub use cumsum::*;
pub use cumsum_axis::*;
pub use mean_axis::*;
pub use mean_scalar::*;
pub use min_max_axis::*;
pub use min_max_scalar::*;
pub use product_axis::*;
pub use product_scalar::*;
pub use sort::*;
pub use std_var_axis::*;
pub use std_var_scalar::*;
pub use sum_axis::*;
pub use sum_scalar::*;
