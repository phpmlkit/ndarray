//! Macros for array operations.
//!
//! This module contains all macros used for binary operations,
//! scalar operations, comparisons, logical operations, and view extraction.
//!
//! Note: All macros are exported at the crate root via `#[macro_export]`,
//! so they can be used directly as `crate::macro_name!` without importing.

pub mod binary_op_arithmetic;
pub mod binary_op_bitwise;
pub mod binary_op_comparison;
pub mod binary_op_logical;
pub mod broadcast_binary;
pub mod core_macros;
pub mod ffi_guard;
pub mod view;
pub mod scalar_op_arithmetic;
pub mod scalar_op_comparison;
pub mod scalar_op_bitwise;
