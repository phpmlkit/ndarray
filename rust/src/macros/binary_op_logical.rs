//! Binary operation macro for logical operations.
//!
//! This module provides the `binary_op_logical` macro which:
//! - For Bool+Bool: extracts native views (zero-copy fast path)
//! - For mixed types: converts both to bool
//! - Performs element-wise operations with broadcasting
//! - Always returns Bool array
//!
//! Supported logical operators: and, or, xor

/// Binary operation macro for logical operations.
///
/// Usage:
/// ```rust
/// let result = binary_op_logical!(a_wrapper, a_meta, b_wrapper, b_meta, and);
/// ```
#[macro_export]
macro_rules! binary_op_logical {
    ($a_wrapper:expr, $a_meta:expr, $b_wrapper:expr, $b_meta:expr, $logical_op:ident) => {{
        use crate::helpers::{
            extract_view_as_bool, extract_view_bool, set_last_error, ERR_GENERIC,
        };
        use crate::types::dtype::DType;
        use crate::types::{ArrayData, NDArrayWrapper};

        if $a_wrapper.dtype == DType::Bool && $b_wrapper.dtype == DType::Bool {
            let Some(a_view) = extract_view_bool($a_wrapper, $a_meta) else {
                set_last_error("Failed to extract Bool operand a".to_string());
                return ERR_GENERIC;
            };
            let Some(b_view) = extract_view_bool($b_wrapper, $b_meta) else {
                set_last_error("Failed to extract Bool operand b".to_string());
                return ERR_GENERIC;
            };
            let result = crate::broadcast_binary!(a_view, b_view, $logical_op);
            NDArrayWrapper {
                data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(result))),
                dtype: DType::Bool,
            }
        } else {
            let Some(a_arr) = extract_view_as_bool($a_wrapper, $a_meta) else {
                set_last_error("Failed to extract operand a as Bool".to_string());
                return ERR_GENERIC;
            };
            let Some(b_arr) = extract_view_as_bool($b_wrapper, $b_meta) else {
                set_last_error("Failed to extract operand b as Bool".to_string());
                return ERR_GENERIC;
            };
            let result = crate::broadcast_binary!(a_arr, b_arr, $logical_op);
            NDArrayWrapper {
                data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(result))),
                dtype: DType::Bool,
            }
        }
    }};
}
