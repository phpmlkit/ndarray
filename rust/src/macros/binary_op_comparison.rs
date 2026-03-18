//! Binary operation macro for comparison operations.
//!
//! This module provides the `binary_op_comparison` macro which:
//! - Extracts views for native type pairs (zero-copy for matching types)
//! - Promotes mixed types and converts before operation
//! - Performs element-wise operations with broadcasting
//! - Works with all numeric types and Bool
//! - Always returns Bool array
//!
//! Supported comparison operators: ==, !=, <, <=, >, >=

/// Binary operation macro for comparison operations.
///
/// Usage:
/// ```rust
/// let result = binary_op_comparison!(a_wrapper, a_meta, b_wrapper, b_meta, eq);
/// ```
#[macro_export]
macro_rules! binary_op_comparison {
    ($a_wrapper:expr, $a_meta:expr, $b_wrapper:expr, $b_meta:expr, $cmp_op:ident) => {{
        use crate::helpers::{
            extract_view_as_bool, extract_view_as_f32, extract_view_as_f64, extract_view_as_i16,
            extract_view_as_i32, extract_view_as_i64, extract_view_as_i8, extract_view_as_u16,
            extract_view_as_u32, extract_view_as_u64, extract_view_as_u8, extract_view_bool,
            extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32,
            extract_view_i64, extract_view_i8, extract_view_u16, extract_view_u32,
            extract_view_u64, extract_view_u8, set_last_error, ERR_GENERIC,
        };
        use crate::types::dtype::DType;
        use crate::types::{ArrayData, NDArrayWrapper};

        let out_dtype = DType::promote($a_wrapper.dtype, $b_wrapper.dtype);

        if $a_wrapper.dtype == $b_wrapper.dtype {
            // Same dtype: use native views (zero-copy fast path)
            match out_dtype {
                DType::Float64 => {
                    let Some(a_view) = extract_view_f64($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract Float64 operand a".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_view) = extract_view_f64($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract Float64 operand b".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_view, b_view, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Float32 => {
                    let Some(a_view) = extract_view_f32($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract Float32 operand a".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_view) = extract_view_f32($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract Float32 operand b".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_view, b_view, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Int64 => {
                    let Some(a_view) = extract_view_i64($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract Int64 operand a".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_view) = extract_view_i64($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract Int64 operand b".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_view, b_view, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Int32 => {
                    let Some(a_view) = extract_view_i32($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract Int32 operand a".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_view) = extract_view_i32($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract Int32 operand b".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_view, b_view, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Int16 => {
                    let Some(a_view) = extract_view_i16($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract Int16 operand a".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_view) = extract_view_i16($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract Int16 operand b".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_view, b_view, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Int8 => {
                    let Some(a_view) = extract_view_i8($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract Int8 operand a".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_view) = extract_view_i8($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract Int8 operand b".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_view, b_view, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Uint64 => {
                    let Some(a_view) = extract_view_u64($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract Uint64 operand a".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_view) = extract_view_u64($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract Uint64 operand b".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_view, b_view, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Uint32 => {
                    let Some(a_view) = extract_view_u32($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract Uint32 operand a".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_view) = extract_view_u32($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract Uint32 operand b".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_view, b_view, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Uint16 => {
                    let Some(a_view) = extract_view_u16($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract Uint16 operand a".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_view) = extract_view_u16($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract Uint16 operand b".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_view, b_view, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Uint8 => {
                    let Some(a_view) = extract_view_u8($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract Uint8 operand a".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_view) = extract_view_u8($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract Uint8 operand b".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_view, b_view, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Bool => {
                    let Some(a_view) = extract_view_bool($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract Bool operand a".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_view) = extract_view_bool($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract Bool operand b".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_view, b_view, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
            }
        } else {
            // Different dtypes: convert to target type
            match out_dtype {
                DType::Float64 => {
                    let Some(a_arr) = extract_view_as_f64($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract operand a as Float64".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_arr) = extract_view_as_f64($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract operand b as Float64".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_arr, b_arr, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Float32 => {
                    let Some(a_arr) = extract_view_as_f32($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract operand a as Float32".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_arr) = extract_view_as_f32($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract operand b as Float32".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_arr, b_arr, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Int64 => {
                    let Some(a_arr) = extract_view_as_i64($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract operand a as Int64".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_arr) = extract_view_as_i64($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract operand b as Int64".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_arr, b_arr, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Int32 => {
                    let Some(a_arr) = extract_view_as_i32($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract operand a as Int32".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_arr) = extract_view_as_i32($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract operand b as Int32".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_arr, b_arr, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Int16 => {
                    let Some(a_arr) = extract_view_as_i16($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract operand a as Int16".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_arr) = extract_view_as_i16($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract operand b as Int16".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_arr, b_arr, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Int8 => {
                    let Some(a_arr) = extract_view_as_i8($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract operand a as Int8".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_arr) = extract_view_as_i8($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract operand b as Int8".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_arr, b_arr, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Uint64 => {
                    let Some(a_arr) = extract_view_as_u64($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract operand a as Uint64".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_arr) = extract_view_as_u64($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract operand b as Uint64".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_arr, b_arr, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Uint32 => {
                    let Some(a_arr) = extract_view_as_u32($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract operand a as Uint32".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_arr) = extract_view_as_u32($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract operand b as Uint32".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_arr, b_arr, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Uint16 => {
                    let Some(a_arr) = extract_view_as_u16($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract operand a as Uint16".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_arr) = extract_view_as_u16($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract operand b as Uint16".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_arr, b_arr, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Uint8 => {
                    let Some(a_arr) = extract_view_as_u8($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract operand a as Uint8".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_arr) = extract_view_as_u8($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract operand b as Uint8".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_arr, b_arr, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Bool => {
                    let Some(a_arr) = extract_view_as_bool($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract operand a as Bool".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_arr) = extract_view_as_bool($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract operand b as Bool".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_arr, b_arr, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
            }
        }
    }};
}
