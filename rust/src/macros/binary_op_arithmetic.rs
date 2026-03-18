//! Binary operation macro for arithmetic operations on NDArrayWrapper instances.
//!
//! This module provides the `binary_op_arithmetic` macro which:
//! - Extracts views for native type pairs (zero-copy for matching types)
//! - Promotes mixed types and converts before operation
//! - Performs element-wise operations with broadcasting
//! - Only works with numeric types and NOT Bool

/// Binary operation macro for performing arithmetic operations on NDArrayWrapper instances.
/// Takes a function (e.g. add, sub, minimum) that maps (a, b) -> result element-wise.
///
/// Usage:
/// ```rust
/// let result = binary_op_arithmetic!(a_wrapper, a_meta, b_wrapper, b_meta, add);
/// ```
#[macro_export]
macro_rules! binary_op_arithmetic {
    ($a_wrapper:expr, $a_meta:expr, $b_wrapper:expr, $b_meta:expr, $fn:path) => {{
        use crate::helpers::{
            extract_view_as_f32, extract_view_as_f64, extract_view_as_i16, extract_view_as_i32,
            extract_view_as_i64, extract_view_as_i8, extract_view_as_u16, extract_view_as_u32,
            extract_view_as_u64, extract_view_as_u8, extract_view_f32, extract_view_f64,
            extract_view_i16, extract_view_i32, extract_view_i64, extract_view_i8,
            extract_view_u16, extract_view_u32, extract_view_u64, extract_view_u8, set_last_error,
            ERR_GENERIC,
        };
        use crate::types::dtype::DType;
        use crate::types::{ArrayData, NDArrayWrapper};

        let out_dtype = DType::promote($a_wrapper.dtype, $b_wrapper.dtype);

        if $a_wrapper.dtype == $b_wrapper.dtype {
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
                    let result = crate::broadcast_binary!(a_view, b_view, $fn);
                    NDArrayWrapper {
                        data: ArrayData::Float64(::std::sync::Arc::new(
                            ::parking_lot::RwLock::new(result),
                        )),
                        dtype: DType::Float64,
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
                    let result = crate::broadcast_binary!(a_view, b_view, $fn);
                    NDArrayWrapper {
                        data: ArrayData::Float32(::std::sync::Arc::new(
                            ::parking_lot::RwLock::new(result),
                        )),
                        dtype: DType::Float32,
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
                    let result = crate::broadcast_binary!(a_view, b_view, $fn);
                    NDArrayWrapper {
                        data: ArrayData::Int64(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Int64,
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
                    let result = crate::broadcast_binary!(a_view, b_view, $fn);
                    NDArrayWrapper {
                        data: ArrayData::Int32(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Int32,
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
                    let result = crate::broadcast_binary!(a_view, b_view, $fn);
                    NDArrayWrapper {
                        data: ArrayData::Int16(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Int16,
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
                    let result = crate::broadcast_binary!(a_view, b_view, $fn);
                    NDArrayWrapper {
                        data: ArrayData::Int8(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Int8,
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
                    let result = crate::broadcast_binary!(a_view, b_view, $fn);
                    NDArrayWrapper {
                        data: ArrayData::Uint64(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Uint64,
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
                    let result = crate::broadcast_binary!(a_view, b_view, $fn);
                    NDArrayWrapper {
                        data: ArrayData::Uint32(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Uint32,
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
                    let result = crate::broadcast_binary!(a_view, b_view, $fn);
                    NDArrayWrapper {
                        data: ArrayData::Uint16(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Uint16,
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
                    let result = crate::broadcast_binary!(a_view, b_view, $fn);
                    NDArrayWrapper {
                        data: ArrayData::Uint8(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Uint8,
                    }
                }
                DType::Bool => {
                    set_last_error("Arithmetic operations not supported for Bool type".to_string());
                    return ERR_GENERIC;
                }
            }
        } else {
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
                    let result = crate::broadcast_binary!(a_arr, b_arr, $fn);
                    NDArrayWrapper {
                        data: ArrayData::Float64(::std::sync::Arc::new(
                            ::parking_lot::RwLock::new(result),
                        )),
                        dtype: DType::Float64,
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
                    let result = crate::broadcast_binary!(a_arr, b_arr, $fn);
                    NDArrayWrapper {
                        data: ArrayData::Float32(::std::sync::Arc::new(
                            ::parking_lot::RwLock::new(result),
                        )),
                        dtype: DType::Float32,
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
                    let result = crate::broadcast_binary!(a_arr, b_arr, $fn);
                    NDArrayWrapper {
                        data: ArrayData::Int64(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Int64,
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
                    let result = crate::broadcast_binary!(a_arr, b_arr, $fn);
                    NDArrayWrapper {
                        data: ArrayData::Int32(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Int32,
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
                    let result = crate::broadcast_binary!(a_arr, b_arr, $fn);
                    NDArrayWrapper {
                        data: ArrayData::Int16(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Int16,
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
                    let result = crate::broadcast_binary!(a_arr, b_arr, $fn);
                    NDArrayWrapper {
                        data: ArrayData::Int8(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Int8,
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
                    let result = crate::broadcast_binary!(a_arr, b_arr, $fn);
                    NDArrayWrapper {
                        data: ArrayData::Uint64(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Uint64,
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
                    let result = crate::broadcast_binary!(a_arr, b_arr, $fn);
                    NDArrayWrapper {
                        data: ArrayData::Uint32(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Uint32,
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
                    let result = crate::broadcast_binary!(a_arr, b_arr, $fn);
                    NDArrayWrapper {
                        data: ArrayData::Uint16(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Uint16,
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
                    let result = crate::broadcast_binary!(a_arr, b_arr, $fn);
                    NDArrayWrapper {
                        data: ArrayData::Uint8(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Uint8,
                    }
                }
                DType::Bool => {
                    set_last_error("Arithmetic operations not supported for Bool type".to_string());
                    return ERR_GENERIC;
                }
            }
        }
    }};
}
