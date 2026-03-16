//! Scalar operation macro for arithmetic operations on NDArrayWrapper instances.
//!
//! This module provides the `scalar_op_arithmetic` macro which:
//! - Extracts native views (zero-copy for matching array dtype)
//! - Casts the scalar (f64 from FFI) to the array's dtype for the operation
//! - Performs element-wise scalar operation
//! - Only works with numeric types and NOT Bool
//!
//! Usage:
//! ```rust
//! let result = scalar_op_arithmetic!(a_wrapper, a_meta, scalar, +);
//! ```

/// Scalar operation macro for arithmetic operations.
#[macro_export]
macro_rules! scalar_op_arithmetic {
    ($wrapper:expr, $meta:expr, $scalar:expr, $op:tt) => {{
        use crate::core::{
            view_helpers::{
                extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32,
                extract_view_i64, extract_view_i8, extract_view_u16, extract_view_u32,
                extract_view_u64, extract_view_u8,
            },
            ArrayData, NDArrayWrapper,
        };
        use crate::core::dtype::DType;
        use crate::core::error::{set_last_error, ERR_GENERIC};

        match $wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64($wrapper, $meta) else {
                    set_last_error("Failed to extract Float64 view".to_string());
                    return ERR_GENERIC;
                };
                let s = $scalar as f64;
                let result = view.mapv(|x| x $op s);
                NDArrayWrapper {
                    data: ArrayData::Float64(::std::sync::Arc::new(::parking_lot::RwLock::new(
                        result,
                    ))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32($wrapper, $meta) else {
                    set_last_error("Failed to extract Float32 view".to_string());
                    return ERR_GENERIC;
                };
                let s = $scalar as f32;
                let result = view.mapv(|x| x $op s);
                NDArrayWrapper {
                    data: ArrayData::Float32(::std::sync::Arc::new(::parking_lot::RwLock::new(
                        result,
                    ))),
                    dtype: DType::Float32,
                }
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64($wrapper, $meta) else {
                    set_last_error("Failed to extract Int64 view".to_string());
                    return ERR_GENERIC;
                };
                let s = $scalar as i64;
                let result = view.mapv(|x| x $op s);
                NDArrayWrapper {
                    data: ArrayData::Int64(::std::sync::Arc::new(::parking_lot::RwLock::new(
                        result,
                    ))),
                    dtype: DType::Int64,
                }
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32($wrapper, $meta) else {
                    set_last_error("Failed to extract Int32 view".to_string());
                    return ERR_GENERIC;
                };
                let s = $scalar as i32;
                let result = view.mapv(|x| x $op s);
                NDArrayWrapper {
                    data: ArrayData::Int32(::std::sync::Arc::new(::parking_lot::RwLock::new(
                        result,
                    ))),
                    dtype: DType::Int32,
                }
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16($wrapper, $meta) else {
                    set_last_error("Failed to extract Int16 view".to_string());
                    return ERR_GENERIC;
                };
                let s = $scalar as i16;
                let result = view.mapv(|x| x $op s);
                NDArrayWrapper {
                    data: ArrayData::Int16(::std::sync::Arc::new(::parking_lot::RwLock::new(
                        result,
                    ))),
                    dtype: DType::Int16,
                }
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8($wrapper, $meta) else {
                    set_last_error("Failed to extract Int8 view".to_string());
                    return ERR_GENERIC;
                };
                let s = $scalar as i8;
                let result = view.mapv(|x| x $op s);
                NDArrayWrapper {
                    data: ArrayData::Int8(::std::sync::Arc::new(::parking_lot::RwLock::new(
                        result,
                    ))),
                    dtype: DType::Int8,
                }
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64($wrapper, $meta) else {
                    set_last_error("Failed to extract Uint64 view".to_string());
                    return ERR_GENERIC;
                };
                let s = $scalar as u64;
                let result = view.mapv(|x| x $op s);
                NDArrayWrapper {
                    data: ArrayData::Uint64(::std::sync::Arc::new(::parking_lot::RwLock::new(
                        result,
                    ))),
                    dtype: DType::Uint64,
                }
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32($wrapper, $meta) else {
                    set_last_error("Failed to extract Uint32 view".to_string());
                    return ERR_GENERIC;
                };
                let s = $scalar as u32;
                let result = view.mapv(|x| x $op s);
                NDArrayWrapper {
                    data: ArrayData::Uint32(::std::sync::Arc::new(::parking_lot::RwLock::new(
                        result,
                    ))),
                    dtype: DType::Uint32,
                }
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16($wrapper, $meta) else {
                    set_last_error("Failed to extract Uint16 view".to_string());
                    return ERR_GENERIC;
                };
                let s = $scalar as u16;
                let result = view.mapv(|x| x $op s);
                NDArrayWrapper {
                    data: ArrayData::Uint16(::std::sync::Arc::new(::parking_lot::RwLock::new(
                        result,
                    ))),
                    dtype: DType::Uint16,
                }
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8($wrapper, $meta) else {
                    set_last_error("Failed to extract Uint8 view".to_string());
                    return ERR_GENERIC;
                };
                let s = $scalar as u8;
                let result = view.mapv(|x| x $op s);
                NDArrayWrapper {
                    data: ArrayData::Uint8(::std::sync::Arc::new(::parking_lot::RwLock::new(
                        result,
                    ))),
                    dtype: DType::Uint8,
                }
            }
            DType::Bool => {
                set_last_error("Arithmetic scalar operations not supported for Bool type".to_string());
                return ERR_GENERIC;
            }
        }
    }};
}
