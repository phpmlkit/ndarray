//! Scalar operation macro for arithmetic operations on NDArrayWrapper instances.
//!
//! This module provides the `scalar_op_arithmetic` macro which:
//! - Computes the promoted dtype from array and scalar dtypes (NumPy-style type promotion)
//! - Extracts the array as the promoted dtype using extract_view_as_* helpers
//! - Reads and casts the scalar value using get_scalar_as_* helpers
//! - Performs element-wise scalar operation
//! - Only works with numeric types and NOT Bool
//!
//! Usage:
//! ```rust
//! let result = scalar_op_arithmetic!(a_wrapper, a_meta, scalar, scalar_dtype, +);
//! ```

#[macro_export]
macro_rules! scalar_op_arithmetic {
    ($wrapper:expr, $meta:expr, $scalar:expr, $scalar_dtype:expr, $op:tt) => {{
        use crate::helpers::{
            extract_view_as_c128, extract_view_as_c64, extract_view_as_f32,
            extract_view_as_f64, extract_view_as_i16, extract_view_as_i32,
            extract_view_as_i64, extract_view_as_i8, extract_view_as_u16,
            extract_view_as_u32, extract_view_as_u64, extract_view_as_u8,
            get_scalar_as_f64, get_scalar_as_f32, get_scalar_as_i64,
            get_scalar_as_i32, get_scalar_as_i16, get_scalar_as_i8,
            get_scalar_as_u64, get_scalar_as_u32, get_scalar_as_u16,
            get_scalar_as_u8, get_scalar_as_c64, get_scalar_as_c128,
            set_last_error, ERR_GENERIC,
        };
        use crate::types::dtype::DType;
        use crate::types::{ArrayData, NDArrayWrapper};

        let array_dtype = $wrapper.dtype;
        let out_dtype = DType::promote(array_dtype, $scalar_dtype);

        match out_dtype {
            DType::Float64 => {
                let Some(view) = extract_view_as_f64($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Float64".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_f64($scalar, $scalar_dtype) };
                let result = view.mapv(|x| x $op s);
                NDArrayWrapper {
                    data: ArrayData::Float64(::std::sync::Arc::new(::parking_lot::RwLock::new(
                        result,
                    ))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let Some(view) = extract_view_as_f32($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Float32".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_f32($scalar, $scalar_dtype) };
                let result = view.mapv(|x| x $op s);
                NDArrayWrapper {
                    data: ArrayData::Float32(::std::sync::Arc::new(::parking_lot::RwLock::new(
                        result,
                    ))),
                    dtype: DType::Float32,
                }
            }
            DType::Int64 => {
                let Some(view) = extract_view_as_i64($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Int64".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_i64($scalar, $scalar_dtype) };
                let result = view.mapv(|x| x $op s);
                NDArrayWrapper {
                    data: ArrayData::Int64(::std::sync::Arc::new(::parking_lot::RwLock::new(
                        result,
                    ))),
                    dtype: DType::Int64,
                }
            }
            DType::Int32 => {
                let Some(view) = extract_view_as_i32($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Int32".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_i32($scalar, $scalar_dtype) };
                let result = view.mapv(|x| x $op s);
                NDArrayWrapper {
                    data: ArrayData::Int32(::std::sync::Arc::new(::parking_lot::RwLock::new(
                        result,
                    ))),
                    dtype: DType::Int32,
                }
            }
            DType::Int16 => {
                let Some(view) = extract_view_as_i16($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Int16".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_i16($scalar, $scalar_dtype) };
                let result = view.mapv(|x| x $op s);
                NDArrayWrapper {
                    data: ArrayData::Int16(::std::sync::Arc::new(::parking_lot::RwLock::new(
                        result,
                    ))),
                    dtype: DType::Int16,
                }
            }
            DType::Int8 => {
                let Some(view) = extract_view_as_i8($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Int8".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_i8($scalar, $scalar_dtype) };
                let result = view.mapv(|x| x $op s);
                NDArrayWrapper {
                    data: ArrayData::Int8(::std::sync::Arc::new(::parking_lot::RwLock::new(
                        result,
                    ))),
                    dtype: DType::Int8,
                }
            }
            DType::Uint64 => {
                let Some(view) = extract_view_as_u64($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Uint64".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_u64($scalar, $scalar_dtype) };
                let result = view.mapv(|x| x $op s);
                NDArrayWrapper {
                    data: ArrayData::Uint64(::std::sync::Arc::new(::parking_lot::RwLock::new(
                        result,
                    ))),
                    dtype: DType::Uint64,
                }
            }
            DType::Uint32 => {
                let Some(view) = extract_view_as_u32($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Uint32".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_u32($scalar, $scalar_dtype) };
                let result = view.mapv(|x| x $op s);
                NDArrayWrapper {
                    data: ArrayData::Uint32(::std::sync::Arc::new(::parking_lot::RwLock::new(
                        result,
                    ))),
                    dtype: DType::Uint32,
                }
            }
            DType::Uint16 => {
                let Some(view) = extract_view_as_u16($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Uint16".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_u16($scalar, $scalar_dtype) };
                let result = view.mapv(|x| x $op s);
                NDArrayWrapper {
                    data: ArrayData::Uint16(::std::sync::Arc::new(::parking_lot::RwLock::new(
                        result,
                    ))),
                    dtype: DType::Uint16,
                }
            }
            DType::Uint8 => {
                let Some(view) = extract_view_as_u8($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Uint8".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_u8($scalar, $scalar_dtype) };
                let result = view.mapv(|x| x $op s);
                NDArrayWrapper {
                    data: ArrayData::Uint8(::std::sync::Arc::new(::parking_lot::RwLock::new(
                        result,
                    ))),
                    dtype: DType::Uint8,
                }
            }
            DType::Complex64 => {
                let Some(view) = extract_view_as_c64($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Complex64".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_c64($scalar, $scalar_dtype) };
                let result = view.mapv(|x| x $op s);
                NDArrayWrapper {
                    data: ArrayData::Complex64(::std::sync::Arc::new(::parking_lot::RwLock::new(
                        result,
                    ))),
                    dtype: DType::Complex64,
                }
            }
            DType::Complex128 => {
                let Some(view) = extract_view_as_c128($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Complex128".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_c128($scalar, $scalar_dtype) };
                let result = view.mapv(|x| x $op s);
                NDArrayWrapper {
                    data: ArrayData::Complex128(::std::sync::Arc::new(::parking_lot::RwLock::new(
                        result,
                    ))),
                    dtype: DType::Complex128,
                }
            }
            DType::Bool => {
                set_last_error(
                    "Arithmetic operations not supported for Bool type".to_string(),
                );
                return ERR_GENERIC;
            }
        }
    }};
}
