//! Scalar operation macro for bitwise operations on NDArrayWrapper instances.
//!
//! This module provides the `scalar_op_bitwise` macro which:
//! - Computes the promoted dtype from array and scalar dtypes (NumPy-style type promotion)
//! - Extracts the array as the promoted dtype using extract_view_as_* helpers
//! - Dereferences the scalar (*const c_void from FFI) to the promoted dtype for the operation
//! - Performs element-wise scalar bitwise operation
//! - Only works with integer types and Bool (NOT Float/Complex types)
//!
//! Supported types: Int8, Int16, Int32, Int64, Uint8, Uint16, Uint32, Uint64, Bool
//! (Bool only for &, |, ^; use `no_bool` suffix for <<, >> which exclude Bool).
//!
//! Usage:
//! ```rust
//! let result = scalar_op_bitwise!(a_wrapper, a_meta, scalar, scalar_dtype, &);   // bitand
//! let result = scalar_op_bitwise!(a_wrapper, a_meta, scalar, scalar_dtype, |);   // bitor
//! let result = scalar_op_bitwise!(a_wrapper, a_meta, scalar, scalar_dtype, ^);   // bitxor
//! let result = scalar_op_bitwise!(a_wrapper, a_meta, scalar, scalar_dtype, <<, no_bool);  // left shift
//! let result = scalar_op_bitwise!(a_wrapper, a_meta, scalar, scalar_dtype, >>, no_bool);  // right shift
//! ```

/// Scalar bitwise operation (supports Bool for &, |, ^).
#[macro_export]
macro_rules! scalar_op_bitwise {
    ($wrapper:expr, $meta:expr, $scalar:expr, $scalar_dtype:expr, $op:tt) => {{
        $crate::scalar_op_bitwise_impl!($wrapper, $meta, $scalar, $scalar_dtype, $op, true)
    }};
    ($wrapper:expr, $meta:expr, $scalar:expr, $scalar_dtype:expr, $op:tt, no_bool) => {{
        $crate::scalar_op_bitwise_impl!($wrapper, $meta, $scalar, $scalar_dtype, $op, false)
    }};
}

/// Internal implementation with allow_bool flag.
#[macro_export]
macro_rules! scalar_op_bitwise_impl {
    ($wrapper:expr, $meta:expr, $scalar:expr, $scalar_dtype:expr, $op:tt, $allow_bool:expr) => {{
        use crate::helpers::{
            extract_view_as_bool, extract_view_as_i16, extract_view_as_i32, extract_view_as_i64,
            extract_view_as_i8, extract_view_as_u16, extract_view_as_u32, extract_view_as_u64,
            extract_view_as_u8, get_scalar_as_i64, get_scalar_as_i32,
            get_scalar_as_i16, get_scalar_as_i8, get_scalar_as_u64, get_scalar_as_u32,
            get_scalar_as_u16, get_scalar_as_u8, set_last_error, ERR_GENERIC,
        };
        use crate::types::dtype::DType;
        use crate::types::{ArrayData, NDArrayWrapper};

        let out_dtype = DType::promote($wrapper.dtype, $scalar_dtype);

        if matches!(out_dtype, DType::Float32 | DType::Float64 | DType::Complex64 | DType::Complex128) {
            set_last_error("Bitwise operations not supported for float or complex types".to_string());
            return ERR_GENERIC;
        }

        if out_dtype == DType::Bool && !$allow_bool {
            set_last_error("Shift operations only supported for integer types".to_string());
            return ERR_GENERIC;
        }

        match out_dtype {
            DType::Bool => {
                let Some(view) = extract_view_as_bool($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Bool".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_u8($scalar, $scalar_dtype) };
                let result = view.mapv(|x| x $op s);
                NDArrayWrapper {
                    data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(result))),
                    dtype: DType::Bool,
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
                    data: ArrayData::Int64(::std::sync::Arc::new(::parking_lot::RwLock::new(result))),
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
                    data: ArrayData::Int32(::std::sync::Arc::new(::parking_lot::RwLock::new(result))),
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
                    data: ArrayData::Int16(::std::sync::Arc::new(::parking_lot::RwLock::new(result))),
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
                    data: ArrayData::Int8(::std::sync::Arc::new(::parking_lot::RwLock::new(result))),
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
                    data: ArrayData::Uint64(::std::sync::Arc::new(::parking_lot::RwLock::new(result))),
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
                    data: ArrayData::Uint32(::std::sync::Arc::new(::parking_lot::RwLock::new(result))),
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
                    data: ArrayData::Uint16(::std::sync::Arc::new(::parking_lot::RwLock::new(result))),
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
                    data: ArrayData::Uint8(::std::sync::Arc::new(::parking_lot::RwLock::new(result))),
                    dtype: DType::Uint8,
                }
            }
            _ => {
                set_last_error("Bitwise operations not supported for this dtype combination".to_string());
                return ERR_GENERIC;
            }
        }
    }};
}
