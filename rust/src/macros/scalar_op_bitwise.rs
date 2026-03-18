//! Scalar operation macro for bitwise operations on NDArrayWrapper instances.
//!
//! This module provides the `scalar_op_bitwise` macro which:
//! - Extracts native views (zero-copy for matching array dtype)
//! - Casts the scalar (i64 from FFI) to the array's dtype for the operation
//! - Performs element-wise scalar bitwise operation
//! - Only works with integer types and Bool (NOT Float types)
//!
//! Supported types: Int8, Int16, Int32, Int64, Uint8, Uint16, Uint32, Uint64, Bool
//! (Bool only for &, |, ^; use `no_bool` suffix for <<, >> which exclude Bool)
//!
//! Usage:
//! ```rust
//! let result = scalar_op_bitwise!(a_wrapper, a_meta, scalar, &);   // bitand
//! let result = scalar_op_bitwise!(a_wrapper, a_meta, scalar, |);   // bitor
//! let result = scalar_op_bitwise!(a_wrapper, a_meta, scalar, ^);   // bitxor
//! let result = scalar_op_bitwise!(a_wrapper, a_meta, scalar, <<, no_bool);  // left shift
//! let result = scalar_op_bitwise!(a_wrapper, a_meta, scalar, >>, no_bool);  // right shift
//! ```

/// Scalar bitwise operation (supports Bool for &, |, ^).
#[macro_export]
macro_rules! scalar_op_bitwise {
    ($wrapper:expr, $meta:expr, $scalar:expr, $op:tt) => {{
        $crate::scalar_op_bitwise_impl!($wrapper, $meta, $scalar, $op, true)
    }};
    ($wrapper:expr, $meta:expr, $scalar:expr, $op:tt, no_bool) => {{
        $crate::scalar_op_bitwise_impl!($wrapper, $meta, $scalar, $op, false)
    }};
}

/// Internal implementation with allow_bool flag.
#[macro_export]
macro_rules! scalar_op_bitwise_impl {
    ($wrapper:expr, $meta:expr, $scalar:expr, $op:tt, $allow_bool:expr) => {{
        use crate::helpers::{
            extract_view_bool, extract_view_i16, extract_view_i32, extract_view_i64,
            extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64,
            extract_view_u8,
            set_last_error, ERR_GENERIC,
        };
        use crate::types::dtype::DType;
        use crate::types::{ArrayData, NDArrayWrapper};

        match $wrapper.dtype {
            DType::Float64 | DType::Float32 => {
                set_last_error("Bitwise operations not supported for float types".to_string());
                return ERR_GENERIC;
            }
            DType::Bool => {
                if !$allow_bool {
                    set_last_error("Shift operations only supported for integer types".to_string());
                    return ERR_GENERIC;
                }
                let Some(view) = extract_view_bool($wrapper, $meta) else {
                    set_last_error("Failed to extract Bool view".to_string());
                    return ERR_GENERIC;
                };
                let s = if $scalar != 0 { 1u8 } else { 0u8 };
                let result = view.mapv(|x| x $op s);
                NDArrayWrapper {
                    data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                        result,
                    ))),
                    dtype: DType::Bool,
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
        }
    }};
}
