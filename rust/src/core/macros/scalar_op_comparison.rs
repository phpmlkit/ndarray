//! Scalar comparison operation macro.
//!
//! Extracts native views based on wrapper dtype (zero-copy), casts scalar (f64 from FFI)
//! to the array's type, and performs element-wise comparison. Always returns Bool array.
//!
//! Works with all numeric types and Bool (same as binary_op_comparison).
//!
//! Usage:
//! ```rust
//! let result = scalar_op_comparison!(a_wrapper, a_meta, scalar, ==);
//! ```

/// Scalar comparison operation macro.
#[macro_export]
macro_rules! scalar_op_comparison {
    ($wrapper:expr, $meta:expr, $scalar:expr, $cmp_op:tt) => {{
        use crate::core::{
            view_helpers::{
                extract_view_bool, extract_view_f32, extract_view_f64, extract_view_i16,
                extract_view_i32, extract_view_i64, extract_view_i8, extract_view_u16,
                extract_view_u32, extract_view_u64, extract_view_u8,
            },
            ArrayData, NDArrayWrapper,
        };
        use crate::core::dtype::DType;
        use crate::core::error::{set_last_error, ERR_GENERIC};

        let result = match $wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64($wrapper, $meta) else {
                    set_last_error("Failed to extract Float64 view".to_string());
                    return ERR_GENERIC;
                };
                let s = $scalar as f64;
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32($wrapper, $meta) else {
                    set_last_error("Failed to extract Float32 view".to_string());
                    return ERR_GENERIC;
                };
                let s = $scalar as f32;
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64($wrapper, $meta) else {
                    set_last_error("Failed to extract Int64 view".to_string());
                    return ERR_GENERIC;
                };
                let s = $scalar as i64;
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32($wrapper, $meta) else {
                    set_last_error("Failed to extract Int32 view".to_string());
                    return ERR_GENERIC;
                };
                let s = $scalar as i32;
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16($wrapper, $meta) else {
                    set_last_error("Failed to extract Int16 view".to_string());
                    return ERR_GENERIC;
                };
                let s = $scalar as i16;
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8($wrapper, $meta) else {
                    set_last_error("Failed to extract Int8 view".to_string());
                    return ERR_GENERIC;
                };
                let s = $scalar as i8;
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64($wrapper, $meta) else {
                    set_last_error("Failed to extract Uint64 view".to_string());
                    return ERR_GENERIC;
                };
                let s = $scalar as u64;
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32($wrapper, $meta) else {
                    set_last_error("Failed to extract Uint32 view".to_string());
                    return ERR_GENERIC;
                };
                let s = $scalar as u32;
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16($wrapper, $meta) else {
                    set_last_error("Failed to extract Uint16 view".to_string());
                    return ERR_GENERIC;
                };
                let s = $scalar as u16;
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8($wrapper, $meta) else {
                    set_last_error("Failed to extract Uint8 view".to_string());
                    return ERR_GENERIC;
                };
                let s = $scalar as u8;
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Bool => {
                let Some(view) = extract_view_bool($wrapper, $meta) else {
                    set_last_error("Failed to extract Bool view".to_string());
                    return ERR_GENERIC;
                };
                let s = if $scalar != 0.0 { 1u8 } else { 0u8 };
                view.mapv(|x| (x $cmp_op s) as u8)
            }
        };

        NDArrayWrapper {
            data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(result))),
            dtype: DType::Bool,
        }
    }};
}
