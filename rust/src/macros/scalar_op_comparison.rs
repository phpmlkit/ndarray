//! Scalar comparison operation macro.
//!
//! Computes the promoted dtype from array and scalar dtypes (NumPy-style type promotion),
//! extracts the array as the promoted dtype using extract_view_as_* helpers,
//! dereferences the scalar pointer to the promoted dtype for the operation,
//! and performs element-wise comparison. Always returns Bool array.
//!
//! Pass `equality` for `==` / `!=` (includes complex dtypes, NumPy-compatible), or `ordering`
//! for `<`, `<=`, `>`, `>=` (complex dtypes are rejected, matching NumPy).
//!
//! Usage:
//! ```rust
//! let result = scalar_op_comparison!(a_wrapper, a_meta, scalar, scalar_dtype, ==, equality);
//! let result = scalar_op_comparison!(a_wrapper, a_meta, scalar, scalar_dtype, <, ordering);
//! ```

#[macro_export]
macro_rules! scalar_op_comparison {
    ($wrapper:expr, $meta:expr, $scalar:expr, $scalar_dtype:expr, $cmp_op:tt, equality) => {{
        use crate::helpers::{
            extract_view_as_bool, extract_view_as_c128, extract_view_as_c64, extract_view_as_f32,
            extract_view_as_f64, extract_view_as_i16, extract_view_as_i32, extract_view_as_i64,
            extract_view_as_i8, extract_view_as_u16, extract_view_as_u32, extract_view_as_u64,
            extract_view_as_u8, get_scalar_as_f64, get_scalar_as_f32, get_scalar_as_i64,
            get_scalar_as_i32, get_scalar_as_i16, get_scalar_as_i8, get_scalar_as_u64,
            get_scalar_as_u32, get_scalar_as_u16, get_scalar_as_u8, get_scalar_as_c64,
            get_scalar_as_c128, set_last_error, ERR_GENERIC,
        };
        use crate::types::dtype::DType;
        use crate::types::{ArrayData, NDArrayWrapper};

        let out_dtype = DType::promote($wrapper.dtype, $scalar_dtype);

        let result = match out_dtype {
            DType::Float64 => {
                let Some(view) = extract_view_as_f64($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Float64".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_f64($scalar, $scalar_dtype) };
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Float32 => {
                let Some(view) = extract_view_as_f32($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Float32".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_f32($scalar, $scalar_dtype) };
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Int64 => {
                let Some(view) = extract_view_as_i64($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Int64".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_i64($scalar, $scalar_dtype) };
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Int32 => {
                let Some(view) = extract_view_as_i32($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Int32".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_i32($scalar, $scalar_dtype) };
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Int16 => {
                let Some(view) = extract_view_as_i16($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Int16".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_i16($scalar, $scalar_dtype) };
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Int8 => {
                let Some(view) = extract_view_as_i8($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Int8".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_i8($scalar, $scalar_dtype) };
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Uint64 => {
                let Some(view) = extract_view_as_u64($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Uint64".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_u64($scalar, $scalar_dtype) };
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Uint32 => {
                let Some(view) = extract_view_as_u32($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Uint32".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_u32($scalar, $scalar_dtype) };
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Uint16 => {
                let Some(view) = extract_view_as_u16($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Uint16".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_u16($scalar, $scalar_dtype) };
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Uint8 => {
                let Some(view) = extract_view_as_u8($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Uint8".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_u8($scalar, $scalar_dtype) };
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Bool => {
                let Some(view) = extract_view_as_bool($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Bool".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_u8($scalar, $scalar_dtype) };
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Complex64 => {
                let Some(view) = extract_view_as_c64($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Complex64".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_c64($scalar, $scalar_dtype) };
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Complex128 => {
                let Some(view) = extract_view_as_c128($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Complex128".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_c128($scalar, $scalar_dtype) };
                view.mapv(|x| (x $cmp_op s) as u8)
            }
        };

        NDArrayWrapper {
            data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(result))),
            dtype: DType::Bool,
        }
    }};
    ($wrapper:expr, $meta:expr, $scalar:expr, $scalar_dtype:expr, $cmp_op:tt, ordering) => {{
        use crate::helpers::{
            extract_view_as_bool, extract_view_as_f32, extract_view_as_f64, extract_view_as_i16,
            extract_view_as_i32, extract_view_as_i64, extract_view_as_i8, extract_view_as_u16,
            extract_view_as_u32, extract_view_as_u64, extract_view_as_u8, get_scalar_as_f64,
            get_scalar_as_f32, get_scalar_as_i64, get_scalar_as_i32, get_scalar_as_i16,
            get_scalar_as_i8, get_scalar_as_u64, get_scalar_as_u32, get_scalar_as_u16,
            get_scalar_as_u8, set_last_error, ERR_GENERIC,
        };
        use crate::types::dtype::DType;
        use crate::types::{ArrayData, NDArrayWrapper};

        let out_dtype = DType::promote($wrapper.dtype, $scalar_dtype);

        let result = match out_dtype {
            DType::Float64 => {
                let Some(view) = extract_view_as_f64($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Float64".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_f64($scalar, $scalar_dtype) };
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Float32 => {
                let Some(view) = extract_view_as_f32($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Float32".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_f32($scalar, $scalar_dtype) };
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Int64 => {
                let Some(view) = extract_view_as_i64($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Int64".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_i64($scalar, $scalar_dtype) };
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Int32 => {
                let Some(view) = extract_view_as_i32($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Int32".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_i32($scalar, $scalar_dtype) };
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Int16 => {
                let Some(view) = extract_view_as_i16($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Int16".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_i16($scalar, $scalar_dtype) };
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Int8 => {
                let Some(view) = extract_view_as_i8($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Int8".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_i8($scalar, $scalar_dtype) };
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Uint64 => {
                let Some(view) = extract_view_as_u64($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Uint64".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_u64($scalar, $scalar_dtype) };
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Uint32 => {
                let Some(view) = extract_view_as_u32($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Uint32".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_u32($scalar, $scalar_dtype) };
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Uint16 => {
                let Some(view) = extract_view_as_u16($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Uint16".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_u16($scalar, $scalar_dtype) };
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Uint8 => {
                let Some(view) = extract_view_as_u8($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Uint8".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_u8($scalar, $scalar_dtype) };
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Bool => {
                let Some(view) = extract_view_as_bool($wrapper, $meta) else {
                    set_last_error("Failed to extract array as Bool".to_string());
                    return ERR_GENERIC;
                };
                let s = unsafe { get_scalar_as_u8($scalar, $scalar_dtype) };
                view.mapv(|x| (x $cmp_op s) as u8)
            }
            DType::Complex64 | DType::Complex128 => {
                set_last_error(
                    "ordering comparison (<, <=, >, >=) is not defined for complex dtypes"
                        .to_string(),
                );
                return ERR_GENERIC;
            }
        };

        NDArrayWrapper {
            data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(result))),
            dtype: DType::Bool,
        }
    }};
}
