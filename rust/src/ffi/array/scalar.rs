//! Extract scalar value from 0-dimensional array.

use std::ffi::c_void;

use crate::core::view_helpers::{
    extract_view_bool, extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32,
    extract_view_i64, extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64,
    extract_view_u8,
};
use crate::dtype::DType;
use crate::error::{set_last_error, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::ffi::{NdArrayHandle, ViewMetadata};

/// Extract the scalar value from a 0-dimensional array or view.
///
/// The array must have ndim == 0.
/// Writes the value to out_value; the caller must allocate a buffer of at least
/// 8 bytes and interpret based on the array's dtype.
///
/// For bool: writes 0 or 1 to the first byte.
/// For numeric types: writes the native value (up to 8 bytes).
#[no_mangle]
pub unsafe extern "C" fn ndarray_scalar(
    handle: *const NdArrayHandle,
    meta: *const ViewMetadata,
    out_value: *mut c_void,
) -> i32 {
    if handle.is_null() || out_value.is_null() || meta.is_null() {
        return ERR_GENERIC;
    }

    let meta = &*meta;
    if meta.ndim != 0 {
        set_last_error("toScalar requires a 0-dimensional array".to_string());
        return ERR_SHAPE;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);

        let result = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, meta) else {
                    set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                view.first().map(|v| {
                    *(out_value as *mut f64) = *v;
                })
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, meta) else {
                    set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                view.first().map(|v| {
                    *(out_value as *mut f32) = *v;
                })
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, meta) else {
                    set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                view.first().map(|v| {
                    *(out_value as *mut i64) = *v;
                })
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, meta) else {
                    set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                view.first().map(|v| {
                    *(out_value as *mut i32) = *v;
                })
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, meta) else {
                    set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                view.first().map(|v| {
                    *(out_value as *mut i16) = *v;
                })
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, meta) else {
                    set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                view.first().map(|v| {
                    *(out_value as *mut i8) = *v;
                })
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, meta) else {
                    set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                view.first().map(|v| {
                    *(out_value as *mut u64) = *v;
                })
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, meta) else {
                    set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                view.first().map(|v| {
                    *(out_value as *mut u32) = *v;
                })
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, meta) else {
                    set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                view.first().map(|v| {
                    *(out_value as *mut u16) = *v;
                })
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, meta) else {
                    set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                view.first().map(|v| {
                    *(out_value as *mut u8) = *v;
                })
            }
            DType::Bool => {
                let Some(view) = extract_view_bool(wrapper, meta) else {
                    set_last_error("Failed to extract bool view".to_string());
                    return ERR_GENERIC;
                };
                view.first().map(|v| {
                    *(out_value as *mut u8) = *v;
                })
            }
        };

        match result {
            Some(()) => SUCCESS,
            None => {
                set_last_error("0-dimensional array has no element".to_string());
                ERR_GENERIC
            }
        }
    })
}
