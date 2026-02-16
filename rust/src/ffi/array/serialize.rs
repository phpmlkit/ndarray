//! JSON serialization FFI functions.
//!
//! Provides optimized serialization of NDArrays and views to JSON format.
//! Uses ndarray's view extraction and iteration for maximum performance.

use ndarray::ArrayViewD;
use serde_json::Value;
use std::ffi::CString;
use std::os::raw::c_char;
use std::slice;

use crate::core::view_helpers::{
    extract_view_bool, extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32,
    extract_view_i64, extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64,
    extract_view_u8,
};
use crate::core::ArrayData;
use crate::error::{self, ERR_GENERIC, SUCCESS};
use crate::ffi::NdArrayHandle;

/// Serialize an NDArray or view to a nested JSON string.
///
/// The output string must be freed using `ndarray_free_string`.
#[no_mangle]
pub unsafe extern "C" fn ndarray_to_json(
    handle: *const NdArrayHandle,
    offset: usize,
    shape_ptr: *const usize,
    strides_ptr: *const usize,
    ndim: usize,
    out_ptr: *mut *mut c_char,
    out_len: *mut usize,
) -> i32 {
    if handle.is_null()
        || shape_ptr.is_null()
        || strides_ptr.is_null()
        || out_ptr.is_null()
        || out_len.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape = slice::from_raw_parts(shape_ptr, ndim);
        let strides = slice::from_raw_parts(strides_ptr, ndim);

        let json_value = match &wrapper.data {
            ArrayData::Float64(_) => {
                let Some(view) = extract_view_f64(wrapper, offset, shape, strides) else {
                    error::set_last_error("Failed to extract f64 view");
                    return ERR_GENERIC;
                };
                view_to_json(&view, |v| serde_json::to_value(v).unwrap_or(Value::Null))
            }
            ArrayData::Float32(_) => {
                let Some(view) = extract_view_f32(wrapper, offset, shape, strides) else {
                    error::set_last_error("Failed to extract f32 view");
                    return ERR_GENERIC;
                };
                view_to_json(&view, |v| serde_json::to_value(v).unwrap_or(Value::Null))
            }
            ArrayData::Int64(_) => {
                let Some(view) = extract_view_i64(wrapper, offset, shape, strides) else {
                    error::set_last_error("Failed to extract i64 view");
                    return ERR_GENERIC;
                };
                view_to_json(&view, |v| serde_json::to_value(v).unwrap_or(Value::Null))
            }
            ArrayData::Int32(_) => {
                let Some(view) = extract_view_i32(wrapper, offset, shape, strides) else {
                    error::set_last_error("Failed to extract i32 view");
                    return ERR_GENERIC;
                };
                view_to_json(&view, |v| serde_json::to_value(v).unwrap_or(Value::Null))
            }
            ArrayData::Int16(_) => {
                let Some(view) = extract_view_i16(wrapper, offset, shape, strides) else {
                    error::set_last_error("Failed to extract i16 view");
                    return ERR_GENERIC;
                };
                view_to_json(&view, |v| serde_json::to_value(v).unwrap_or(Value::Null))
            }
            ArrayData::Int8(_) => {
                let Some(view) = extract_view_i8(wrapper, offset, shape, strides) else {
                    error::set_last_error("Failed to extract i8 view");
                    return ERR_GENERIC;
                };
                view_to_json(&view, |v| serde_json::to_value(v).unwrap_or(Value::Null))
            }
            ArrayData::Uint64(_) => {
                let Some(view) = extract_view_u64(wrapper, offset, shape, strides) else {
                    error::set_last_error("Failed to extract u64 view");
                    return ERR_GENERIC;
                };
                view_to_json(&view, |v| serde_json::to_value(v).unwrap_or(Value::Null))
            }
            ArrayData::Uint32(_) => {
                let Some(view) = extract_view_u32(wrapper, offset, shape, strides) else {
                    error::set_last_error("Failed to extract u32 view");
                    return ERR_GENERIC;
                };
                view_to_json(&view, |v| serde_json::to_value(v).unwrap_or(Value::Null))
            }
            ArrayData::Uint16(_) => {
                let Some(view) = extract_view_u16(wrapper, offset, shape, strides) else {
                    error::set_last_error("Failed to extract u16 view");
                    return ERR_GENERIC;
                };
                view_to_json(&view, |v| serde_json::to_value(v).unwrap_or(Value::Null))
            }
            ArrayData::Uint8(_) => {
                let Some(view) = extract_view_u8(wrapper, offset, shape, strides) else {
                    error::set_last_error("Failed to extract u8 view");
                    return ERR_GENERIC;
                };
                view_to_json(&view, |v| serde_json::to_value(v).unwrap_or(Value::Null))
            }
            ArrayData::Bool(_) => {
                let Some(view) = extract_view_bool(wrapper, offset, shape, strides) else {
                    error::set_last_error("Failed to extract bool view");
                    return ERR_GENERIC;
                };
                view_to_json(&view, |v| Value::Bool(*v != 0))
            }
        };

        match serde_json::to_string(&json_value) {
            Ok(s) => match CString::new(s) {
                Ok(c_str) => {
                    *out_len = c_str.as_bytes().len();
                    *out_ptr = c_str.into_raw();
                    SUCCESS
                }
                Err(_) => {
                    error::set_last_error("Failed to convert JSON to CString (null byte found)");
                    ERR_GENERIC
                }
            },
            Err(e) => {
                error::set_last_error(e);
                ERR_GENERIC
            }
        }
    })
}

/// Free a string allocated by Rust.
///
/// # Safety
/// The pointer must be valid and allocated by Rust.
#[no_mangle]
pub unsafe extern "C" fn ndarray_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        let _ = CString::from_raw(ptr);
    }
}

/// Recursively convert an ndarray view to JSON Value.
///
fn view_to_json<T, F>(view: &ArrayViewD<T>, mapper: F) -> Value
where
    T: Clone,
    F: Fn(&T) -> Value + Copy,
{
    if view.ndim() == 0 {
        view.first().map(mapper).unwrap_or(Value::Null)
    } else {
        let values: Vec<Value> = view
            .outer_iter()
            .map(|subview| view_to_json(&subview, mapper))
            .collect();
        Value::Array(values)
    }
}
