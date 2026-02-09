//! JSON serialization FFI functions.

use std::ffi::CString;
use std::os::raw::c_char;

use crate::core::ArrayData;
use crate::error::{self, ERR_GENERIC, SUCCESS};
use crate::ffi::serialize_helpers::{bool_mapper, to_json, to_val};
use crate::ffi::NdArrayHandle;

// ============================================================================
// Serialization
// ============================================================================

/// Serialize an NDArray to a nested JSON string.
///
/// This moves the complexity of unflattening nested arrays from PHP to Rust.
/// Returns status code.
/// The output string must be freed using `ndarray_free_string`.
#[no_mangle]
pub unsafe extern "C" fn ndarray_to_json(
    handle: *const NdArrayHandle,
    out_ptr: *mut *mut c_char,
    out_len: *mut usize,
    _max_precision: usize,
) -> i32 {
    if handle.is_null() || out_ptr.is_null() || out_len.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);

        let json_value = match &wrapper.data {
            ArrayData::Int8(arr) => to_json(arr.read().view(), to_val),
            ArrayData::Int16(arr) => to_json(arr.read().view(), to_val),
            ArrayData::Int32(arr) => to_json(arr.read().view(), to_val),
            ArrayData::Int64(arr) => to_json(arr.read().view(), to_val),
            ArrayData::Uint8(arr) => to_json(arr.read().view(), to_val),
            ArrayData::Uint16(arr) => to_json(arr.read().view(), to_val),
            ArrayData::Uint32(arr) => to_json(arr.read().view(), to_val),
            ArrayData::Uint64(arr) => to_json(arr.read().view(), to_val),
            ArrayData::Float32(arr) => to_json(arr.read().view(), to_val),
            ArrayData::Float64(arr) => to_json(arr.read().view(), to_val),
            ArrayData::Bool(arr) => to_json(arr.read().view(), bool_mapper),
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

/// Free a string allocated by `ndarray_to_json`.
///
/// # Safety
/// The pointer must be valid and allocated by `ndarray_to_json`.
#[no_mangle]
pub unsafe extern "C" fn ndarray_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        let _ = CString::from_raw(ptr);
    }
}
