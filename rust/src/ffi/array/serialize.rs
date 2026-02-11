//! JSON serialization FFI functions.

use ndarray::ArrayViewD;
use serde::Serialize;
use serde_json::Value;
use std::ffi::CString;
use std::os::raw::c_char;
use std::slice;

use crate::core::{ArrayData, NDArrayWrapper};
use crate::error::{self, ERR_GENERIC, SUCCESS};
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

// Helper to serialize array recursively
pub fn to_json<T, F>(view: ArrayViewD<T>, mapper: F) -> Value
where
    T: Clone,
    F: Fn(&T) -> Value + Copy,
{
    if view.ndim() == 0 {
        if let Some(val) = view.first() {
            mapper(val)
        } else {
            Value::Null
        }
    } else {
        let vec: Vec<Value> = view.outer_iter().map(|sub| to_json(sub, mapper)).collect();
        Value::Array(vec)
    }
}

// Helper to map serializable values to JSON Value
pub fn to_val<T: Serialize>(v: &T) -> Value {
    serde_json::to_value(v).unwrap_or(Value::Null)
}

// Helper to map boolean values (stored as u8) to JSON Value
pub fn bool_mapper(v: &u8) -> Value {
    Value::Bool(*v != 0)
}

/// Serialize a strided view of an NDArray to JSON.
///
/// This allows PHP to serialize views without creating copies. The offset,
/// shape, and strides define which elements to include.
#[no_mangle]
pub unsafe extern "C" fn ndarray_view_to_json(
    handle: *const NdArrayHandle,
    offset: usize,
    shape_ptr: *const usize,
    strides_ptr: *const usize,
    ndim: usize,
    precision: usize,
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

        let json_value = view_to_json_value(wrapper, offset, shape, strides);

        let _ = precision; // reserved for future float formatting

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

/// Recursively build a JSON value from a strided view.
fn view_to_json_value(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> serde_json::Value {
    use serde_json::Value;

    if shape.is_empty() {
        // Scalar: read single element at offset
        return scalar_to_json(wrapper, offset);
    }

    if shape.len() == 1 {
        // 1D: collect elements along stride
        let values: Vec<Value> = (0..shape[0])
            .map(|i| scalar_to_json(wrapper, offset + i * strides[0]))
            .collect();
        return Value::Array(values);
    }

    // Multi-dimensional: recurse
    let values: Vec<Value> = (0..shape[0])
        .map(|i| view_to_json_value(wrapper, offset + i * strides[0], &shape[1..], &strides[1..]))
        .collect();
    Value::Array(values)
}

/// Convert a single element at a flat index to a JSON value.
fn scalar_to_json(wrapper: &NDArrayWrapper, flat_index: usize) -> serde_json::Value {
    match &wrapper.data {
        ArrayData::Int8(arr) => {
            let guard = arr.read();
            guard
                .as_slice_memory_order()
                .and_then(|s| s.get(flat_index))
                .map(to_val)
                .unwrap_or(serde_json::Value::Null)
        }
        ArrayData::Int16(arr) => {
            let guard = arr.read();
            guard
                .as_slice_memory_order()
                .and_then(|s| s.get(flat_index))
                .map(to_val)
                .unwrap_or(serde_json::Value::Null)
        }
        ArrayData::Int32(arr) => {
            let guard = arr.read();
            guard
                .as_slice_memory_order()
                .and_then(|s| s.get(flat_index))
                .map(to_val)
                .unwrap_or(serde_json::Value::Null)
        }
        ArrayData::Int64(arr) => {
            let guard = arr.read();
            guard
                .as_slice_memory_order()
                .and_then(|s| s.get(flat_index))
                .map(to_val)
                .unwrap_or(serde_json::Value::Null)
        }
        ArrayData::Uint8(arr) => {
            let guard = arr.read();
            guard
                .as_slice_memory_order()
                .and_then(|s| s.get(flat_index))
                .map(to_val)
                .unwrap_or(serde_json::Value::Null)
        }
        ArrayData::Uint16(arr) => {
            let guard = arr.read();
            guard
                .as_slice_memory_order()
                .and_then(|s| s.get(flat_index))
                .map(to_val)
                .unwrap_or(serde_json::Value::Null)
        }
        ArrayData::Uint32(arr) => {
            let guard = arr.read();
            guard
                .as_slice_memory_order()
                .and_then(|s| s.get(flat_index))
                .map(to_val)
                .unwrap_or(serde_json::Value::Null)
        }
        ArrayData::Uint64(arr) => {
            let guard = arr.read();
            guard
                .as_slice_memory_order()
                .and_then(|s| s.get(flat_index))
                .map(to_val)
                .unwrap_or(serde_json::Value::Null)
        }
        ArrayData::Float32(arr) => {
            let guard = arr.read();
            guard
                .as_slice_memory_order()
                .and_then(|s| s.get(flat_index))
                .map(to_val)
                .unwrap_or(serde_json::Value::Null)
        }
        ArrayData::Float64(arr) => {
            let guard = arr.read();
            guard
                .as_slice_memory_order()
                .and_then(|s| s.get(flat_index))
                .map(to_val)
                .unwrap_or(serde_json::Value::Null)
        }
        ArrayData::Bool(arr) => {
            let guard = arr.read();
            guard
                .as_slice_memory_order()
                .and_then(|s| s.get(flat_index))
                .map(bool_mapper)
                .unwrap_or(serde_json::Value::Null)
        }
    }
}
