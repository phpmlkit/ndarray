//! Element-level indexing FFI functions.
//!
//! Provides type-specific get/set operations for single elements at a flat index.
//! Also provides strided view serialization to JSON.

use std::ffi::CString;
use std::os::raw::c_char;
use std::slice;

use crate::core::{ArrayData, NDArrayWrapper};
use crate::error::{self, ERR_DTYPE, ERR_GENERIC, ERR_INDEX, SUCCESS};
use crate::ffi::serialize_helpers::{bool_mapper, to_val};
use crate::ffi::NdArrayHandle;

// ============================================================================
// Element Get Helpers
// ============================================================================

unsafe fn get_element_helper<T, F>(
    handle: *const NdArrayHandle,
    flat_index: usize,
    out_value: *mut T,
    method: F,
) -> i32
where
    T: Copy,
    F: Fn(&NDArrayWrapper, usize) -> Result<T, String> + std::panic::UnwindSafe,
{
    if handle.is_null() || out_value.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        match method(wrapper, flat_index) {
            Ok(value) => {
                *out_value = value;
                SUCCESS
            }
            Err(e) => {
                if e.contains("Type mismatch") {
                    error::set_last_error(e);
                    ERR_DTYPE
                } else {
                    error::set_last_error(e);
                    ERR_INDEX
                }
            }
        }
    })
}

// ============================================================================
// Element Set Helpers
// ============================================================================

unsafe fn set_element_helper<T, F>(
    handle: *mut NdArrayHandle,
    flat_index: usize,
    value: T,
    method: F,
) -> i32
where
    T: Copy,
    F: Fn(&NDArrayWrapper, usize, T) -> Result<(), String> + std::panic::UnwindSafe,
{
    if handle.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle);
        match method(wrapper, flat_index, value) {
            Ok(()) => SUCCESS,
            Err(e) => {
                if e.contains("Type mismatch") {
                    error::set_last_error(e);
                    ERR_DTYPE
                } else {
                    error::set_last_error(e);
                    ERR_INDEX
                }
            }
        }
    })
}

// ============================================================================
// Fill Slice Helpers
// ============================================================================

unsafe fn fill_slice_helper<T, F>(
    handle: *const NdArrayHandle,
    value: T,
    offset: usize,
    shape_ptr: *const usize,
    strides_ptr: *const usize,
    ndim: usize,
    method: F,
) -> i32
where
    T: Copy,
    F: Fn(&NDArrayWrapper, T, usize, &[usize], &[usize]) -> Result<(), String>
        + std::panic::UnwindSafe,
{
    if handle.is_null() || shape_ptr.is_null() || strides_ptr.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape = slice::from_raw_parts(shape_ptr, ndim);
        let strides = slice::from_raw_parts(strides_ptr, ndim);

        match method(wrapper, value, offset, shape, strides) {
            Ok(()) => SUCCESS,
            Err(e) => {
                error::set_last_error(e);
                // Could be dtype or generic error, assuming generic for shape/bounds issues
                ERR_GENERIC
            }
        }
    })
}

// ============================================================================
// Get Element Functions (11 types)
// ============================================================================

/// Get an int8 element at the given flat index.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_element_int8(
    handle: *const NdArrayHandle,
    flat_index: usize,
    out_value: *mut i8,
) -> i32 {
    get_element_helper(handle, flat_index, out_value, |w, i| w.get_element_i8(i))
}

/// Get an int16 element at the given flat index.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_element_int16(
    handle: *const NdArrayHandle,
    flat_index: usize,
    out_value: *mut i16,
) -> i32 {
    get_element_helper(handle, flat_index, out_value, |w, i| w.get_element_i16(i))
}

/// Get an int32 element at the given flat index.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_element_int32(
    handle: *const NdArrayHandle,
    flat_index: usize,
    out_value: *mut i32,
) -> i32 {
    get_element_helper(handle, flat_index, out_value, |w, i| w.get_element_i32(i))
}

/// Get an int64 element at the given flat index.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_element_int64(
    handle: *const NdArrayHandle,
    flat_index: usize,
    out_value: *mut i64,
) -> i32 {
    get_element_helper(handle, flat_index, out_value, |w, i| w.get_element_i64(i))
}

/// Get a uint8 element at the given flat index.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_element_uint8(
    handle: *const NdArrayHandle,
    flat_index: usize,
    out_value: *mut u8,
) -> i32 {
    get_element_helper(handle, flat_index, out_value, |w, i| w.get_element_u8(i))
}

/// Get a uint16 element at the given flat index.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_element_uint16(
    handle: *const NdArrayHandle,
    flat_index: usize,
    out_value: *mut u16,
) -> i32 {
    get_element_helper(handle, flat_index, out_value, |w, i| w.get_element_u16(i))
}

/// Get a uint32 element at the given flat index.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_element_uint32(
    handle: *const NdArrayHandle,
    flat_index: usize,
    out_value: *mut u32,
) -> i32 {
    get_element_helper(handle, flat_index, out_value, |w, i| w.get_element_u32(i))
}

/// Get a uint64 element at the given flat index.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_element_uint64(
    handle: *const NdArrayHandle,
    flat_index: usize,
    out_value: *mut u64,
) -> i32 {
    get_element_helper(handle, flat_index, out_value, |w, i| w.get_element_u64(i))
}

/// Get a float32 element at the given flat index.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_element_float32(
    handle: *const NdArrayHandle,
    flat_index: usize,
    out_value: *mut f32,
) -> i32 {
    get_element_helper(handle, flat_index, out_value, |w, i| w.get_element_f32(i))
}

/// Get a float64 element at the given flat index.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_element_float64(
    handle: *const NdArrayHandle,
    flat_index: usize,
    out_value: *mut f64,
) -> i32 {
    get_element_helper(handle, flat_index, out_value, |w, i| w.get_element_f64(i))
}

/// Get a bool element at the given flat index (stored as u8).
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_element_bool(
    handle: *const NdArrayHandle,
    flat_index: usize,
    out_value: *mut u8,
) -> i32 {
    get_element_helper(handle, flat_index, out_value, |w, i| w.get_element_bool(i))
}

// ============================================================================
// Set Element Functions (11 types)
// ============================================================================

/// Set an int8 element at the given flat index.
#[no_mangle]
pub unsafe extern "C" fn ndarray_set_element_int8(
    handle: *mut NdArrayHandle,
    flat_index: usize,
    value: i8,
) -> i32 {
    set_element_helper(handle, flat_index, value, |w, i, v| w.set_element_i8(i, v))
}

/// Set an int16 element at the given flat index.
#[no_mangle]
pub unsafe extern "C" fn ndarray_set_element_int16(
    handle: *mut NdArrayHandle,
    flat_index: usize,
    value: i16,
) -> i32 {
    set_element_helper(handle, flat_index, value, |w, i, v| w.set_element_i16(i, v))
}

/// Set an int32 element at the given flat index.
#[no_mangle]
pub unsafe extern "C" fn ndarray_set_element_int32(
    handle: *mut NdArrayHandle,
    flat_index: usize,
    value: i32,
) -> i32 {
    set_element_helper(handle, flat_index, value, |w, i, v| w.set_element_i32(i, v))
}

/// Set an int64 element at the given flat index.
#[no_mangle]
pub unsafe extern "C" fn ndarray_set_element_int64(
    handle: *mut NdArrayHandle,
    flat_index: usize,
    value: i64,
) -> i32 {
    set_element_helper(handle, flat_index, value, |w, i, v| w.set_element_i64(i, v))
}

/// Set a uint8 element at the given flat index.
#[no_mangle]
pub unsafe extern "C" fn ndarray_set_element_uint8(
    handle: *mut NdArrayHandle,
    flat_index: usize,
    value: u8,
) -> i32 {
    set_element_helper(handle, flat_index, value, |w, i, v| w.set_element_u8(i, v))
}

/// Set a uint16 element at the given flat index.
#[no_mangle]
pub unsafe extern "C" fn ndarray_set_element_uint16(
    handle: *mut NdArrayHandle,
    flat_index: usize,
    value: u16,
) -> i32 {
    set_element_helper(handle, flat_index, value, |w, i, v| w.set_element_u16(i, v))
}

/// Set a uint32 element at the given flat index.
#[no_mangle]
pub unsafe extern "C" fn ndarray_set_element_uint32(
    handle: *mut NdArrayHandle,
    flat_index: usize,
    value: u32,
) -> i32 {
    set_element_helper(handle, flat_index, value, |w, i, v| w.set_element_u32(i, v))
}

/// Set a uint64 element at the given flat index.
#[no_mangle]
pub unsafe extern "C" fn ndarray_set_element_uint64(
    handle: *mut NdArrayHandle,
    flat_index: usize,
    value: u64,
) -> i32 {
    set_element_helper(handle, flat_index, value, |w, i, v| w.set_element_u64(i, v))
}

/// Set a float32 element at the given flat index.
#[no_mangle]
pub unsafe extern "C" fn ndarray_set_element_float32(
    handle: *mut NdArrayHandle,
    flat_index: usize,
    value: f32,
) -> i32 {
    set_element_helper(handle, flat_index, value, |w, i, v| w.set_element_f32(i, v))
}

/// Set a float64 element at the given flat index.
#[no_mangle]
pub unsafe extern "C" fn ndarray_set_element_float64(
    handle: *mut NdArrayHandle,
    flat_index: usize,
    value: f64,
) -> i32 {
    set_element_helper(handle, flat_index, value, |w, i, v| w.set_element_f64(i, v))
}

/// Set a bool element at the given flat index (stored as u8).
#[no_mangle]
pub unsafe extern "C" fn ndarray_set_element_bool(
    handle: *mut NdArrayHandle,
    flat_index: usize,
    value: u8,
) -> i32 {
    set_element_helper(handle, flat_index, value, |w, i, v| {
        w.set_element_bool(i, v)
    })
}

// ============================================================================
// Fill Slice Functions (11 types)
// ============================================================================

/// Fill a slice with an int8 value.
#[no_mangle]
pub unsafe extern "C" fn ndarray_fill_int8(
    handle: *const NdArrayHandle,
    value: i8,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
) -> i32 {
    fill_slice_helper(
        handle,
        value,
        offset,
        shape,
        strides,
        ndim,
        |w, v, o, s, st| w.fill_slice_i8(v, o, s, st),
    )
}

/// Fill a slice with an int16 value.
#[no_mangle]
pub unsafe extern "C" fn ndarray_fill_int16(
    handle: *const NdArrayHandle,
    value: i16,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
) -> i32 {
    fill_slice_helper(
        handle,
        value,
        offset,
        shape,
        strides,
        ndim,
        |w, v, o, s, st| w.fill_slice_i16(v, o, s, st),
    )
}

/// Fill a slice with an int32 value.
#[no_mangle]
pub unsafe extern "C" fn ndarray_fill_int32(
    handle: *const NdArrayHandle,
    value: i32,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
) -> i32 {
    fill_slice_helper(
        handle,
        value,
        offset,
        shape,
        strides,
        ndim,
        |w, v, o, s, st| w.fill_slice_i32(v, o, s, st),
    )
}

/// Fill a slice with an int64 value.
#[no_mangle]
pub unsafe extern "C" fn ndarray_fill_int64(
    handle: *const NdArrayHandle,
    value: i64,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
) -> i32 {
    fill_slice_helper(
        handle,
        value,
        offset,
        shape,
        strides,
        ndim,
        |w, v, o, s, st| w.fill_slice_i64(v, o, s, st),
    )
}

/// Fill a slice with a uint8 value.
#[no_mangle]
pub unsafe extern "C" fn ndarray_fill_uint8(
    handle: *const NdArrayHandle,
    value: u8,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
) -> i32 {
    fill_slice_helper(
        handle,
        value,
        offset,
        shape,
        strides,
        ndim,
        |w, v, o, s, st| w.fill_slice_u8(v, o, s, st),
    )
}

/// Fill a slice with a uint16 value.
#[no_mangle]
pub unsafe extern "C" fn ndarray_fill_uint16(
    handle: *const NdArrayHandle,
    value: u16,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
) -> i32 {
    fill_slice_helper(
        handle,
        value,
        offset,
        shape,
        strides,
        ndim,
        |w, v, o, s, st| w.fill_slice_u16(v, o, s, st),
    )
}

/// Fill a slice with a uint32 value.
#[no_mangle]
pub unsafe extern "C" fn ndarray_fill_uint32(
    handle: *const NdArrayHandle,
    value: u32,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
) -> i32 {
    fill_slice_helper(
        handle,
        value,
        offset,
        shape,
        strides,
        ndim,
        |w, v, o, s, st| w.fill_slice_u32(v, o, s, st),
    )
}

/// Fill a slice with a uint64 value.
#[no_mangle]
pub unsafe extern "C" fn ndarray_fill_uint64(
    handle: *const NdArrayHandle,
    value: u64,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
) -> i32 {
    fill_slice_helper(
        handle,
        value,
        offset,
        shape,
        strides,
        ndim,
        |w, v, o, s, st| w.fill_slice_u64(v, o, s, st),
    )
}

/// Fill a slice with a float32 value.
#[no_mangle]
pub unsafe extern "C" fn ndarray_fill_float32(
    handle: *const NdArrayHandle,
    value: f32,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
) -> i32 {
    fill_slice_helper(
        handle,
        value,
        offset,
        shape,
        strides,
        ndim,
        |w, v, o, s, st| w.fill_slice_f32(v, o, s, st),
    )
}

/// Fill a slice with a float64 value.
#[no_mangle]
pub unsafe extern "C" fn ndarray_fill_float64(
    handle: *const NdArrayHandle,
    value: f64,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
) -> i32 {
    fill_slice_helper(
        handle,
        value,
        offset,
        shape,
        strides,
        ndim,
        |w, v, o, s, st| w.fill_slice_f64(v, o, s, st),
    )
}

/// Fill a slice with a bool value (passed as u8).
#[no_mangle]
pub unsafe extern "C" fn ndarray_fill_bool(
    handle: *const NdArrayHandle,
    value: u8,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
) -> i32 {
    fill_slice_helper(
        handle,
        value,
        offset,
        shape,
        strides,
        ndim,
        |w, v, o, s, st| w.fill_slice_bool(v, o, s, st),
    )
}

// ============================================================================
// View JSON Serialization
// ============================================================================

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
