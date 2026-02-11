//! Element-level set operations.
//!
//! Provides type-specific set operations for single elements at a flat index.

use crate::core::NDArrayWrapper;
use crate::error::{self, ERR_DTYPE, ERR_GENERIC, ERR_INDEX, SUCCESS};
use crate::ffi::NdArrayHandle;

/// Helper for setting element values.
pub unsafe fn set_element_helper<T, F>(
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
