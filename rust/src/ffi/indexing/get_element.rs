//! Element-level get operations.
//!
//! Provides type-specific get operations for single elements at a flat index.

use crate::core::NDArrayWrapper;
use crate::error::{self, ERR_DTYPE, ERR_GENERIC, ERR_INDEX, SUCCESS};
use crate::ffi::NdArrayHandle;

/// Helper for getting element values.
pub unsafe fn get_element_helper<T, F>(
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
