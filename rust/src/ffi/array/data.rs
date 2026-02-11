//! Array data access FFI functions.

use crate::core::NDArrayWrapper;
use crate::error::{self, ERR_DTYPE, ERR_GENERIC, SUCCESS};
use crate::ffi::NdArrayHandle;
use std::slice;

// ============================================================================
// Array Data Access Functions
// ============================================================================
// Note: These accessors return an integer status code.

unsafe fn get_data_helper<T, F>(
    handle: *const NdArrayHandle,
    out_data: *mut T,
    max_len: usize,
    out_len: *mut usize,
    method: F,
) -> i32
where
    T: Copy,
    F: Fn(&NDArrayWrapper) -> Option<Vec<T>> + std::panic::UnwindSafe,
{
    if handle.is_null() || out_data.is_null() || out_len.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        if let Some(data) = method(wrapper) {
            let len = data.len().min(max_len);
            let out_slice = slice::from_raw_parts_mut(out_data, len);
            out_slice.copy_from_slice(&data[..len]);
            *out_len = data.len();
            SUCCESS
        } else {
            error::set_last_error("Data type mismatch");
            ERR_DTYPE
        }
    })
}

/// Get int8 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_data_int8(
    handle: *const NdArrayHandle,
    out_data: *mut i8,
    max_len: usize,
    out_len: *mut usize,
) -> i32 {
    get_data_helper(handle, out_data, max_len, out_len, |w| w.to_int8_vec())
}

/// Get int16 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_data_int16(
    handle: *const NdArrayHandle,
    out_data: *mut i16,
    max_len: usize,
    out_len: *mut usize,
) -> i32 {
    get_data_helper(handle, out_data, max_len, out_len, |w| w.to_int16_vec())
}

/// Get int32 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_data_int32(
    handle: *const NdArrayHandle,
    out_data: *mut i32,
    max_len: usize,
    out_len: *mut usize,
) -> i32 {
    get_data_helper(handle, out_data, max_len, out_len, |w| w.to_int32_vec())
}

/// Get int64 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_data_int64(
    handle: *const NdArrayHandle,
    out_data: *mut i64,
    max_len: usize,
    out_len: *mut usize,
) -> i32 {
    get_data_helper(handle, out_data, max_len, out_len, |w| w.to_int64_vec())
}

/// Get uint8 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_data_uint8(
    handle: *const NdArrayHandle,
    out_data: *mut u8,
    max_len: usize,
    out_len: *mut usize,
) -> i32 {
    get_data_helper(handle, out_data, max_len, out_len, |w| w.to_uint8_vec())
}

/// Get uint16 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_data_uint16(
    handle: *const NdArrayHandle,
    out_data: *mut u16,
    max_len: usize,
    out_len: *mut usize,
) -> i32 {
    get_data_helper(handle, out_data, max_len, out_len, |w| w.to_uint16_vec())
}

/// Get uint32 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_data_uint32(
    handle: *const NdArrayHandle,
    out_data: *mut u32,
    max_len: usize,
    out_len: *mut usize,
) -> i32 {
    get_data_helper(handle, out_data, max_len, out_len, |w| w.to_uint32_vec())
}

/// Get uint64 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_data_uint64(
    handle: *const NdArrayHandle,
    out_data: *mut u64,
    max_len: usize,
    out_len: *mut usize,
) -> i32 {
    get_data_helper(handle, out_data, max_len, out_len, |w| w.to_uint64_vec())
}

/// Get float32 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_data_float32(
    handle: *const NdArrayHandle,
    out_data: *mut f32,
    max_len: usize,
    out_len: *mut usize,
) -> i32 {
    get_data_helper(handle, out_data, max_len, out_len, |w| w.to_float32_vec())
}

/// Get float64 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_data_float64(
    handle: *const NdArrayHandle,
    out_data: *mut f64,
    max_len: usize,
    out_len: *mut usize,
) -> i32 {
    get_data_helper(handle, out_data, max_len, out_len, |w| w.to_float64_vec())
}

/// Get bool data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_data_bool(
    handle: *const NdArrayHandle,
    out_data: *mut u8,
    max_len: usize,
    out_len: *mut usize,
) -> i32 {
    get_data_helper(handle, out_data, max_len, out_len, |w| w.to_bool_vec())
}
