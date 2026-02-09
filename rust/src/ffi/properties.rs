//! Array property and lifecycle FFI functions.

use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::NdArrayHandle;
use std::slice;

// ============================================================================
// Array Destruction
// ============================================================================

/// Destroy an NDArray and free its memory.
#[no_mangle]
pub unsafe extern "C" fn ndarray_free(handle: *mut NdArrayHandle) -> i32 {
    crate::ffi_guard!({
        if !handle.is_null() {
            let _ = NdArrayHandle::into_wrapper(handle);
        }
        SUCCESS
    })
}

// ============================================================================
// Array Properties
// ============================================================================

/// Get the number of dimensions of an array.
#[no_mangle]
pub unsafe extern "C" fn ndarray_ndim(handle: *const NdArrayHandle, out_ndim: *mut usize) -> i32 {
    if handle.is_null() || out_ndim.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        *out_ndim = NdArrayHandle::as_wrapper(handle as *mut _).ndim();
        SUCCESS
    })
}

/// Get the total number of elements in an array.
#[no_mangle]
pub unsafe extern "C" fn ndarray_len(handle: *const NdArrayHandle, out_len: *mut usize) -> i32 {
    if handle.is_null() || out_len.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        *out_len = NdArrayHandle::as_wrapper(handle as *mut _).len();
        SUCCESS
    })
}

/// Get the dtype of an array.
#[no_mangle]
pub unsafe extern "C" fn ndarray_dtype(handle: *const NdArrayHandle, out_dtype: *mut u8) -> i32 {
    if handle.is_null() || out_dtype.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        *out_dtype = NdArrayHandle::as_wrapper(handle as *mut _).dtype.to_u8();
        SUCCESS
    })
}

/// Get the shape of an array.
///
/// Writes up to `max_ndim` dimensions to `out_shape` and returns the actual ndim in `out_ndim`.
#[no_mangle]
pub unsafe extern "C" fn ndarray_shape(
    handle: *const NdArrayHandle,
    out_shape: *mut usize,
    max_ndim: usize,
    out_ndim: *mut usize,
) -> i32 {
    if handle.is_null() || out_shape.is_null() || out_ndim.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape = wrapper.shape();
        let ndim = shape.len().min(max_ndim);

        let out_slice = slice::from_raw_parts_mut(out_shape, ndim);
        out_slice.copy_from_slice(&shape[..ndim]);

        *out_ndim = shape.len();
        SUCCESS
    })
}
