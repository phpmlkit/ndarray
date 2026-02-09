//! Array property and lifecycle FFI functions.

use crate::ffi::NdArrayHandle;
use std::slice;

// ============================================================================
// Array Destruction
// ============================================================================

/// Destroy an NDArray and free its memory.
///
/// # Safety
/// The pointer must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn ndarray_free(handle: *mut NdArrayHandle) {
    if !handle.is_null() {
        let _ = NdArrayHandle::into_wrapper(handle);
    }
}

// ============================================================================
// Array Properties
// ============================================================================

/// Get the number of dimensions of an array.
#[no_mangle]
pub unsafe extern "C" fn ndarray_ndim(handle: *const NdArrayHandle) -> usize {
    if handle.is_null() {
        return 0;
    }
    NdArrayHandle::as_wrapper(handle as *mut _).ndim()
}

/// Get the total number of elements in an array.
#[no_mangle]
pub unsafe extern "C" fn ndarray_len(handle: *const NdArrayHandle) -> usize {
    if handle.is_null() {
        return 0;
    }
    NdArrayHandle::as_wrapper(handle as *mut _).len()
}

/// Get the dtype of an array.
#[no_mangle]
pub unsafe extern "C" fn ndarray_dtype(handle: *const NdArrayHandle) -> u8 {
    if handle.is_null() {
        return 255; // Invalid
    }
    NdArrayHandle::as_wrapper(handle as *mut _).dtype.to_u8()
}

/// Get the shape of an array.
///
/// Writes up to `max_ndim` dimensions to `out_shape` and returns the actual ndim.
#[no_mangle]
pub unsafe extern "C" fn ndarray_shape(
    handle: *const NdArrayHandle,
    out_shape: *mut usize,
    max_ndim: usize,
) -> usize {
    if handle.is_null() || out_shape.is_null() {
        return 0;
    }

    let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
    let shape = wrapper.shape();
    let ndim = shape.len().min(max_ndim);

    let out_slice = slice::from_raw_parts_mut(out_shape, ndim);
    out_slice.copy_from_slice(&shape[..ndim]);

    shape.len()
}
