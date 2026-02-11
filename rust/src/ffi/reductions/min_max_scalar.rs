//! Scalar min/max reductions.

use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::reductions::helpers::{create_scalar_wrapper, extract_view};
use crate::ffi::NdArrayHandle;

/// Compute the minimum of all elements in the array (scalar reduction).
/// Returns a 0-dimensional array handle containing the minimum value.
#[no_mangle]
pub unsafe extern "C" fn ndarray_min(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    _strides: *const usize,
    ndim: usize,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || out_handle.is_null() || shape.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = std::slice::from_raw_parts(shape, ndim);

        let view = extract_view(wrapper, offset, shape_slice);

        // Get minimum - ndarray returns Option
        match view.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
            Some(&min_val) => {
                let result_wrapper = create_scalar_wrapper(min_val, wrapper.dtype);
                *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
                SUCCESS
            }
            None => {
                crate::error::set_last_error("Cannot compute min of empty array".to_string());
                ERR_GENERIC
            }
        }
    })
}

/// Compute the maximum of all elements in the array (scalar reduction).
/// Returns a 0-dimensional array handle containing the maximum value.
#[no_mangle]
pub unsafe extern "C" fn ndarray_max(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    _strides: *const usize,
    ndim: usize,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || out_handle.is_null() || shape.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = std::slice::from_raw_parts(shape, ndim);

        let view = extract_view(wrapper, offset, shape_slice);

        match view.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
            Some(&max_val) => {
                let result_wrapper = create_scalar_wrapper(max_val, wrapper.dtype);
                *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
                SUCCESS
            }
            None => {
                crate::error::set_last_error("Cannot compute max of empty array".to_string());
                ERR_GENERIC
            }
        }
    })
}
