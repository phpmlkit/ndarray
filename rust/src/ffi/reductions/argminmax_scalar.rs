//! Scalar argmin/argmax reductions (index of min/max).

use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::reductions::helpers::{create_scalar_wrapper_i64, extract_view};
use crate::ffi::NdArrayHandle;

/// Find the index of the minimum value (scalar - flattened index).
/// Returns a 0-dimensional array handle containing the index as Int64.
#[no_mangle]
pub unsafe extern "C" fn ndarray_argmin(
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

        // Find index of minimum
        match view
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        {
            Some((idx, _)) => {
                let result_wrapper = create_scalar_wrapper_i64(idx as i64);
                *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
                SUCCESS
            }
            None => {
                crate::error::set_last_error("Cannot compute argmin of empty array".to_string());
                ERR_GENERIC
            }
        }
    })
}

/// Find the index of the maximum value (scalar - flattened index).
/// Returns a 0-dimensional array handle containing the index as Int64.
#[no_mangle]
pub unsafe extern "C" fn ndarray_argmax(
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

        // Find index of maximum
        match view
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        {
            Some((idx, _)) => {
                let result_wrapper = create_scalar_wrapper_i64(idx as i64);
                *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
                SUCCESS
            }
            None => {
                crate::error::set_last_error("Cannot compute argmax of empty array".to_string());
                ERR_GENERIC
            }
        }
    })
}
