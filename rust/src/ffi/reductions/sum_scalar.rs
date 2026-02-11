//! Scalar sum reduction.

use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::reductions::helpers::{create_scalar_wrapper, extract_view};
use crate::ffi::NdArrayHandle;

/// Compute the sum of all elements in the array (scalar reduction).
/// Returns a 0-dimensional array handle containing the sum value.
#[no_mangle]
pub unsafe extern "C" fn ndarray_sum(
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

        // Extract view data
        let view = extract_view(wrapper, offset, shape_slice);

        // Compute sum using ndarray's built-in sum
        let sum_val = view.sum();

        // Create scalar result wrapper with same dtype as input
        let result_wrapper = create_scalar_wrapper(sum_val, wrapper.dtype);
        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));

        SUCCESS
    })
}
