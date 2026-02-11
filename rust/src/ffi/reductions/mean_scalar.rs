//! Scalar mean reduction.

use crate::dtype::DType;
use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::reductions::helpers::{create_scalar_wrapper, extract_view};
use crate::ffi::NdArrayHandle;

/// Compute the mean of all elements in the array (scalar reduction).
/// Returns a 0-dimensional array handle containing the mean value.
/// Mean is always computed in Float64 for precision.
#[no_mangle]
pub unsafe extern "C" fn ndarray_mean(
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

        // Compute mean - ndarray returns Option, default to 0.0 for empty
        let mean_val = view.mean().unwrap_or(0.0);

        // Mean should always be Float64 for precision
        let result_wrapper = create_scalar_wrapper(mean_val, DType::Float64);
        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));

        SUCCESS
    })
}
