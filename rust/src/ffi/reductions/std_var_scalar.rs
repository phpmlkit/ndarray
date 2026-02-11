//! Scalar variance and standard deviation reductions.

use crate::dtype::DType;
use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::reductions::helpers::{create_scalar_wrapper, extract_view};
use crate::ffi::NdArrayHandle;

/// Compute the variance of all elements (scalar reduction).
/// ddof: Delta degrees of freedom (0 for population, 1 for sample).
/// Returns a 0-dimensional array handle containing the variance as Float64.
#[no_mangle]
pub unsafe extern "C" fn ndarray_var(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    _strides: *const usize,
    ndim: usize,
    ddof: f64,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || out_handle.is_null() || shape.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = std::slice::from_raw_parts(shape, ndim);

        let view = extract_view(wrapper, offset, shape_slice);
        let n = view.len() as f64;

        if n <= ddof {
            crate::error::set_last_error(format!(
                "ddof ({}) must be less than number of elements ({})",
                ddof, n as usize
            ));
            return ERR_GENERIC;
        }

        // Compute variance using ndarray's var method
        let var_val = view.var(ddof);

        // Variance should be Float64
        let result_wrapper = create_scalar_wrapper(var_val, DType::Float64);
        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));

        SUCCESS
    })
}

/// Compute the standard deviation of all elements (scalar reduction).
/// ddof: Delta degrees of freedom (0 for population, 1 for sample).
/// Returns a 0-dimensional array handle containing the std as Float64.
#[no_mangle]
pub unsafe extern "C" fn ndarray_std(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    _strides: *const usize,
    ndim: usize,
    ddof: f64,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || out_handle.is_null() || shape.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = std::slice::from_raw_parts(shape, ndim);

        let view = extract_view(wrapper, offset, shape_slice);
        let n = view.len() as f64;

        if n <= ddof {
            crate::error::set_last_error(format!(
                "ddof ({}) must be less than number of elements ({})",
                ddof, n as usize
            ));
            return ERR_GENERIC;
        }

        // Compute std using ndarray's std method
        let std_val = view.std(ddof);

        // Std should be Float64
        let result_wrapper = create_scalar_wrapper(std_val, DType::Float64);
        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));

        SUCCESS
    })
}
