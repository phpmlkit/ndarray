//! Axis mean reduction.

use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::reductions::helpers::{compute_axis_output_shape, extract_view, validate_axis};
use crate::ffi::NdArrayHandle;
use ndarray::{ArrayD, Axis, IxDyn};
use parking_lot::RwLock;
use std::sync::Arc;

/// Mean along axis.
#[no_mangle]
pub unsafe extern "C" fn ndarray_mean_axis(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    _strides: *const usize,
    ndim: usize,
    axis: i32,
    keepdims: bool,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || out_handle.is_null() || shape.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = std::slice::from_raw_parts(shape, ndim);

        // Validate axis
        let axis_usize = match validate_axis(shape_slice, axis) {
            Ok(a) => a,
            Err(e) => {
                crate::error::set_last_error(e);
                return ERR_GENERIC;
            }
        };

        // Extract view and compute mean along axis
        let view = extract_view(wrapper, offset, shape_slice);

        // ndarray's mean_axis returns Option, handle empty case
        let result_arr = match view.mean_axis(Axis(axis_usize)) {
            Some(arr) => arr,
            None => {
                // Empty axis - create array of zeros with appropriate shape
                let out_shape = compute_axis_output_shape(shape_slice, axis_usize, false);
                ArrayD::from_shape_vec(IxDyn(&out_shape), vec![0.0; out_shape.iter().product()])
                    .expect("Failed to create empty mean array")
            }
        };

        // Handle keepdims
        let final_arr = if keepdims {
            result_arr.insert_axis(Axis(axis_usize))
        } else {
            result_arr
        };

        // Mean should always be Float64
        let result_wrapper = NDArrayWrapper {
            data: ArrayData::Float64(Arc::new(RwLock::new(final_arr))),
            dtype: DType::Float64,
        };

        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
