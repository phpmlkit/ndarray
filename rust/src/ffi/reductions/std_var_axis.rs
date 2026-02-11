//! Axis variance and standard deviation reductions.

use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::reductions::helpers::{extract_view, validate_axis};
use crate::ffi::NdArrayHandle;
use ndarray::Axis;
use parking_lot::RwLock;
use std::sync::Arc;

/// Variance along axis.
#[no_mangle]
pub unsafe extern "C" fn ndarray_var_axis(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    _strides: *const usize,
    ndim: usize,
    axis: i32,
    keepdims: bool,
    ddof: f64,
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

        let view = extract_view(wrapper, offset, shape_slice);
        let axis_len = shape_slice[axis_usize];

        if (axis_len as f64) <= ddof {
            crate::error::set_last_error(format!(
                "ddof ({}) must be less than axis length ({})",
                ddof, axis_len
            ));
            return ERR_GENERIC;
        }

        // Compute variance along axis
        let result_arr = view.var_axis(Axis(axis_usize), ddof);

        // Handle keepdims
        let final_arr = if keepdims {
            result_arr.insert_axis(Axis(axis_usize))
        } else {
            result_arr
        };

        // Variance is always Float64
        let result_wrapper = NDArrayWrapper {
            data: ArrayData::Float64(Arc::new(RwLock::new(final_arr))),
            dtype: DType::Float64,
        };

        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}

/// Standard deviation along axis.
#[no_mangle]
pub unsafe extern "C" fn ndarray_std_axis(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    _strides: *const usize,
    ndim: usize,
    axis: i32,
    keepdims: bool,
    ddof: f64,
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

        let view = extract_view(wrapper, offset, shape_slice);
        let axis_len = shape_slice[axis_usize];

        if (axis_len as f64) <= ddof {
            crate::error::set_last_error(format!(
                "ddof ({}) must be less than axis length ({})",
                ddof, axis_len
            ));
            return ERR_GENERIC;
        }

        // Compute std along axis
        let result_arr = view.std_axis(Axis(axis_usize), ddof);

        // Handle keepdims
        let final_arr = if keepdims {
            result_arr.insert_axis(Axis(axis_usize))
        } else {
            result_arr
        };

        // Std is always Float64
        let result_wrapper = NDArrayWrapper {
            data: ArrayData::Float64(Arc::new(RwLock::new(final_arr))),
            dtype: DType::Float64,
        };

        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
