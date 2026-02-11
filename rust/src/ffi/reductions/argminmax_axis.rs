//! Axis argmin/argmax reductions.

use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::reductions::helpers::{compute_axis_output_shape, extract_view, validate_axis};
use crate::ffi::NdArrayHandle;
use ndarray::{ArrayD, Axis, IxDyn};
use parking_lot::RwLock;
use std::sync::Arc;

/// Argmin along axis (indices of minimum values).
#[no_mangle]
pub unsafe extern "C" fn ndarray_argmin_axis(
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

        let view = extract_view(wrapper, offset, shape_slice);
        let axis_len = shape_slice[axis_usize];

        if axis_len == 0 {
            crate::error::set_last_error("Cannot compute argmin along empty axis".to_string());
            return ERR_GENERIC;
        }

        // Map along axis to find index of minimum
        let result_arr: ArrayD<i64> = view.map_axis(Axis(axis_usize), |lane| {
            lane.iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i64)
                .unwrap_or(0)
        });

        // Handle keepdims
        let out_shape = compute_axis_output_shape(shape_slice, axis_usize, keepdims);
        let final_arr = if keepdims {
            // Need to insert dimension
            let reshaped =
                ArrayD::from_shape_vec(IxDyn(&out_shape), result_arr.iter().cloned().collect())
                    .expect("Failed to reshape argmin result");
            reshaped
        } else {
            result_arr
        };

        // Argmin indices are always Int64
        let result_wrapper = NDArrayWrapper {
            data: ArrayData::Int64(Arc::new(RwLock::new(final_arr))),
            dtype: DType::Int64,
        };

        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}

/// Argmax along axis (indices of maximum values).
#[no_mangle]
pub unsafe extern "C" fn ndarray_argmax_axis(
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

        let view = extract_view(wrapper, offset, shape_slice);
        let axis_len = shape_slice[axis_usize];

        if axis_len == 0 {
            crate::error::set_last_error("Cannot compute argmax along empty axis".to_string());
            return ERR_GENERIC;
        }

        // Map along axis to find index of maximum
        let result_arr: ArrayD<i64> = view.map_axis(Axis(axis_usize), |lane| {
            lane.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i64)
                .unwrap_or(0)
        });

        // Handle keepdims
        let out_shape = compute_axis_output_shape(shape_slice, axis_usize, keepdims);
        let final_arr = if keepdims {
            ArrayD::from_shape_vec(IxDyn(&out_shape), result_arr.iter().cloned().collect())
                .expect("Failed to reshape argmax result")
        } else {
            result_arr
        };

        // Argmax indices are always Int64
        let result_wrapper = NDArrayWrapper {
            data: ArrayData::Int64(Arc::new(RwLock::new(final_arr))),
            dtype: DType::Int64,
        };

        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
