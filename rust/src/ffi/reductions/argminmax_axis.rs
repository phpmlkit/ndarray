//! Axis argmin/argmax reductions.

use crate::core::view_helpers::{
    extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32, extract_view_i64,
    extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64, extract_view_u8,
};
use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::reductions::helpers::{compute_axis_output_shape, validate_axis};
use crate::ffi::NdArrayHandle;
use ndarray::{ArrayD, IxDyn};
use parking_lot::RwLock;
use std::sync::Arc;

/// Compute argmin along axis using proper strided view.
fn argmin_axis_view(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: usize,
) -> ndarray::ArrayD<i64> {
    unsafe {
        // Try Float64 first (fastest path)
        if let Some(view) = extract_view_f64(wrapper, offset, shape, strides) {
            return view.map_axis(ndarray::Axis(axis), |lane| {
                lane.iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(0)
            });
        }
        // Float32
        if let Some(view) = extract_view_f32(wrapper, offset, shape, strides) {
            return view.map_axis(ndarray::Axis(axis), |lane| {
                lane.iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(0)
            });
        }
        // Int64
        if let Some(view) = extract_view_i64(wrapper, offset, shape, strides) {
            return view.map_axis(ndarray::Axis(axis), |lane| {
                lane.iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(0)
            });
        }
        // Int32
        if let Some(view) = extract_view_i32(wrapper, offset, shape, strides) {
            return view.map_axis(ndarray::Axis(axis), |lane| {
                lane.iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(0)
            });
        }
        // Int16
        if let Some(view) = extract_view_i16(wrapper, offset, shape, strides) {
            return view.map_axis(ndarray::Axis(axis), |lane| {
                lane.iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(0)
            });
        }
        // Int8
        if let Some(view) = extract_view_i8(wrapper, offset, shape, strides) {
            return view.map_axis(ndarray::Axis(axis), |lane| {
                lane.iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(0)
            });
        }
        // Uint64
        if let Some(view) = extract_view_u64(wrapper, offset, shape, strides) {
            return view.map_axis(ndarray::Axis(axis), |lane| {
                lane.iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(0)
            });
        }
        // Uint32
        if let Some(view) = extract_view_u32(wrapper, offset, shape, strides) {
            return view.map_axis(ndarray::Axis(axis), |lane| {
                lane.iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(0)
            });
        }
        // Uint16
        if let Some(view) = extract_view_u16(wrapper, offset, shape, strides) {
            return view.map_axis(ndarray::Axis(axis), |lane| {
                lane.iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(0)
            });
        }
        // Uint8
        if let Some(view) = extract_view_u8(wrapper, offset, shape, strides) {
            return view.map_axis(ndarray::Axis(axis), |lane| {
                lane.iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(0)
            });
        }
    }
    // Return empty array if no type matched
    ndarray::ArrayD::zeros(ndarray::IxDyn(&[]))
}

/// Compute argmax along axis using proper strided view.
fn argmax_axis_view(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: usize,
) -> ndarray::ArrayD<i64> {
    unsafe {
        // Try Float64 first (fastest path)
        if let Some(view) = extract_view_f64(wrapper, offset, shape, strides) {
            return view.map_axis(ndarray::Axis(axis), |lane| {
                lane.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(0)
            });
        }
        // Float32
        if let Some(view) = extract_view_f32(wrapper, offset, shape, strides) {
            return view.map_axis(ndarray::Axis(axis), |lane| {
                lane.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(0)
            });
        }
        // Int64
        if let Some(view) = extract_view_i64(wrapper, offset, shape, strides) {
            return view.map_axis(ndarray::Axis(axis), |lane| {
                lane.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(0)
            });
        }
        // Int32
        if let Some(view) = extract_view_i32(wrapper, offset, shape, strides) {
            return view.map_axis(ndarray::Axis(axis), |lane| {
                lane.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(0)
            });
        }
        // Int16
        if let Some(view) = extract_view_i16(wrapper, offset, shape, strides) {
            return view.map_axis(ndarray::Axis(axis), |lane| {
                lane.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(0)
            });
        }
        // Int8
        if let Some(view) = extract_view_i8(wrapper, offset, shape, strides) {
            return view.map_axis(ndarray::Axis(axis), |lane| {
                lane.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(0)
            });
        }
        // Uint64
        if let Some(view) = extract_view_u64(wrapper, offset, shape, strides) {
            return view.map_axis(ndarray::Axis(axis), |lane| {
                lane.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(0)
            });
        }
        // Uint32
        if let Some(view) = extract_view_u32(wrapper, offset, shape, strides) {
            return view.map_axis(ndarray::Axis(axis), |lane| {
                lane.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(0)
            });
        }
        // Uint16
        if let Some(view) = extract_view_u16(wrapper, offset, shape, strides) {
            return view.map_axis(ndarray::Axis(axis), |lane| {
                lane.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(0)
            });
        }
        // Uint8
        if let Some(view) = extract_view_u8(wrapper, offset, shape, strides) {
            return view.map_axis(ndarray::Axis(axis), |lane| {
                lane.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(0)
            });
        }
    }
    // Return empty array if no type matched
    ndarray::ArrayD::zeros(ndarray::IxDyn(&[]))
}

/// Argmin along axis (indices of minimum values).
#[no_mangle]
pub unsafe extern "C" fn ndarray_argmin_axis(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
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
        let strides_slice = std::slice::from_raw_parts(strides, ndim);

        // Validate axis
        let axis_usize = match validate_axis(shape_slice, axis) {
            Ok(a) => a,
            Err(e) => {
                crate::error::set_last_error(e);
                return ERR_GENERIC;
            }
        };

        let axis_len = shape_slice[axis_usize];

        if axis_len == 0 {
            crate::error::set_last_error("Cannot compute argmin along empty axis".to_string());
            return ERR_GENERIC;
        }

        // Compute argmin along axis using proper strided view
        let result_arr = argmin_axis_view(wrapper, offset, shape_slice, strides_slice, axis_usize);

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
    strides: *const usize,
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
        let strides_slice = std::slice::from_raw_parts(strides, ndim);

        // Validate axis
        let axis_usize = match validate_axis(shape_slice, axis) {
            Ok(a) => a,
            Err(e) => {
                crate::error::set_last_error(e);
                return ERR_GENERIC;
            }
        };

        let axis_len = shape_slice[axis_usize];

        if axis_len == 0 {
            crate::error::set_last_error("Cannot compute argmax along empty axis".to_string());
            return ERR_GENERIC;
        }

        // Compute argmax along axis using proper strided view
        let result_arr = argmax_axis_view(wrapper, offset, shape_slice, strides_slice, axis_usize);

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
