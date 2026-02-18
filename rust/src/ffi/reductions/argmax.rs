//! Argmax reduction.

use crate::core::view_helpers::{
    extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32, extract_view_i64,
    extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64, extract_view_u8,
};
use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::reductions::helpers::{compute_axis_output_shape, validate_axis, write_scalar};
use crate::ffi::{write_output_metadata, NdArrayHandle};
use ndarray::{ArrayD, IxDyn};
use parking_lot::RwLock;
use std::ffi::c_void;
use std::slice;
use std::sync::Arc;

/// Argmax along axis.
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
    out_dtype: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if handle.is_null()
        || out_handle.is_null()
        || shape.is_null()
        || out_dtype.is_null()
        || out_ndim.is_null()
        || out_shape.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = slice::from_raw_parts(shape, ndim);
        let strides_slice = slice::from_raw_parts(strides, ndim);

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

        // Match on dtype, extract view, compute argmax along axis, and create result wrapper
        let result_arr: ArrayD<i64> = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                view.map_axis(ndarray::Axis(axis_usize), |lane| {
                    lane.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx as i64)
                        .unwrap_or(0)
                })
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                view.map_axis(ndarray::Axis(axis_usize), |lane| {
                    lane.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx as i64)
                        .unwrap_or(0)
                })
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                view.map_axis(ndarray::Axis(axis_usize), |lane| {
                    lane.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx as i64)
                        .unwrap_or(0)
                })
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                view.map_axis(ndarray::Axis(axis_usize), |lane| {
                    lane.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx as i64)
                        .unwrap_or(0)
                })
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                view.map_axis(ndarray::Axis(axis_usize), |lane| {
                    lane.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx as i64)
                        .unwrap_or(0)
                })
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                view.map_axis(ndarray::Axis(axis_usize), |lane| {
                    lane.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx as i64)
                        .unwrap_or(0)
                })
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                view.map_axis(ndarray::Axis(axis_usize), |lane| {
                    lane.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx as i64)
                        .unwrap_or(0)
                })
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                view.map_axis(ndarray::Axis(axis_usize), |lane| {
                    lane.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx as i64)
                        .unwrap_or(0)
                })
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                view.map_axis(ndarray::Axis(axis_usize), |lane| {
                    lane.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx as i64)
                        .unwrap_or(0)
                })
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                view.map_axis(ndarray::Axis(axis_usize), |lane| {
                    lane.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx as i64)
                        .unwrap_or(0)
                })
            }
            DType::Bool => {
                crate::error::set_last_error(
                    "argmax_axis() not supported for Bool type".to_string(),
                );
                return ERR_GENERIC;
            }
        };

        // Handle keepdims
        let result_shape = compute_axis_output_shape(shape_slice, axis_usize, keepdims);
        let final_arr = if keepdims {
            ArrayD::from_shape_vec(IxDyn(&result_shape), result_arr.iter().cloned().collect())
                .expect("Failed to reshape argmax result")
        } else {
            result_arr
        };

        // Argmax indices are always Int64
        let result_wrapper = NDArrayWrapper {
            data: ArrayData::Int64(Arc::new(RwLock::new(final_arr))),
            dtype: DType::Int64,
        };

        if let Err(e) =
            write_output_metadata(&result_wrapper, out_dtype, out_ndim, out_shape, max_ndim)
        {
            crate::error::set_last_error(e);
            return ERR_GENERIC;
        }
        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}

/// Compute the index of the maximum element.
/// Returns Int64 index.
#[no_mangle]
pub unsafe extern "C" fn ndarray_argmax(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
    out_value: *mut c_void,
    out_dtype: *mut u8,
) -> i32 {
    if handle.is_null() || out_value.is_null() || out_dtype.is_null() || shape.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = slice::from_raw_parts(shape, ndim);
        let strides_slice = slice::from_raw_parts(strides, ndim);

        let argmax_result = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(-1)
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(-1)
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.cmp(b))
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(-1)
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.cmp(b))
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(-1)
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.cmp(b))
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(-1)
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.cmp(b))
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(-1)
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.cmp(b))
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(-1)
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.cmp(b))
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(-1)
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.cmp(b))
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(-1)
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.cmp(b))
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(-1)
            }
            DType::Bool => {
                crate::error::set_last_error("argmax() not supported for Bool type".to_string());
                return ERR_GENERIC;
            }
        };

        write_scalar(out_value, out_dtype, argmax_result as f64, DType::Int64);
        SUCCESS
    })
}
