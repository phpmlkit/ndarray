//! Variance reduction.

use std::ffi::c_void;

use crate::core::view_helpers::{
    extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32, extract_view_i64,
    extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64, extract_view_u8,
};
use crate::dtype::DType;
use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::{NdArrayHandle, write_output_metadata};
use std::slice;
use ndarray::Axis;
use parking_lot::RwLock;
use std::sync::Arc;
use crate::core::{ArrayData, NDArrayWrapper};
use crate::ffi::reductions::helpers::{validate_axis, write_scalar};

/// Compute the variance of all elements in the array.
#[no_mangle]
pub unsafe extern "C" fn ndarray_var(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
    ddof: f64,
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

        let n = shape_slice.iter().map(|&x| x as f64).product::<f64>();

        if n <= ddof {
            crate::error::set_last_error(format!(
                "ddof ({}) must be less than number of elements ({})",
                ddof, n as usize
            ));
            return ERR_GENERIC;
        }

        // Match on dtype, extract view, compute variance
        let var_result = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                view.var(ddof)
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                view.var(ddof as f32) as f64
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                view.mapv(|x| x as f64).var(ddof)
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                view.mapv(|x| x as f64).var(ddof)
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                view.mapv(|x| x as f64).var(ddof)
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                view.mapv(|x| x as f64).var(ddof)
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                view.mapv(|x| x as f64).var(ddof)
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                view.mapv(|x| x as f64).var(ddof)
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                view.mapv(|x| x as f64).var(ddof)
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                view.mapv(|x| x as f64).var(ddof)
            }
            DType::Bool => {
                crate::error::set_last_error("var() not supported for Bool type".to_string());
                return ERR_GENERIC;
            }
        };

        write_scalar(out_value, out_dtype, var_result, DType::Float64);
        SUCCESS
    })
}

/// Compute the variance along an axis in the array.
#[no_mangle]
pub unsafe extern "C" fn ndarray_var_axis(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
    axis: i32,
    keepdims: bool,
    ddof: f64,
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

        if (axis_len as f64) <= ddof {
            crate::error::set_last_error(format!(
                "ddof ({}) must be less than axis length ({})",
                ddof, axis_len
            ));
            return ERR_GENERIC;
        }

        // Match on dtype, extract view, compute variance along axis, and create result wrapper
        let result = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                view.var_axis(Axis(axis_usize), ddof)
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                view.mapv(|x| x as f64).var_axis(Axis(axis_usize), ddof)
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                view.mapv(|x| x as f64).var_axis(Axis(axis_usize), ddof)
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                view.mapv(|x| x as f64).var_axis(Axis(axis_usize), ddof)
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                view.mapv(|x| x as f64).var_axis(Axis(axis_usize), ddof)
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                view.mapv(|x| x as f64).var_axis(Axis(axis_usize), ddof)
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                view.mapv(|x| x as f64).var_axis(Axis(axis_usize), ddof)
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                view.mapv(|x| x as f64).var_axis(Axis(axis_usize), ddof)
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                view.mapv(|x| x as f64).var_axis(Axis(axis_usize), ddof)
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                view.mapv(|x| x as f64).var_axis(Axis(axis_usize), ddof)
            }
            DType::Bool => {
                crate::error::set_last_error("var_axis() not supported for Bool type".to_string());
                return ERR_GENERIC;
            }
        };

        let final_arr = if keepdims {
            result.insert_axis(Axis(axis_usize))
        } else {
            result
        };
        let result_wrapper = NDArrayWrapper {
            data: ArrayData::Float64(Arc::new(RwLock::new(final_arr))),
            dtype: DType::Float64,
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