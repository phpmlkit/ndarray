//! Axis variance and standard deviation reductions.

use crate::core::view_helpers::{
    extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32, extract_view_i64,
    extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64, extract_view_u8,
};
use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::reductions::helpers::validate_axis;
use crate::ffi::NdArrayHandle;
use ndarray::Axis;
use parking_lot::RwLock;
use std::sync::Arc;

/// Compute variance along axis using proper strided view.
fn var_axis_view(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: usize,
    ddof: f64,
) -> ndarray::ArrayD<f64> {
    unsafe {
        // Try Float64 first (fastest path)
        if let Some(view) = extract_view_f64(wrapper, offset, shape, strides) {
            return view.var_axis(Axis(axis), ddof);
        }
        // Float32 - convert to f64 for variance calculation
        if let Some(view) = extract_view_f32(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).var_axis(Axis(axis), ddof);
        }
        // Int64
        if let Some(view) = extract_view_i64(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).var_axis(Axis(axis), ddof);
        }
        // Int32
        if let Some(view) = extract_view_i32(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).var_axis(Axis(axis), ddof);
        }
        // Int16
        if let Some(view) = extract_view_i16(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).var_axis(Axis(axis), ddof);
        }
        // Int8
        if let Some(view) = extract_view_i8(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).var_axis(Axis(axis), ddof);
        }
        // Uint64
        if let Some(view) = extract_view_u64(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).var_axis(Axis(axis), ddof);
        }
        // Uint32
        if let Some(view) = extract_view_u32(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).var_axis(Axis(axis), ddof);
        }
        // Uint16
        if let Some(view) = extract_view_u16(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).var_axis(Axis(axis), ddof);
        }
        // Uint8
        if let Some(view) = extract_view_u8(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).var_axis(Axis(axis), ddof);
        }
    }
    // Return empty array if no type matched
    ndarray::ArrayD::zeros(ndarray::IxDyn(&[]))
}

/// Compute std along axis using proper strided view.
fn std_axis_view(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: usize,
    ddof: f64,
) -> ndarray::ArrayD<f64> {
    unsafe {
        // Try Float64 first (fastest path)
        if let Some(view) = extract_view_f64(wrapper, offset, shape, strides) {
            return view.std_axis(Axis(axis), ddof);
        }
        // Float32 - convert to f64 for std calculation
        if let Some(view) = extract_view_f32(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).std_axis(Axis(axis), ddof);
        }
        // Int64
        if let Some(view) = extract_view_i64(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).std_axis(Axis(axis), ddof);
        }
        // Int32
        if let Some(view) = extract_view_i32(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).std_axis(Axis(axis), ddof);
        }
        // Int16
        if let Some(view) = extract_view_i16(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).std_axis(Axis(axis), ddof);
        }
        // Int8
        if let Some(view) = extract_view_i8(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).std_axis(Axis(axis), ddof);
        }
        // Uint64
        if let Some(view) = extract_view_u64(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).std_axis(Axis(axis), ddof);
        }
        // Uint32
        if let Some(view) = extract_view_u32(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).std_axis(Axis(axis), ddof);
        }
        // Uint16
        if let Some(view) = extract_view_u16(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).std_axis(Axis(axis), ddof);
        }
        // Uint8
        if let Some(view) = extract_view_u8(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).std_axis(Axis(axis), ddof);
        }
    }
    // Return empty array if no type matched
    ndarray::ArrayD::zeros(ndarray::IxDyn(&[]))
}

/// Variance along axis.
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

        if (axis_len as f64) <= ddof {
            crate::error::set_last_error(format!(
                "ddof ({}) must be less than axis length ({})",
                ddof, axis_len
            ));
            return ERR_GENERIC;
        }

        // Compute variance along axis using proper strided view
        let result_arr = var_axis_view(
            wrapper,
            offset,
            shape_slice,
            strides_slice,
            axis_usize,
            ddof,
        );

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
    strides: *const usize,
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

        if (axis_len as f64) <= ddof {
            crate::error::set_last_error(format!(
                "ddof ({}) must be less than axis length ({})",
                ddof, axis_len
            ));
            return ERR_GENERIC;
        }

        // Compute std along axis using proper strided view
        let result_arr = std_axis_view(
            wrapper,
            offset,
            shape_slice,
            strides_slice,
            axis_usize,
            ddof,
        );

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
