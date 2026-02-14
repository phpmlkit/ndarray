//! Scalar variance and standard deviation reductions.

use crate::core::view_helpers::{
    extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32, extract_view_i64,
    extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64, extract_view_u8,
};
use crate::core::NDArrayWrapper;
use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::NdArrayHandle;

/// Compute variance using ndarray's native method.
///
/// For float types (f64, f32), uses native var() method.
/// For integers, extracts as f64 then computes variance.
fn compute_var(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
    ddof: f64,
) -> f64 {
    unsafe {
        // Float64 - native support
        if let Some(view) = extract_view_f64(wrapper, offset, shape, strides) {
            return view.var(ddof);
        }
        // Float32 - native support with f32 precision
        if let Some(view) = extract_view_f32(wrapper, offset, shape, strides) {
            return view.var(ddof as f32) as f64;
        }
        // Integers - convert to f64, then use ndarray's var
        if let Some(view) = extract_view_i64(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).var(ddof);
        }
        if let Some(view) = extract_view_i32(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).var(ddof);
        }
        if let Some(view) = extract_view_i16(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).var(ddof);
        }
        if let Some(view) = extract_view_i8(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).var(ddof);
        }
        if let Some(view) = extract_view_u64(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).var(ddof);
        }
        if let Some(view) = extract_view_u32(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).var(ddof);
        }
        if let Some(view) = extract_view_u16(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).var(ddof);
        }
        if let Some(view) = extract_view_u8(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).var(ddof);
        }
    }
    0.0
}

/// Compute standard deviation using ndarray's native method.
fn compute_std(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
    ddof: f64,
) -> f64 {
    unsafe {
        // Float64 - native support
        if let Some(view) = extract_view_f64(wrapper, offset, shape, strides) {
            return view.std(ddof);
        }
        // Float32 - native support
        if let Some(view) = extract_view_f32(wrapper, offset, shape, strides) {
            return view.std(ddof as f32) as f64;
        }
        // Integers - convert to f64, then use ndarray's std
        if let Some(view) = extract_view_i64(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).std(ddof);
        }
        if let Some(view) = extract_view_i32(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).std(ddof);
        }
        if let Some(view) = extract_view_i16(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).std(ddof);
        }
        if let Some(view) = extract_view_i8(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).std(ddof);
        }
        if let Some(view) = extract_view_u64(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).std(ddof);
        }
        if let Some(view) = extract_view_u32(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).std(ddof);
        }
        if let Some(view) = extract_view_u16(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).std(ddof);
        }
        if let Some(view) = extract_view_u8(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).std(ddof);
        }
    }
    0.0
}

/// Compute the variance of all elements (scalar reduction).
#[no_mangle]
pub unsafe extern "C" fn ndarray_var(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
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
        let strides_slice = std::slice::from_raw_parts(strides, ndim);

        let n = shape_slice.iter().map(|&x| x as f64).product::<f64>();

        if n <= ddof {
            crate::error::set_last_error(format!(
                "ddof ({}) must be less than number of elements ({})",
                ddof, n as usize
            ));
            return ERR_GENERIC;
        }

        let result = compute_var(wrapper, offset, shape_slice, strides_slice, ddof);
        let result_wrapper =
            NDArrayWrapper::create_scalar_wrapper(result, crate::dtype::DType::Float64);
        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));

        SUCCESS
    })
}

/// Compute the standard deviation of all elements (scalar reduction).
#[no_mangle]
pub unsafe extern "C" fn ndarray_std(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
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
        let strides_slice = std::slice::from_raw_parts(strides, ndim);

        let n = shape_slice.iter().map(|&x| x as f64).product::<f64>();

        if n <= ddof {
            crate::error::set_last_error(format!(
                "ddof ({}) must be less than number of elements ({})",
                ddof, n as usize
            ));
            return ERR_GENERIC;
        }

        let result = compute_std(wrapper, offset, shape_slice, strides_slice, ddof);
        let result_wrapper =
            NDArrayWrapper::create_scalar_wrapper(result, crate::dtype::DType::Float64);
        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));

        SUCCESS
    })
}
