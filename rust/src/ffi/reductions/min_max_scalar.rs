//! Scalar min/max reductions.

use crate::core::view_helpers::{
    create_scalar_wrapper_with_dtype, extract_view_f32, extract_view_f64, extract_view_i16,
    extract_view_i32, extract_view_i64, extract_view_i8, extract_view_u16, extract_view_u32,
    extract_view_u64, extract_view_u8,
};
use crate::core::NDArrayWrapper;
use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::NdArrayHandle;

/// Compute min of elements using proper strided view.
fn compute_min(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> Option<f64> {
    unsafe {
        if let Some(view) = extract_view_f64(wrapper, offset, shape, strides) {
            return view.iter().cloned().fold(None, |min: Option<f64>, x| {
                Some(min.map_or(x, |m| m.min(x)))
            });
        }
        if let Some(view) = extract_view_f32(wrapper, offset, shape, strides) {
            return view
                .iter()
                .map(|x| *x as f64)
                .fold(None, |min: Option<f64>, x| {
                    Some(min.map_or(x, |m| m.min(x)))
                });
        }
        if let Some(view) = extract_view_i64(wrapper, offset, shape, strides) {
            return view
                .iter()
                .map(|x| *x as f64)
                .fold(None, |min: Option<f64>, x| {
                    Some(min.map_or(x, |m| m.min(x)))
                });
        }
        if let Some(view) = extract_view_i32(wrapper, offset, shape, strides) {
            return view
                .iter()
                .map(|x| *x as f64)
                .fold(None, |min: Option<f64>, x| {
                    Some(min.map_or(x, |m| m.min(x)))
                });
        }
        if let Some(view) = extract_view_i16(wrapper, offset, shape, strides) {
            return view
                .iter()
                .map(|x| *x as f64)
                .fold(None, |min: Option<f64>, x| {
                    Some(min.map_or(x, |m| m.min(x)))
                });
        }
        if let Some(view) = extract_view_i8(wrapper, offset, shape, strides) {
            return view
                .iter()
                .map(|x| *x as f64)
                .fold(None, |min: Option<f64>, x| {
                    Some(min.map_or(x, |m| m.min(x)))
                });
        }
        if let Some(view) = extract_view_u64(wrapper, offset, shape, strides) {
            return view
                .iter()
                .map(|x| *x as f64)
                .fold(None, |min: Option<f64>, x| {
                    Some(min.map_or(x, |m| m.min(x)))
                });
        }
        if let Some(view) = extract_view_u32(wrapper, offset, shape, strides) {
            return view
                .iter()
                .map(|x| *x as f64)
                .fold(None, |min: Option<f64>, x| {
                    Some(min.map_or(x, |m| m.min(x)))
                });
        }
        if let Some(view) = extract_view_u16(wrapper, offset, shape, strides) {
            return view
                .iter()
                .map(|x| *x as f64)
                .fold(None, |min: Option<f64>, x| {
                    Some(min.map_or(x, |m| m.min(x)))
                });
        }
        if let Some(view) = extract_view_u8(wrapper, offset, shape, strides) {
            return view
                .iter()
                .map(|x| *x as f64)
                .fold(None, |min: Option<f64>, x| {
                    Some(min.map_or(x, |m| m.min(x)))
                });
        }
    }
    None
}

/// Compute max of elements using proper strided view.
fn compute_max(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> Option<f64> {
    unsafe {
        if let Some(view) = extract_view_f64(wrapper, offset, shape, strides) {
            return view.iter().cloned().fold(None, |max: Option<f64>, x| {
                Some(max.map_or(x, |m| m.max(x)))
            });
        }
        if let Some(view) = extract_view_f32(wrapper, offset, shape, strides) {
            return view
                .iter()
                .map(|x| *x as f64)
                .fold(None, |max: Option<f64>, x| {
                    Some(max.map_or(x, |m| m.max(x)))
                });
        }
        if let Some(view) = extract_view_i64(wrapper, offset, shape, strides) {
            return view
                .iter()
                .map(|x| *x as f64)
                .fold(None, |max: Option<f64>, x| {
                    Some(max.map_or(x, |m| m.max(x)))
                });
        }
        if let Some(view) = extract_view_i32(wrapper, offset, shape, strides) {
            return view
                .iter()
                .map(|x| *x as f64)
                .fold(None, |max: Option<f64>, x| {
                    Some(max.map_or(x, |m| m.max(x)))
                });
        }
        if let Some(view) = extract_view_i16(wrapper, offset, shape, strides) {
            return view
                .iter()
                .map(|x| *x as f64)
                .fold(None, |max: Option<f64>, x| {
                    Some(max.map_or(x, |m| m.max(x)))
                });
        }
        if let Some(view) = extract_view_i8(wrapper, offset, shape, strides) {
            return view
                .iter()
                .map(|x| *x as f64)
                .fold(None, |max: Option<f64>, x| {
                    Some(max.map_or(x, |m| m.max(x)))
                });
        }
        if let Some(view) = extract_view_u64(wrapper, offset, shape, strides) {
            return view
                .iter()
                .map(|x| *x as f64)
                .fold(None, |max: Option<f64>, x| {
                    Some(max.map_or(x, |m| m.max(x)))
                });
        }
        if let Some(view) = extract_view_u32(wrapper, offset, shape, strides) {
            return view
                .iter()
                .map(|x| *x as f64)
                .fold(None, |max: Option<f64>, x| {
                    Some(max.map_or(x, |m| m.max(x)))
                });
        }
        if let Some(view) = extract_view_u16(wrapper, offset, shape, strides) {
            return view
                .iter()
                .map(|x| *x as f64)
                .fold(None, |max: Option<f64>, x| {
                    Some(max.map_or(x, |m| m.max(x)))
                });
        }
        if let Some(view) = extract_view_u8(wrapper, offset, shape, strides) {
            return view
                .iter()
                .map(|x| *x as f64)
                .fold(None, |max: Option<f64>, x| {
                    Some(max.map_or(x, |m| m.max(x)))
                });
        }
    }
    None
}

/// Compute the minimum of all elements in the array (scalar reduction).
#[no_mangle]
pub unsafe extern "C" fn ndarray_min(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || out_handle.is_null() || shape.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = std::slice::from_raw_parts(shape, ndim);
        let strides_slice = std::slice::from_raw_parts(strides, ndim);

        let result = compute_min(wrapper, offset, shape_slice, strides_slice).unwrap_or(f64::NAN);
        let result_wrapper = create_scalar_wrapper_with_dtype(result, wrapper.dtype);
        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));

        SUCCESS
    })
}

/// Compute the maximum of all elements in the array (scalar reduction).
#[no_mangle]
pub unsafe extern "C" fn ndarray_max(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || out_handle.is_null() || shape.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = std::slice::from_raw_parts(shape, ndim);
        let strides_slice = std::slice::from_raw_parts(strides, ndim);

        let result = compute_max(wrapper, offset, shape_slice, strides_slice).unwrap_or(f64::NAN);
        let result_wrapper = create_scalar_wrapper_with_dtype(result, wrapper.dtype);
        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));

        SUCCESS
    })
}
