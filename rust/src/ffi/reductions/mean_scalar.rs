//! Scalar mean reduction.

use crate::core::view_helpers::{
    create_scalar_wrapper_f64, extract_view_f32, extract_view_f64, extract_view_i16,
    extract_view_i32, extract_view_i64, extract_view_i8, extract_view_u16, extract_view_u32,
    extract_view_u64, extract_view_u8,
};
use crate::core::NDArrayWrapper;
use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::NdArrayHandle;

/// Compute mean using ndarray's native mean() method.
///
/// Returns f64 regardless of input type.
fn compute_mean(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> f64 {
    unsafe {
        // Float64 - native support
        if let Some(view) = extract_view_f64(wrapper, offset, shape, strides) {
            return view.mean().unwrap_or(0.0);
        }
        // Float32 - native support
        if let Some(view) = extract_view_f32(wrapper, offset, shape, strides) {
            return view.mean().map(|x| x as f64).unwrap_or(0.0);
        }
        // Integers - convert to f64, then use ndarray's mean
        if let Some(view) = extract_view_i64(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).mean().unwrap_or(0.0);
        }
        if let Some(view) = extract_view_i32(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).mean().unwrap_or(0.0);
        }
        if let Some(view) = extract_view_i16(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).mean().unwrap_or(0.0);
        }
        if let Some(view) = extract_view_i8(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).mean().unwrap_or(0.0);
        }
        if let Some(view) = extract_view_u64(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).mean().unwrap_or(0.0);
        }
        if let Some(view) = extract_view_u32(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).mean().unwrap_or(0.0);
        }
        if let Some(view) = extract_view_u16(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).mean().unwrap_or(0.0);
        }
        if let Some(view) = extract_view_u8(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).mean().unwrap_or(0.0);
        }
    }
    0.0
}

/// Compute the mean of all elements in the array (scalar reduction).
///
/// Returns Float64 regardless of input dtype.
#[no_mangle]
pub unsafe extern "C" fn ndarray_mean(
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

        let result = compute_mean(wrapper, offset, shape_slice, strides_slice);
        let result_wrapper = create_scalar_wrapper_f64(result);
        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));

        SUCCESS
    })
}
