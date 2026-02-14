//! Scalar argmin/argmax reductions.

use crate::core::view_helpers::{
    extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32, extract_view_i64,
    extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64, extract_view_u8,
};
use crate::core::NDArrayWrapper;
use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::NdArrayHandle;

/// Compute argmin using proper strided view.
fn compute_argmin(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> Option<i64> {
    unsafe {
        if let Some(view) = extract_view_f64(wrapper, offset, shape, strides) {
            return view
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i64);
        }
        if let Some(view) = extract_view_f32(wrapper, offset, shape, strides) {
            return view
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i64);
        }
        if let Some(view) = extract_view_i64(wrapper, offset, shape, strides) {
            return view
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i64);
        }
        if let Some(view) = extract_view_i32(wrapper, offset, shape, strides) {
            return view
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i64);
        }
        if let Some(view) = extract_view_i16(wrapper, offset, shape, strides) {
            return view
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i64);
        }
        if let Some(view) = extract_view_i8(wrapper, offset, shape, strides) {
            return view
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i64);
        }
        if let Some(view) = extract_view_u64(wrapper, offset, shape, strides) {
            return view
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i64);
        }
        if let Some(view) = extract_view_u32(wrapper, offset, shape, strides) {
            return view
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i64);
        }
        if let Some(view) = extract_view_u16(wrapper, offset, shape, strides) {
            return view
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i64);
        }
        if let Some(view) = extract_view_u8(wrapper, offset, shape, strides) {
            return view
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i64);
        }
    }
    None
}

/// Compute argmax using proper strided view.
fn compute_argmax(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> Option<i64> {
    unsafe {
        if let Some(view) = extract_view_f64(wrapper, offset, shape, strides) {
            return view
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i64);
        }
        if let Some(view) = extract_view_f32(wrapper, offset, shape, strides) {
            return view
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i64);
        }
        if let Some(view) = extract_view_i64(wrapper, offset, shape, strides) {
            return view
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i64);
        }
        if let Some(view) = extract_view_i32(wrapper, offset, shape, strides) {
            return view
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i64);
        }
        if let Some(view) = extract_view_i16(wrapper, offset, shape, strides) {
            return view
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i64);
        }
        if let Some(view) = extract_view_i8(wrapper, offset, shape, strides) {
            return view
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i64);
        }
        if let Some(view) = extract_view_u64(wrapper, offset, shape, strides) {
            return view
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i64);
        }
        if let Some(view) = extract_view_u32(wrapper, offset, shape, strides) {
            return view
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i64);
        }
        if let Some(view) = extract_view_u16(wrapper, offset, shape, strides) {
            return view
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i64);
        }
        if let Some(view) = extract_view_u8(wrapper, offset, shape, strides) {
            return view
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i64);
        }
    }
    None
}

/// Compute the index of the minimum element (scalar reduction).
/// Returns Int64 index.
#[no_mangle]
pub unsafe extern "C" fn ndarray_argmin(
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

        let result = compute_argmin(wrapper, offset, shape_slice, strides_slice).unwrap_or(-1);

        let result_wrapper =
            NDArrayWrapper::create_scalar_wrapper(result as f64, crate::dtype::DType::Int64);
        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));

        SUCCESS
    })
}

/// Compute the index of the maximum element (scalar reduction).
/// Returns Int64 index.
#[no_mangle]
pub unsafe extern "C" fn ndarray_argmax(
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

        let result = compute_argmax(wrapper, offset, shape_slice, strides_slice).unwrap_or(-1);

        let result_wrapper =
            NDArrayWrapper::create_scalar_wrapper(result as f64, crate::dtype::DType::Int64);
        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));

        SUCCESS
    })
}

/// Compute the index of the minimum element for Int64 arrays.
/// Returns Int64 index.
#[no_mangle]
pub unsafe extern "C" fn ndarray_argmin_i64(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    // For now, delegate to regular argmin since we convert everything to f64 anyway
    ndarray_argmin(handle, offset, shape, strides, ndim, out_handle)
}
