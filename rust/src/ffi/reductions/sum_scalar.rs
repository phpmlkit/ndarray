//! Scalar sum reduction.

use crate::core::view_helpers::{
    create_scalar_wrapper_with_dtype, extract_view_f32, extract_view_f64, extract_view_i16,
    extract_view_i32, extract_view_i64, extract_view_i8, extract_view_u16, extract_view_u32,
    extract_view_u64, extract_view_u8,
};
use crate::core::{ArrayData, NDArrayWrapper};
use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::NdArrayHandle;

/// Compute sum of elements using proper strided view.
///
/// For integer types, computes exact integer sum.
/// For float types, computes native float sum.
fn compute_sum(wrapper: &NDArrayWrapper, offset: usize, shape: &[usize], strides: &[usize]) -> f64 {
    unsafe {
        // Try Float64 first (fastest path)
        if let Some(view) = extract_view_f64(wrapper, offset, shape, strides) {
            return view.sum();
        }
        // Float32
        if let Some(view) = extract_view_f32(wrapper, offset, shape, strides) {
            return view.mapv(|x| x as f64).sum();
        }
        // Int64 - exact integer sum, convert result to f64
        if let Some(view) = extract_view_i64(wrapper, offset, shape, strides) {
            return view.iter().map(|&x| x as f64).sum();
        }
        // Int32 - exact integer sum, convert result to f64
        if let Some(view) = extract_view_i32(wrapper, offset, shape, strides) {
            return view.iter().map(|&x| x as f64).sum();
        }
        // Int16
        if let Some(view) = extract_view_i16(wrapper, offset, shape, strides) {
            return view.iter().map(|&x| x as f64).sum();
        }
        // Int8
        if let Some(view) = extract_view_i8(wrapper, offset, shape, strides) {
            return view.iter().map(|&x| x as f64).sum();
        }
        // Uint64 - exact integer sum, convert result to f64
        if let Some(view) = extract_view_u64(wrapper, offset, shape, strides) {
            return view.iter().map(|&x| x as f64).sum();
        }
        // Uint32
        if let Some(view) = extract_view_u32(wrapper, offset, shape, strides) {
            return view.iter().map(|&x| x as f64).sum();
        }
        // Uint16
        if let Some(view) = extract_view_u16(wrapper, offset, shape, strides) {
            return view.iter().map(|&x| x as f64).sum();
        }
        // Uint8
        if let Some(view) = extract_view_u8(wrapper, offset, shape, strides) {
            return view.iter().map(|&x| x as f64).sum();
        }
    }
    0.0
}

/// Compute the sum of all elements in the array (scalar reduction).
///
/// This function properly handles strided views (non-contiguous arrays)
/// by using the strides parameter to correctly calculate element positions.
#[no_mangle]
pub unsafe extern "C" fn ndarray_sum(
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

        // Compute sum using proper strided view
        let result = compute_sum(wrapper, offset, shape_slice, strides_slice);

        // Create scalar result wrapper with same dtype as input
        let result_wrapper = create_scalar_wrapper_with_dtype(result, wrapper.dtype);
        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));

        SUCCESS
    })
}
