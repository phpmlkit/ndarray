//! Scalar product reduction.

use crate::core::view_helpers::{
    extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32, extract_view_i64,
    extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64, extract_view_u8,
};
use crate::core::NDArrayWrapper;
use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::NdArrayHandle;

/// Compute product using ndarray's native product() method.
///
/// Works with all types that implement Mul + One traits.
fn compute_product(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> f64 {
    unsafe {
        // Float64 - native support
        if let Some(view) = extract_view_f64(wrapper, offset, shape, strides) {
            return view.product();
        }
        // Float32 - native support
        if let Some(view) = extract_view_f32(wrapper, offset, shape, strides) {
            return view.product() as f64;
        }
        // Int64 - native support via Mul + One
        if let Some(view) = extract_view_i64(wrapper, offset, shape, strides) {
            return view.product() as f64;
        }
        // Int32
        if let Some(view) = extract_view_i32(wrapper, offset, shape, strides) {
            return view.product() as f64;
        }
        // Int16
        if let Some(view) = extract_view_i16(wrapper, offset, shape, strides) {
            return view.product() as f64;
        }
        // Int8
        if let Some(view) = extract_view_i8(wrapper, offset, shape, strides) {
            return view.product() as f64;
        }
        // Uint64
        if let Some(view) = extract_view_u64(wrapper, offset, shape, strides) {
            return view.product() as f64;
        }
        // Uint32
        if let Some(view) = extract_view_u32(wrapper, offset, shape, strides) {
            return view.product() as f64;
        }
        // Uint16
        if let Some(view) = extract_view_u16(wrapper, offset, shape, strides) {
            return view.product() as f64;
        }
        // Uint8
        if let Some(view) = extract_view_u8(wrapper, offset, shape, strides) {
            return view.product() as f64;
        }
    }
    1.0
}

/// Compute the product of all elements in the array (scalar reduction).
#[no_mangle]
pub unsafe extern "C" fn ndarray_product(
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

        let result = compute_product(wrapper, offset, shape_slice, strides_slice);
        let result_wrapper = NDArrayWrapper::create_scalar_wrapper(result, wrapper.dtype);
        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));

        SUCCESS
    })
}
