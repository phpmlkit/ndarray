//! Array data access FFI functions.

use std::ffi::c_void;
use std::slice;

use crate::core::view_helpers::{
    extract_view_bool, extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32,
    extract_view_i64, extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64,
    extract_view_u8,
};
use crate::dtype::DType;
use crate::error::{self, ERR_DTYPE, ERR_GENERIC, SUCCESS};
use crate::ffi::NdArrayHandle;
use ndarray::ArrayViewD;

/// Get flattened data for an arbitrary array view (offset/shape/strides aware).
///
/// # Arguments
/// * `handle` - Array handle
/// * `offset` - View offset in elements
/// * `shape` - View shape pointer
/// * `strides` - View strides pointer
/// * `ndim` - Number of dimensions
/// * `out_data` - Pointer to output buffer (type depends on array dtype)
/// * `max_len` - Maximum number of elements to copy
/// * `out_len` - Output: actual number of elements in the view
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_data(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
    out_data: *mut c_void,
    max_len: usize,
    out_len: *mut usize,
) -> i32 {
    if handle.is_null()
        || shape.is_null()
        || strides.is_null()
        || out_data.is_null()
        || out_len.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = slice::from_raw_parts(shape, ndim);
        let strides_slice = slice::from_raw_parts(strides, ndim);

        let result = match wrapper.dtype {
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, offset, shape_slice, strides_slice) else {
                    error::set_last_error("Failed to extract i8 view");
                    return ERR_DTYPE;
                };
                copy_view_to_buffer(view, out_data as *mut i8, max_len, out_len)
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, offset, shape_slice, strides_slice) else {
                    error::set_last_error("Failed to extract i16 view");
                    return ERR_DTYPE;
                };
                copy_view_to_buffer(view, out_data as *mut i16, max_len, out_len)
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, offset, shape_slice, strides_slice) else {
                    error::set_last_error("Failed to extract i32 view");
                    return ERR_DTYPE;
                };
                copy_view_to_buffer(view, out_data as *mut i32, max_len, out_len)
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, offset, shape_slice, strides_slice) else {
                    error::set_last_error("Failed to extract i64 view");
                    return ERR_DTYPE;
                };
                copy_view_to_buffer(view, out_data as *mut i64, max_len, out_len)
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, offset, shape_slice, strides_slice) else {
                    error::set_last_error("Failed to extract u8 view");
                    return ERR_DTYPE;
                };
                copy_view_to_buffer(view, out_data as *mut u8, max_len, out_len)
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, offset, shape_slice, strides_slice) else {
                    error::set_last_error("Failed to extract u16 view");
                    return ERR_DTYPE;
                };
                copy_view_to_buffer(view, out_data as *mut u16, max_len, out_len)
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, offset, shape_slice, strides_slice) else {
                    error::set_last_error("Failed to extract u32 view");
                    return ERR_DTYPE;
                };
                copy_view_to_buffer(view, out_data as *mut u32, max_len, out_len)
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, offset, shape_slice, strides_slice) else {
                    error::set_last_error("Failed to extract u64 view");
                    return ERR_DTYPE;
                };
                copy_view_to_buffer(view, out_data as *mut u64, max_len, out_len)
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, offset, shape_slice, strides_slice) else {
                    error::set_last_error("Failed to extract f32 view");
                    return ERR_DTYPE;
                };
                copy_view_to_buffer(view, out_data as *mut f32, max_len, out_len)
            }
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, offset, shape_slice, strides_slice) else {
                    error::set_last_error("Failed to extract f64 view");
                    return ERR_DTYPE;
                };
                copy_view_to_buffer(view, out_data as *mut f64, max_len, out_len)
            }
            DType::Bool => {
                let Some(view) = extract_view_bool(wrapper, offset, shape_slice, strides_slice) else {
                    error::set_last_error("Failed to extract bool view");
                    return ERR_DTYPE;
                };
                copy_view_to_buffer(view, out_data as *mut u8, max_len, out_len)
            }
        };

        result
    })
}

unsafe fn copy_view_to_buffer<T: Copy>(
    view: ArrayViewD<'_, T>,
    out_data: *mut T,
    max_len: usize,
    out_len: *mut usize,
) -> i32 {
    let total_len = view.len();
    let copy_len = total_len.min(max_len);
    let out_slice = slice::from_raw_parts_mut(out_data, copy_len);
    for (dst, src) in out_slice.iter_mut().zip(view.iter().take(copy_len)) {
        *dst = *src;
    }
    *out_len = total_len;
    SUCCESS
}
