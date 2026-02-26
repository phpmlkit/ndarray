//! Array data access FFI functions.

use std::ffi::c_void;
use std::slice;

use crate::core::view_helpers::{
    extract_view_bool, extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32,
    extract_view_i64, extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64,
    extract_view_u8,
};
use crate::dtype::DType;
use crate::error::{self, ERR_DTYPE, ERR_GENERIC, ERR_INDEX, SUCCESS};
use crate::ffi::{NdArrayHandle, ViewMetadata};
use ndarray::ArrayViewD;

/// Get flattened data for an array view with optional offset and length.
///
/// # Arguments
/// * `handle` - Array handle
/// * `meta` - View metadata
/// * `start` - Starting element offset (0-indexed)
/// * `len` - Number of elements to copy
/// * `out_data` - Pointer to output buffer (type depends on array dtype)
/// * `out_len` - Output: actual number of elements copied
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_data(
    handle: *const NdArrayHandle,
    meta: *const ViewMetadata,
    start: usize,
    len: usize,
    out_data: *mut c_void,
    out_len: *mut usize,
) -> i32 {
    if handle.is_null() || meta.is_null() || out_data.is_null() || out_len.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let meta = &*meta;

        let result = match wrapper.dtype {
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, meta) else {
                    error::set_last_error("Failed to extract i8 view");
                    return ERR_DTYPE;
                };
                copy_view_to_buffer(view, start, len, out_data as *mut i8, out_len)
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, meta) else {
                    error::set_last_error("Failed to extract i16 view");
                    return ERR_DTYPE;
                };
                copy_view_to_buffer(view, start, len, out_data as *mut i16, out_len)
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, meta) else {
                    error::set_last_error("Failed to extract i32 view");
                    return ERR_DTYPE;
                };
                copy_view_to_buffer(view, start, len, out_data as *mut i32, out_len)
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, meta) else {
                    error::set_last_error("Failed to extract i64 view");
                    return ERR_DTYPE;
                };
                copy_view_to_buffer(view, start, len, out_data as *mut i64, out_len)
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, meta) else {
                    error::set_last_error("Failed to extract u8 view");
                    return ERR_DTYPE;
                };
                copy_view_to_buffer(view, start, len, out_data as *mut u8, out_len)
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, meta) else {
                    error::set_last_error("Failed to extract u16 view");
                    return ERR_DTYPE;
                };
                copy_view_to_buffer(view, start, len, out_data as *mut u16, out_len)
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, meta) else {
                    error::set_last_error("Failed to extract u32 view");
                    return ERR_DTYPE;
                };
                copy_view_to_buffer(view, start, len, out_data as *mut u32, out_len)
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, meta) else {
                    error::set_last_error("Failed to extract u64 view");
                    return ERR_DTYPE;
                };
                copy_view_to_buffer(view, start, len, out_data as *mut u64, out_len)
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, meta) else {
                    error::set_last_error("Failed to extract f32 view");
                    return ERR_DTYPE;
                };
                copy_view_to_buffer(view, start, len, out_data as *mut f32, out_len)
            }
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, meta) else {
                    error::set_last_error("Failed to extract f64 view");
                    return ERR_DTYPE;
                };
                copy_view_to_buffer(view, start, len, out_data as *mut f64, out_len)
            }
            DType::Bool => {
                let Some(view) = extract_view_bool(wrapper, meta) else {
                    error::set_last_error("Failed to extract bool view");
                    return ERR_DTYPE;
                };
                copy_view_to_buffer(view, start, len, out_data as *mut u8, out_len)
            }
        };

        result
    })
}

unsafe fn copy_view_to_buffer<T: Copy>(
    view: ArrayViewD<'_, T>,
    start: usize,
    len: usize,
    out_data: *mut T,
    out_len: *mut usize,
) -> i32 {
    let total_len = view.len();

    // Validate start offset
    if start >= total_len {
        error::set_last_error("Start offset is out of bounds");
        return ERR_INDEX;
    }

    // Calculate actual copy length
    let available = total_len - start;
    let copy_len = len.min(available);

    let out_slice = slice::from_raw_parts_mut(out_data, copy_len);

    // Skip to start position, then take len elements
    for (dst, src) in out_slice
        .iter_mut()
        .zip(view.iter().skip(start).take(copy_len))
    {
        *dst = *src;
    }

    *out_len = copy_len;
    SUCCESS
}
