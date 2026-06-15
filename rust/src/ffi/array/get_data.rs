//! Array data access FFI functions.

use std::ffi::c_void;
use std::slice;

use crate::helpers::error::{self, ERR_DTYPE, ERR_GENERIC, ERR_INDEX, SUCCESS};
use crate::helpers::{
    extract_array_bool, extract_array_c128, extract_array_c64, extract_array_f32,
    extract_array_f64, extract_array_i16, extract_array_i32, extract_array_i64, extract_array_i8,
    extract_array_u16, extract_array_u32, extract_array_u64, extract_array_u8,
};
use crate::types::dtype::DType;
use crate::types::{ArrayMetadata, NdArrayHandle};
use ndarray::ArrayD;

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
    meta: *const ArrayMetadata,
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
                let Some(arr) = extract_array_i8(wrapper, meta) else {
                    error::set_last_error("Failed to extract i8 view");
                    return ERR_DTYPE;
                };
                copy_array_to_buffer(&arr, start, len, out_data as *mut i8, out_len)
            }
            DType::Int16 => {
                let Some(arr) = extract_array_i16(wrapper, meta) else {
                    error::set_last_error("Failed to extract i16 view");
                    return ERR_DTYPE;
                };
                copy_array_to_buffer(&arr, start, len, out_data as *mut i16, out_len)
            }
            DType::Int32 => {
                let Some(arr) = extract_array_i32(wrapper, meta) else {
                    error::set_last_error("Failed to extract i32 view");
                    return ERR_DTYPE;
                };
                copy_array_to_buffer(&arr, start, len, out_data as *mut i32, out_len)
            }
            DType::Int64 => {
                let Some(arr) = extract_array_i64(wrapper, meta) else {
                    error::set_last_error("Failed to extract i64 view");
                    return ERR_DTYPE;
                };
                copy_array_to_buffer(&arr, start, len, out_data as *mut i64, out_len)
            }
            DType::Uint8 => {
                let Some(arr) = extract_array_u8(wrapper, meta) else {
                    error::set_last_error("Failed to extract u8 view");
                    return ERR_DTYPE;
                };
                copy_array_to_buffer(&arr, start, len, out_data as *mut u8, out_len)
            }
            DType::Uint16 => {
                let Some(arr) = extract_array_u16(wrapper, meta) else {
                    error::set_last_error("Failed to extract u16 view");
                    return ERR_DTYPE;
                };
                copy_array_to_buffer(&arr, start, len, out_data as *mut u16, out_len)
            }
            DType::Uint32 => {
                let Some(arr) = extract_array_u32(wrapper, meta) else {
                    error::set_last_error("Failed to extract u32 view");
                    return ERR_DTYPE;
                };
                copy_array_to_buffer(&arr, start, len, out_data as *mut u32, out_len)
            }
            DType::Uint64 => {
                let Some(arr) = extract_array_u64(wrapper, meta) else {
                    error::set_last_error("Failed to extract u64 view");
                    return ERR_DTYPE;
                };
                copy_array_to_buffer(&arr, start, len, out_data as *mut u64, out_len)
            }
            DType::Float32 => {
                let Some(arr) = extract_array_f32(wrapper, meta) else {
                    error::set_last_error("Failed to extract f32 view");
                    return ERR_DTYPE;
                };
                copy_array_to_buffer(&arr, start, len, out_data as *mut f32, out_len)
            }
            DType::Float64 => {
                let Some(arr) = extract_array_f64(wrapper, meta) else {
                    error::set_last_error("Failed to extract f64 view");
                    return ERR_DTYPE;
                };
                copy_array_to_buffer(&arr, start, len, out_data as *mut f64, out_len)
            }
            DType::Bool => {
                let Some(arr) = extract_array_bool(wrapper, meta) else {
                    error::set_last_error("Failed to extract bool view");
                    return ERR_DTYPE;
                };
                copy_array_to_buffer(&arr, start, len, out_data as *mut u8, out_len)
            }
            DType::Complex64 => {
                let Some(arr) = extract_array_c64(wrapper, meta) else {
                    error::set_last_error("Failed to extract Complex64 view");
                    return ERR_DTYPE;
                };
                copy_array_to_buffer(
                    &arr,
                    start,
                    len,
                    out_data as *mut num_complex::Complex32,
                    out_len,
                )
            }
            DType::Complex128 => {
                let Some(arr) = extract_array_c128(wrapper, meta) else {
                    error::set_last_error("Failed to extract Complex128 view");
                    return ERR_DTYPE;
                };
                copy_array_to_buffer(
                    &arr,
                    start,
                    len,
                    out_data as *mut num_complex::Complex64,
                    out_len,
                )
            }
        };

        result
    })
}

unsafe fn copy_array_to_buffer<T: Copy>(
    arr: &ArrayD<T>,
    start: usize,
    len: usize,
    out_data: *mut T,
    out_len: *mut usize,
) -> i32 {
    let total_len = arr.len();

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
        .zip(arr.iter().skip(start).take(copy_len))
    {
        *dst = *src;
    }

    *out_len = copy_len;
    SUCCESS
}
