//! Array data access FFI functions.

use std::sync::Arc;
use std::ffi::c_void;
use std::slice;

use parking_lot::RwLock;

use crate::helpers::error::{self, ERR_DTYPE, ERR_GENERIC, ERR_INDEX, SUCCESS};
use crate::helpers::{
    extract_array_bool, extract_array_c128, extract_array_c64, extract_array_f32,
    extract_array_f64, extract_array_i16, extract_array_i32, extract_array_i64, extract_array_i8,
    extract_array_u16, extract_array_u32, extract_array_u64, extract_array_u8,
};
use crate::helpers::is_c_contiguous;
use crate::types::dtype::DType;
use crate::types::{ArrayData, ArrayMetadata, NdArrayHandle};
use ndarray::ArrayD;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a closure that extracts the typed `Arc<RwLock<ArrayD<T>>>` from an
/// `ArrayData` enum, returning `None` when the variant doesn't match.
macro_rules! extractor {
    ($variant:ident, $ty:ty) => {
        |d: &ArrayData| -> Option<&Arc<RwLock<ArrayD<$ty>>>> {
            if let ArrayData::$variant(arr) = d {
                Some(arr)
            } else {
                None
            }
        }
    };
}

/// Copy elements directly from the underlying contiguous buffer.
///
/// Returns `true` if the copy succeeded, `false` if the `ArrayData` variant
/// did not match (programmer error).
unsafe fn copy_contiguous<T: Copy>(
    data: &ArrayData,
    offset: usize,
    start: usize,
    copy_len: usize,
    out: *mut T,
    extract: impl Fn(&ArrayData) -> Option<&Arc<RwLock<ArrayD<T>>>>,
) -> bool {
    let Some(arr) = extract(data) else {
        return false;
    };
    let guard = arr.read();
    let src = slice::from_raw_parts(guard.as_ptr().add(offset + start), copy_len);
    slice::from_raw_parts_mut(out, copy_len).copy_from_slice(src);
    true
}

// ---------------------------------------------------------------------------
// FFI entry
// ---------------------------------------------------------------------------

/// Get flattened data for an array view with optional offset and length.
///
/// # Arguments
/// * `handle` - Array handle
/// * `meta` - View metadata
/// * `start` - Starting element offset (0-indexed)
/// * `len` - Number of elements to copy
/// * `out_data` - Pointer to output buffer (type must match array dtype)
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

        let shape = meta.shape_slice();
        let strides = meta.strides_slice();
        let total: usize = shape.iter().product();

        if start >= total {
            error::set_last_error("Start offset is out of bounds");
            return ERR_INDEX;
        }

        let copy_len = len.min(total - start);
        *out_len = copy_len;

        // ── Fast path: C-contiguous → direct memcpy (0 allocations, 1 copy) ──
        if is_c_contiguous(shape, strides) {
            let ok = match wrapper.dtype {
                DType::Int8 => copy_contiguous(
                    &wrapper.data, meta.offset, start, copy_len,
                    out_data as *mut i8, extractor!(Int8, i8),
                ),
                DType::Int16 => copy_contiguous(
                    &wrapper.data, meta.offset, start, copy_len,
                    out_data as *mut i16, extractor!(Int16, i16),
                ),
                DType::Int32 => copy_contiguous(
                    &wrapper.data, meta.offset, start, copy_len,
                    out_data as *mut i32, extractor!(Int32, i32),
                ),
                DType::Int64 => copy_contiguous(
                    &wrapper.data, meta.offset, start, copy_len,
                    out_data as *mut i64, extractor!(Int64, i64),
                ),
                DType::Uint8 => copy_contiguous(
                    &wrapper.data, meta.offset, start, copy_len,
                    out_data as *mut u8, extractor!(Uint8, u8),
                ),
                DType::Uint16 => copy_contiguous(
                    &wrapper.data, meta.offset, start, copy_len,
                    out_data as *mut u16, extractor!(Uint16, u16),
                ),
                DType::Uint32 => copy_contiguous(
                    &wrapper.data, meta.offset, start, copy_len,
                    out_data as *mut u32, extractor!(Uint32, u32),
                ),
                DType::Uint64 => copy_contiguous(
                    &wrapper.data, meta.offset, start, copy_len,
                    out_data as *mut u64, extractor!(Uint64, u64),
                ),
                DType::Float32 => copy_contiguous(
                    &wrapper.data, meta.offset, start, copy_len,
                    out_data as *mut f32, extractor!(Float32, f32),
                ),
                DType::Float64 => copy_contiguous(
                    &wrapper.data, meta.offset, start, copy_len,
                    out_data as *mut f64, extractor!(Float64, f64),
                ),
                DType::Bool => copy_contiguous(
                    &wrapper.data, meta.offset, start, copy_len,
                    out_data as *mut u8, extractor!(Bool, u8),
                ),
                DType::Complex64 => copy_contiguous(
                    &wrapper.data, meta.offset, start, copy_len,
                    out_data as *mut num_complex::Complex32,
                    extractor!(Complex64, num_complex::Complex32),
                ),
                DType::Complex128 => copy_contiguous(
                    &wrapper.data, meta.offset, start, copy_len,
                    out_data as *mut num_complex::Complex64,
                    extractor!(Complex128, num_complex::Complex64),
                ),
            };
            if ok {
                return SUCCESS;
            }
        }

        // ── Slow path: non-contiguous views → materialise, then copy ──
        let result = match wrapper.dtype {
            DType::Int8 => {
                let Some(arr) = extract_array_i8(wrapper, meta) else {
                    error::set_last_error("Failed to extract i8 view");
                    return ERR_DTYPE;
                };
                copy_array_to_buffer(&arr, start, copy_len, out_data as *mut i8)
            }
            DType::Int16 => {
                let Some(arr) = extract_array_i16(wrapper, meta) else {
                    error::set_last_error("Failed to extract i16 view");
                    return ERR_DTYPE;
                };
                copy_array_to_buffer(&arr, start, copy_len, out_data as *mut i16)
            }
            DType::Int32 => {
                let Some(arr) = extract_array_i32(wrapper, meta) else {
                    error::set_last_error("Failed to extract i32 view");
                    return ERR_DTYPE;
                };
                copy_array_to_buffer(&arr, start, copy_len, out_data as *mut i32)
            }
            DType::Int64 => {
                let Some(arr) = extract_array_i64(wrapper, meta) else {
                    error::set_last_error("Failed to extract i64 view");
                    return ERR_DTYPE;
                };
                copy_array_to_buffer(&arr, start, copy_len, out_data as *mut i64)
            }
            DType::Uint8 => {
                let Some(arr) = extract_array_u8(wrapper, meta) else {
                    error::set_last_error("Failed to extract u8 view");
                    return ERR_DTYPE;
                };
                copy_array_to_buffer(&arr, start, copy_len, out_data as *mut u8)
            }
            DType::Uint16 => {
                let Some(arr) = extract_array_u16(wrapper, meta) else {
                    error::set_last_error("Failed to extract u16 view");
                    return ERR_DTYPE;
                };
                copy_array_to_buffer(&arr, start, copy_len, out_data as *mut u16)
            }
            DType::Uint32 => {
                let Some(arr) = extract_array_u32(wrapper, meta) else {
                    error::set_last_error("Failed to extract u32 view");
                    return ERR_DTYPE;
                };
                copy_array_to_buffer(&arr, start, copy_len, out_data as *mut u32)
            }
            DType::Uint64 => {
                let Some(arr) = extract_array_u64(wrapper, meta) else {
                    error::set_last_error("Failed to extract u64 view");
                    return ERR_DTYPE;
                };
                copy_array_to_buffer(&arr, start, copy_len, out_data as *mut u64)
            }
            DType::Float32 => {
                let Some(arr) = extract_array_f32(wrapper, meta) else {
                    error::set_last_error("Failed to extract f32 view");
                    return ERR_DTYPE;
                };
                copy_array_to_buffer(&arr, start, copy_len, out_data as *mut f32)
            }
            DType::Float64 => {
                let Some(arr) = extract_array_f64(wrapper, meta) else {
                    error::set_last_error("Failed to extract f64 view");
                    return ERR_DTYPE;
                };
                copy_array_to_buffer(&arr, start, copy_len, out_data as *mut f64)
            }
            DType::Bool => {
                let Some(arr) = extract_array_bool(wrapper, meta) else {
                    error::set_last_error("Failed to extract bool view");
                    return ERR_DTYPE;
                };
                copy_array_to_buffer(&arr, start, copy_len, out_data as *mut u8)
            }
            DType::Complex64 => {
                let Some(arr) = extract_array_c64(wrapper, meta) else {
                    error::set_last_error("Failed to extract Complex64 view");
                    return ERR_DTYPE;
                };
                copy_array_to_buffer(&arr, start, copy_len, out_data as *mut num_complex::Complex32)
            }
            DType::Complex128 => {
                let Some(arr) = extract_array_c128(wrapper, meta) else {
                    error::set_last_error("Failed to extract Complex128 view");
                    return ERR_DTYPE;
                };
                copy_array_to_buffer(&arr, start, copy_len, out_data as *mut num_complex::Complex64)
            }
        };

        result
    })
}

// ---------------------------------------------------------------------------
// Slow-path copy
// ---------------------------------------------------------------------------

/// Copy a range of elements from an extracted `ArrayD` into a flat buffer.
///
/// Uses `copy_from_slice` (memcpy) when the array is contiguous, or
/// stride-based element iteration otherwise.
unsafe fn copy_array_to_buffer<T: Copy>(
    arr: &ArrayD<T>,
    start: usize,
    len: usize,
    out: *mut T,
) -> i32 {
    let total = arr.len();

    if start >= total {
        error::set_last_error("Start offset is out of bounds");
        return ERR_INDEX;
    }

    let copy_len = len.min(total - start);
    let out_slice = slice::from_raw_parts_mut(out, copy_len);

    if let Some(data) = arr.as_slice() {
        out_slice.copy_from_slice(&data[start..start + copy_len]);
    } else {
        for (dst, src) in out_slice.iter_mut().zip(arr.iter().skip(start).take(copy_len)) {
            *dst = *src;
        }
    }

    SUCCESS
}
