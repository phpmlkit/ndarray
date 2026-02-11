//! Array data access FFI functions.

use std::ffi::c_void;
use std::slice;

use crate::dtype::DType;
use crate::error::{self, ERR_DTYPE, ERR_GENERIC, SUCCESS};
use crate::ffi::NdArrayHandle;

/// Get array data.
///
/// # Arguments
/// * `handle` - Array handle
/// * `out_data` - Pointer to output buffer (type depends on array dtype)
/// * `max_len` - Maximum number of elements to copy
/// * `out_len` - Output: actual number of elements in the array
///
/// The caller must ensure `out_data` points to a buffer of the correct type
/// and sufficient size for the array's dtype.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_data(
    handle: *const NdArrayHandle,
    out_data: *mut c_void,
    max_len: usize,
    out_len: *mut usize,
) -> i32 {
    if handle.is_null() || out_data.is_null() || out_len.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);

        let result = match wrapper.dtype {
            DType::Int8 => {
                if let Some(data) = wrapper.to_int8_vec() {
                    copy_data_to_buffer(data, out_data as *mut i8, max_len, out_len)
                } else {
                    error::set_last_error("Data type mismatch");
                    ERR_DTYPE
                }
            }
            DType::Int16 => {
                if let Some(data) = wrapper.to_int16_vec() {
                    copy_data_to_buffer(data, out_data as *mut i16, max_len, out_len)
                } else {
                    error::set_last_error("Data type mismatch");
                    ERR_DTYPE
                }
            }
            DType::Int32 => {
                if let Some(data) = wrapper.to_int32_vec() {
                    copy_data_to_buffer(data, out_data as *mut i32, max_len, out_len)
                } else {
                    error::set_last_error("Data type mismatch");
                    ERR_DTYPE
                }
            }
            DType::Int64 => {
                if let Some(data) = wrapper.to_int64_vec() {
                    copy_data_to_buffer(data, out_data as *mut i64, max_len, out_len)
                } else {
                    error::set_last_error("Data type mismatch");
                    ERR_DTYPE
                }
            }
            DType::Uint8 => {
                if let Some(data) = wrapper.to_uint8_vec() {
                    copy_data_to_buffer(data, out_data as *mut u8, max_len, out_len)
                } else {
                    error::set_last_error("Data type mismatch");
                    ERR_DTYPE
                }
            }
            DType::Uint16 => {
                if let Some(data) = wrapper.to_uint16_vec() {
                    copy_data_to_buffer(data, out_data as *mut u16, max_len, out_len)
                } else {
                    error::set_last_error("Data type mismatch");
                    ERR_DTYPE
                }
            }
            DType::Uint32 => {
                if let Some(data) = wrapper.to_uint32_vec() {
                    copy_data_to_buffer(data, out_data as *mut u32, max_len, out_len)
                } else {
                    error::set_last_error("Data type mismatch");
                    ERR_DTYPE
                }
            }
            DType::Uint64 => {
                if let Some(data) = wrapper.to_uint64_vec() {
                    copy_data_to_buffer(data, out_data as *mut u64, max_len, out_len)
                } else {
                    error::set_last_error("Data type mismatch");
                    ERR_DTYPE
                }
            }
            DType::Float32 => {
                if let Some(data) = wrapper.to_float32_vec() {
                    copy_data_to_buffer(data, out_data as *mut f32, max_len, out_len)
                } else {
                    error::set_last_error("Data type mismatch");
                    ERR_DTYPE
                }
            }
            DType::Float64 => {
                if let Some(data) = wrapper.to_float64_vec() {
                    copy_data_to_buffer(data, out_data as *mut f64, max_len, out_len)
                } else {
                    error::set_last_error("Data type mismatch");
                    ERR_DTYPE
                }
            }
            DType::Bool => {
                if let Some(data) = wrapper.to_bool_vec() {
                    copy_data_to_buffer(data, out_data as *mut u8, max_len, out_len)
                } else {
                    error::set_last_error("Data type mismatch");
                    ERR_DTYPE
                }
            }
        };

        result
    })
}

/// Helper to copy data to output buffer
unsafe fn copy_data_to_buffer<T: Copy>(
    data: Vec<T>,
    out_data: *mut T,
    max_len: usize,
    out_len: *mut usize,
) -> i32 {
    let len = data.len().min(max_len);
    let out_slice = slice::from_raw_parts_mut(out_data, len);
    out_slice.copy_from_slice(&data[..len]);
    *out_len = data.len();
    SUCCESS
}
