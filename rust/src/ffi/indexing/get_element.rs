//! Element-level get operations.
//!
//! Provides type-specific get operations for single elements at a flat index.

use std::ffi::c_void;

use crate::dtype::DType;
use crate::error::{self, ERR_DTYPE, ERR_GENERIC, ERR_INDEX, SUCCESS};
use crate::ffi::NdArrayHandle;

/// Get an element at the given flat index.
///
/// # Arguments
/// * `handle` - Array handle
/// * `flat_index` - Flat index of the element
/// * `out_value` - Pointer to store the value (type depends on array dtype)
///
/// The caller must ensure `out_value` points to the correct type for the array's dtype.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_element(
    handle: *const NdArrayHandle,
    flat_index: usize,
    out_value: *mut c_void,
) -> i32 {
    if handle.is_null() || out_value.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);

        let result = match wrapper.dtype {
            DType::Int8 => match wrapper.get_element_i8(flat_index) {
                Ok(v) => {
                    *(out_value as *mut i8) = v;
                    SUCCESS
                }
                Err(e) => return handle_get_error(e),
            },
            DType::Int16 => match wrapper.get_element_i16(flat_index) {
                Ok(v) => {
                    *(out_value as *mut i16) = v;
                    SUCCESS
                }
                Err(e) => return handle_get_error(e),
            },
            DType::Int32 => match wrapper.get_element_i32(flat_index) {
                Ok(v) => {
                    *(out_value as *mut i32) = v;
                    SUCCESS
                }
                Err(e) => return handle_get_error(e),
            },
            DType::Int64 => match wrapper.get_element_i64(flat_index) {
                Ok(v) => {
                    *(out_value as *mut i64) = v;
                    SUCCESS
                }
                Err(e) => return handle_get_error(e),
            },
            DType::Uint8 => match wrapper.get_element_u8(flat_index) {
                Ok(v) => {
                    *(out_value as *mut u8) = v;
                    SUCCESS
                }
                Err(e) => return handle_get_error(e),
            },
            DType::Uint16 => match wrapper.get_element_u16(flat_index) {
                Ok(v) => {
                    *(out_value as *mut u16) = v;
                    SUCCESS
                }
                Err(e) => return handle_get_error(e),
            },
            DType::Uint32 => match wrapper.get_element_u32(flat_index) {
                Ok(v) => {
                    *(out_value as *mut u32) = v;
                    SUCCESS
                }
                Err(e) => return handle_get_error(e),
            },
            DType::Uint64 => match wrapper.get_element_u64(flat_index) {
                Ok(v) => {
                    *(out_value as *mut u64) = v;
                    SUCCESS
                }
                Err(e) => return handle_get_error(e),
            },
            DType::Float32 => match wrapper.get_element_f32(flat_index) {
                Ok(v) => {
                    *(out_value as *mut f32) = v;
                    SUCCESS
                }
                Err(e) => return handle_get_error(e),
            },
            DType::Float64 => match wrapper.get_element_f64(flat_index) {
                Ok(v) => {
                    *(out_value as *mut f64) = v;
                    SUCCESS
                }
                Err(e) => return handle_get_error(e),
            },
            DType::Bool => match wrapper.get_element_bool(flat_index) {
                Ok(v) => {
                    *(out_value as *mut u8) = v;
                    SUCCESS
                }
                Err(e) => return handle_get_error(e),
            },
        };

        result
    })
}

/// Helper to handle get_element errors
fn handle_get_error(e: String) -> i32 {
    if e.contains("Type mismatch") {
        error::set_last_error(e);
        ERR_DTYPE
    } else {
        error::set_last_error(e);
        ERR_INDEX
    }
}
