//! Element-level set operations.
//!
//! Provides unified set operations for single elements at a flat index.

use std::ffi::c_void;

use crate::dtype::DType;
use crate::error::{self, ERR_DTYPE, ERR_GENERIC, ERR_INDEX, SUCCESS};
use crate::ffi::NdArrayHandle;

/// Set an element at the given flat index.
///
/// # Arguments
/// * `handle` - Array handle
/// * `flat_index` - Flat index of the element
/// * `value` - Pointer to the value (type depends on array dtype)
///
/// The caller must ensure `value` points to the correct type for the array's dtype.
#[no_mangle]
pub unsafe extern "C" fn ndarray_set_element(
    handle: *mut NdArrayHandle,
    flat_index: usize,
    value: *const c_void,
) -> i32 {
    if handle.is_null() || value.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle);

        let result = match wrapper.dtype {
            DType::Int8 => {
                let v = *(value as *const i8);
                wrapper.set_element_i8(flat_index, v)
            }
            DType::Int16 => {
                let v = *(value as *const i16);
                wrapper.set_element_i16(flat_index, v)
            }
            DType::Int32 => {
                let v = *(value as *const i32);
                wrapper.set_element_i32(flat_index, v)
            }
            DType::Int64 => {
                let v = *(value as *const i64);
                wrapper.set_element_i64(flat_index, v)
            }
            DType::Uint8 => {
                let v = *(value as *const u8);
                wrapper.set_element_u8(flat_index, v)
            }
            DType::Uint16 => {
                let v = *(value as *const u16);
                wrapper.set_element_u16(flat_index, v)
            }
            DType::Uint32 => {
                let v = *(value as *const u32);
                wrapper.set_element_u32(flat_index, v)
            }
            DType::Uint64 => {
                let v = *(value as *const u64);
                wrapper.set_element_u64(flat_index, v)
            }
            DType::Float32 => {
                let v = *(value as *const f32);
                wrapper.set_element_f32(flat_index, v)
            }
            DType::Float64 => {
                let v = *(value as *const f64);
                wrapper.set_element_f64(flat_index, v)
            }
            DType::Bool => {
                let v = *(value as *const u8);
                wrapper.set_element_bool(flat_index, v)
            }
        };

        match result {
            Ok(()) => SUCCESS,
            Err(e) => {
                if e.contains("Type mismatch") {
                    error::set_last_error(e);
                    ERR_DTYPE
                } else {
                    error::set_last_error(e);
                    ERR_INDEX
                }
            }
        }
    })
}
