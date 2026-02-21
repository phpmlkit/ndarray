//! Slice fill operations.
//!
//! Provides unified fill operations for strided array views.

use std::ffi::c_void;

use crate::dtype::DType;
use crate::error::{self, ERR_GENERIC, SUCCESS};
use crate::ffi::{NdArrayHandle, ViewMetadata};

/// Fill a slice with a value.
///
/// # Arguments
/// * `handle` - Array handle
/// * `meta` - View metadata
/// * `value` - Pointer to the fill value (type depends on array dtype)
#[no_mangle]
pub unsafe extern "C" fn ndarray_fill(
    handle: *const NdArrayHandle,
    meta: *const ViewMetadata,
    value: *const c_void,
) -> i32 {
    if handle.is_null() || value.is_null() || meta.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let meta = &*meta;

        let result = match wrapper.dtype {
            DType::Int8 => {
                let v = *(value as *const i8);
                wrapper.fill_slice_i8(v, meta)
            }
            DType::Int16 => {
                let v = *(value as *const i16);
                wrapper.fill_slice_i16(v, meta)
            }
            DType::Int32 => {
                let v = *(value as *const i32);
                wrapper.fill_slice_i32(v, meta)
            }
            DType::Int64 => {
                let v = *(value as *const i64);
                wrapper.fill_slice_i64(v, meta)
            }
            DType::Uint8 => {
                let v = *(value as *const u8);
                wrapper.fill_slice_u8(v, meta)
            }
            DType::Uint16 => {
                let v = *(value as *const u16);
                wrapper.fill_slice_u16(v, meta)
            }
            DType::Uint32 => {
                let v = *(value as *const u32);
                wrapper.fill_slice_u32(v, meta)
            }
            DType::Uint64 => {
                let v = *(value as *const u64);
                wrapper.fill_slice_u64(v, meta)
            }
            DType::Float32 => {
                let v = *(value as *const f32);
                wrapper.fill_slice_f32(v, meta)
            }
            DType::Float64 => {
                let v = *(value as *const f64);
                wrapper.fill_slice_f64(v, meta)
            }
            DType::Bool => {
                let v = *(value as *const u8);
                wrapper.fill_slice_bool(v, meta)
            }
        };

        match result {
            Ok(()) => SUCCESS,
            Err(e) => {
                error::set_last_error(e);
                ERR_GENERIC
            }
        }
    })
}
