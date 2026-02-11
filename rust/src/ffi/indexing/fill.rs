//! Slice fill operations.
//!
//! Provides unified fill operations for strided array views.

use std::ffi::c_void;
use std::slice;

use crate::dtype::DType;
use crate::error::{self, ERR_GENERIC, SUCCESS};
use crate::ffi::NdArrayHandle;

/// Fill a slice with a value.
///
/// # Arguments
/// * `handle` - Array handle
/// * `value` - Pointer to the fill value (type depends on array dtype)
/// * `offset` - Starting offset in the array
/// * `shape` - Shape of the slice to fill
/// * `strides` - Strides of the slice
/// * `ndim` - Number of dimensions
#[no_mangle]
pub unsafe extern "C" fn ndarray_fill(
    handle: *const NdArrayHandle,
    value: *const c_void,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
) -> i32 {
    if handle.is_null() || value.is_null() || shape.is_null() || strides.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = slice::from_raw_parts(shape, ndim);
        let strides_slice = slice::from_raw_parts(strides, ndim);

        let result = match wrapper.dtype {
            DType::Int8 => {
                let v = *(value as *const i8);
                wrapper.fill_slice_i8(v, offset, shape_slice, strides_slice)
            }
            DType::Int16 => {
                let v = *(value as *const i16);
                wrapper.fill_slice_i16(v, offset, shape_slice, strides_slice)
            }
            DType::Int32 => {
                let v = *(value as *const i32);
                wrapper.fill_slice_i32(v, offset, shape_slice, strides_slice)
            }
            DType::Int64 => {
                let v = *(value as *const i64);
                wrapper.fill_slice_i64(v, offset, shape_slice, strides_slice)
            }
            DType::Uint8 => {
                let v = *(value as *const u8);
                wrapper.fill_slice_u8(v, offset, shape_slice, strides_slice)
            }
            DType::Uint16 => {
                let v = *(value as *const u16);
                wrapper.fill_slice_u16(v, offset, shape_slice, strides_slice)
            }
            DType::Uint32 => {
                let v = *(value as *const u32);
                wrapper.fill_slice_u32(v, offset, shape_slice, strides_slice)
            }
            DType::Uint64 => {
                let v = *(value as *const u64);
                wrapper.fill_slice_u64(v, offset, shape_slice, strides_slice)
            }
            DType::Float32 => {
                let v = *(value as *const f32);
                wrapper.fill_slice_f32(v, offset, shape_slice, strides_slice)
            }
            DType::Float64 => {
                let v = *(value as *const f64);
                wrapper.fill_slice_f64(v, offset, shape_slice, strides_slice)
            }
            DType::Bool => {
                let v = *(value as *const u8);
                wrapper.fill_slice_bool(v, offset, shape_slice, strides_slice)
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
