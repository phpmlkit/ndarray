//! Slice fill operations.
//!
//! Provides type-specific fill operations for strided array views.

use crate::core::NDArrayWrapper;
use crate::error::{self, ERR_GENERIC, SUCCESS};
use crate::ffi::NdArrayHandle;
use std::slice;

/// Helper for filling slice values.
pub unsafe fn fill_slice_helper<T, F>(
    handle: *const NdArrayHandle,
    value: T,
    offset: usize,
    shape_ptr: *const usize,
    strides_ptr: *const usize,
    ndim: usize,
    method: F,
) -> i32
where
    T: Copy,
    F: Fn(&NDArrayWrapper, T, usize, &[usize], &[usize]) -> Result<(), String>
        + std::panic::UnwindSafe,
{
    if handle.is_null() || shape_ptr.is_null() || strides_ptr.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape = slice::from_raw_parts(shape_ptr, ndim);
        let strides = slice::from_raw_parts(strides_ptr, ndim);

        match method(wrapper, value, offset, shape, strides) {
            Ok(()) => SUCCESS,
            Err(e) => {
                error::set_last_error(e);
                ERR_GENERIC
            }
        }
    })
}

/// Fill a slice with an int8 value.
#[no_mangle]
pub unsafe extern "C" fn ndarray_fill_int8(
    handle: *const NdArrayHandle,
    value: i8,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
) -> i32 {
    fill_slice_helper(
        handle,
        value,
        offset,
        shape,
        strides,
        ndim,
        |w, v, o, s, st| w.fill_slice_i8(v, o, s, st),
    )
}

/// Fill a slice with an int16 value.
#[no_mangle]
pub unsafe extern "C" fn ndarray_fill_int16(
    handle: *const NdArrayHandle,
    value: i16,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
) -> i32 {
    fill_slice_helper(
        handle,
        value,
        offset,
        shape,
        strides,
        ndim,
        |w, v, o, s, st| w.fill_slice_i16(v, o, s, st),
    )
}

/// Fill a slice with an int32 value.
#[no_mangle]
pub unsafe extern "C" fn ndarray_fill_int32(
    handle: *const NdArrayHandle,
    value: i32,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
) -> i32 {
    fill_slice_helper(
        handle,
        value,
        offset,
        shape,
        strides,
        ndim,
        |w, v, o, s, st| w.fill_slice_i32(v, o, s, st),
    )
}

/// Fill a slice with an int64 value.
#[no_mangle]
pub unsafe extern "C" fn ndarray_fill_int64(
    handle: *const NdArrayHandle,
    value: i64,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
) -> i32 {
    fill_slice_helper(
        handle,
        value,
        offset,
        shape,
        strides,
        ndim,
        |w, v, o, s, st| w.fill_slice_i64(v, o, s, st),
    )
}

/// Fill a slice with a uint8 value.
#[no_mangle]
pub unsafe extern "C" fn ndarray_fill_uint8(
    handle: *const NdArrayHandle,
    value: u8,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
) -> i32 {
    fill_slice_helper(
        handle,
        value,
        offset,
        shape,
        strides,
        ndim,
        |w, v, o, s, st| w.fill_slice_u8(v, o, s, st),
    )
}

/// Fill a slice with a uint16 value.
#[no_mangle]
pub unsafe extern "C" fn ndarray_fill_uint16(
    handle: *const NdArrayHandle,
    value: u16,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
) -> i32 {
    fill_slice_helper(
        handle,
        value,
        offset,
        shape,
        strides,
        ndim,
        |w, v, o, s, st| w.fill_slice_u16(v, o, s, st),
    )
}

/// Fill a slice with a uint32 value.
#[no_mangle]
pub unsafe extern "C" fn ndarray_fill_uint32(
    handle: *const NdArrayHandle,
    value: u32,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
) -> i32 {
    fill_slice_helper(
        handle,
        value,
        offset,
        shape,
        strides,
        ndim,
        |w, v, o, s, st| w.fill_slice_u32(v, o, s, st),
    )
}

/// Fill a slice with a uint64 value.
#[no_mangle]
pub unsafe extern "C" fn ndarray_fill_uint64(
    handle: *const NdArrayHandle,
    value: u64,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
) -> i32 {
    fill_slice_helper(
        handle,
        value,
        offset,
        shape,
        strides,
        ndim,
        |w, v, o, s, st| w.fill_slice_u64(v, o, s, st),
    )
}

/// Fill a slice with a float32 value.
#[no_mangle]
pub unsafe extern "C" fn ndarray_fill_float32(
    handle: *const NdArrayHandle,
    value: f32,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
) -> i32 {
    fill_slice_helper(
        handle,
        value,
        offset,
        shape,
        strides,
        ndim,
        |w, v, o, s, st| w.fill_slice_f32(v, o, s, st),
    )
}

/// Fill a slice with a float64 value.
#[no_mangle]
pub unsafe extern "C" fn ndarray_fill_float64(
    handle: *const NdArrayHandle,
    value: f64,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
) -> i32 {
    fill_slice_helper(
        handle,
        value,
        offset,
        shape,
        strides,
        ndim,
        |w, v, o, s, st| w.fill_slice_f64(v, o, s, st),
    )
}

/// Fill a slice with a bool value (passed as u8).
#[no_mangle]
pub unsafe extern "C" fn ndarray_fill_bool(
    handle: *const NdArrayHandle,
    value: u8,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
) -> i32 {
    fill_slice_helper(
        handle,
        value,
        offset,
        shape,
        strides,
        ndim,
        |w, v, o, s, st| w.fill_slice_bool(v, o, s, st),
    )
}
