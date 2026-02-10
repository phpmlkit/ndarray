//! Array creation FFI functions.

use std::slice;

use crate::core::NDArrayWrapper;
use crate::error::{self, ERR_GENERIC, ERR_PANIC, ERR_SHAPE, SUCCESS};
use crate::ffi::NdArrayHandle;

// ============================================================================
// Array Creation Functions
// ============================================================================
// Note: These return an integer status code.
// 0 = Success
// Non-zero = Error code (message in thread-local storage)

/// Helper for creation functions to reduce boilerplate
unsafe fn create_helper<T, F>(
    data: *const T,
    len: usize,
    shape: *const usize,
    ndim: usize,
    out_handle: *mut *mut NdArrayHandle,
    method: F,
) -> i32
where
    F: Fn(&[T], &[usize]) -> Result<NDArrayWrapper, String> + std::panic::UnwindSafe,
{
    if data.is_null() || shape.is_null() || out_handle.is_null() {
        return ERR_GENERIC;
    }

    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let data_slice = slice::from_raw_parts(data, len);
        let shape_slice = slice::from_raw_parts(shape, ndim);

        match method(data_slice, shape_slice) {
            Ok(wrapper) => {
                *out_handle = NdArrayHandle::from_wrapper(Box::new(wrapper));
                SUCCESS
            }
            Err(e) => {
                error::set_last_error(e);
                ERR_SHAPE
            }
        }
    })) {
        Ok(code) => code,
        Err(_) => {
            error::set_last_error("Panic during array creation");
            ERR_PANIC
        }
    }
}

/// Create an NDArray from i8 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_create_int8(
    data: *const i8,
    len: usize,
    shape: *const usize,
    ndim: usize,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    create_helper(
        data,
        len,
        shape,
        ndim,
        out_handle,
        NDArrayWrapper::from_slice_i8,
    )
}

/// Create an NDArray from i16 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_create_int16(
    data: *const i16,
    len: usize,
    shape: *const usize,
    ndim: usize,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    create_helper(
        data,
        len,
        shape,
        ndim,
        out_handle,
        NDArrayWrapper::from_slice_i16,
    )
}

/// Create an NDArray from i32 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_create_int32(
    data: *const i32,
    len: usize,
    shape: *const usize,
    ndim: usize,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    create_helper(
        data,
        len,
        shape,
        ndim,
        out_handle,
        NDArrayWrapper::from_slice_i32,
    )
}

/// Create an NDArray from i64 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_create_int64(
    data: *const i64,
    len: usize,
    shape: *const usize,
    ndim: usize,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    create_helper(
        data,
        len,
        shape,
        ndim,
        out_handle,
        NDArrayWrapper::from_slice_i64,
    )
}

/// Create an NDArray from u8 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_create_uint8(
    data: *const u8,
    len: usize,
    shape: *const usize,
    ndim: usize,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    create_helper(
        data,
        len,
        shape,
        ndim,
        out_handle,
        NDArrayWrapper::from_slice_u8,
    )
}

/// Create an NDArray from u16 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_create_uint16(
    data: *const u16,
    len: usize,
    shape: *const usize,
    ndim: usize,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    create_helper(
        data,
        len,
        shape,
        ndim,
        out_handle,
        NDArrayWrapper::from_slice_u16,
    )
}

/// Create an NDArray from u32 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_create_uint32(
    data: *const u32,
    len: usize,
    shape: *const usize,
    ndim: usize,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    create_helper(
        data,
        len,
        shape,
        ndim,
        out_handle,
        NDArrayWrapper::from_slice_u32,
    )
}

/// Create an NDArray from u64 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_create_uint64(
    data: *const u64,
    len: usize,
    shape: *const usize,
    ndim: usize,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    create_helper(
        data,
        len,
        shape,
        ndim,
        out_handle,
        NDArrayWrapper::from_slice_u64,
    )
}

/// Create an NDArray from f32 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_create_float32(
    data: *const f32,
    len: usize,
    shape: *const usize,
    ndim: usize,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    create_helper(
        data,
        len,
        shape,
        ndim,
        out_handle,
        NDArrayWrapper::from_slice_f32,
    )
}

/// Create an NDArray from f64 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_create_float64(
    data: *const f64,
    len: usize,
    shape: *const usize,
    ndim: usize,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    create_helper(
        data,
        len,
        shape,
        ndim,
        out_handle,
        NDArrayWrapper::from_slice_f64,
    )
}

/// Create an NDArray from bool data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_create_bool(
    data: *const u8,
    len: usize,
    shape: *const usize,
    ndim: usize,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    create_helper(
        data,
        len,
        shape,
        ndim,
        out_handle,
        NDArrayWrapper::from_slice_bool,
    )
}

// ============================================================================
// Array Copy (View to Owned)
// ============================================================================

/// Create a deep copy of an array view.
#[no_mangle]
pub unsafe extern "C" fn ndarray_copy(
    handle: *const NdArrayHandle,
    offset: usize,
    shape_ptr: *const usize,
    strides_ptr: *const usize,
    ndim: usize,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || shape_ptr.is_null() || strides_ptr.is_null() || out_handle.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape = slice::from_raw_parts(shape_ptr, ndim);
        let strides = slice::from_raw_parts(strides_ptr, ndim);

        use crate::dtype::DType::*;
        let result = match wrapper.dtype {
            Int8 => wrapper.copy_view_i8(offset, shape, strides),
            Int16 => wrapper.copy_view_i16(offset, shape, strides),
            Int32 => wrapper.copy_view_i32(offset, shape, strides),
            Int64 => wrapper.copy_view_i64(offset, shape, strides),
            Uint8 => wrapper.copy_view_u8(offset, shape, strides),
            Uint16 => wrapper.copy_view_u16(offset, shape, strides),
            Uint32 => wrapper.copy_view_u32(offset, shape, strides),
            Uint64 => wrapper.copy_view_u64(offset, shape, strides),
            Float32 => wrapper.copy_view_f32(offset, shape, strides),
            Float64 => wrapper.copy_view_f64(offset, shape, strides),
            Bool => wrapper.copy_view_bool(offset, shape, strides),
        };

        match result {
            Ok(new_wrapper) => {
                *out_handle = NdArrayHandle::from_wrapper(Box::new(new_wrapper));
                SUCCESS
            }
            Err(e) => {
                error::set_last_error(e);
                ERR_GENERIC
            }
        }
    })
}
