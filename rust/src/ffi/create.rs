//! Array creation FFI functions.

use std::slice;

use crate::core::NDArrayWrapper;
use crate::ffi::NdArrayHandle;

// ============================================================================
// Array Creation Functions
// ============================================================================
// Note: These must be written explicitly (not via macro) so cbindgen can
// generate the C header declarations.

/// Create an NDArray from i8 data.
///
/// # Safety
/// All pointers must be valid and point to sufficient memory.
#[no_mangle]
pub unsafe extern "C" fn ndarray_create_int8(
    data: *const i8,
    len: usize,
    shape: *const usize,
    ndim: usize,
) -> *mut NdArrayHandle {
    if data.is_null() || shape.is_null() {
        return std::ptr::null_mut();
    }

    let data_slice = slice::from_raw_parts(data, len);
    let shape_slice = slice::from_raw_parts(shape, ndim);

    match NDArrayWrapper::from_slice_i8(data_slice, shape_slice) {
        Ok(wrapper) => NdArrayHandle::from_wrapper(Box::new(wrapper)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Create an NDArray from i16 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_create_int16(
    data: *const i16,
    len: usize,
    shape: *const usize,
    ndim: usize,
) -> *mut NdArrayHandle {
    if data.is_null() || shape.is_null() {
        return std::ptr::null_mut();
    }

    let data_slice = slice::from_raw_parts(data, len);
    let shape_slice = slice::from_raw_parts(shape, ndim);

    match NDArrayWrapper::from_slice_i16(data_slice, shape_slice) {
        Ok(wrapper) => NdArrayHandle::from_wrapper(Box::new(wrapper)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Create an NDArray from i32 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_create_int32(
    data: *const i32,
    len: usize,
    shape: *const usize,
    ndim: usize,
) -> *mut NdArrayHandle {
    if data.is_null() || shape.is_null() {
        return std::ptr::null_mut();
    }

    let data_slice = slice::from_raw_parts(data, len);
    let shape_slice = slice::from_raw_parts(shape, ndim);

    match NDArrayWrapper::from_slice_i32(data_slice, shape_slice) {
        Ok(wrapper) => NdArrayHandle::from_wrapper(Box::new(wrapper)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Create an NDArray from i64 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_create_int64(
    data: *const i64,
    len: usize,
    shape: *const usize,
    ndim: usize,
) -> *mut NdArrayHandle {
    if data.is_null() || shape.is_null() {
        return std::ptr::null_mut();
    }

    let data_slice = slice::from_raw_parts(data, len);
    let shape_slice = slice::from_raw_parts(shape, ndim);

    match NDArrayWrapper::from_slice_i64(data_slice, shape_slice) {
        Ok(wrapper) => NdArrayHandle::from_wrapper(Box::new(wrapper)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Create an NDArray from u8 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_create_uint8(
    data: *const u8,
    len: usize,
    shape: *const usize,
    ndim: usize,
) -> *mut NdArrayHandle {
    if data.is_null() || shape.is_null() {
        return std::ptr::null_mut();
    }

    let data_slice = slice::from_raw_parts(data, len);
    let shape_slice = slice::from_raw_parts(shape, ndim);

    match NDArrayWrapper::from_slice_u8(data_slice, shape_slice) {
        Ok(wrapper) => NdArrayHandle::from_wrapper(Box::new(wrapper)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Create an NDArray from u16 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_create_uint16(
    data: *const u16,
    len: usize,
    shape: *const usize,
    ndim: usize,
) -> *mut NdArrayHandle {
    if data.is_null() || shape.is_null() {
        return std::ptr::null_mut();
    }

    let data_slice = slice::from_raw_parts(data, len);
    let shape_slice = slice::from_raw_parts(shape, ndim);

    match NDArrayWrapper::from_slice_u16(data_slice, shape_slice) {
        Ok(wrapper) => NdArrayHandle::from_wrapper(Box::new(wrapper)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Create an NDArray from u32 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_create_uint32(
    data: *const u32,
    len: usize,
    shape: *const usize,
    ndim: usize,
) -> *mut NdArrayHandle {
    if data.is_null() || shape.is_null() {
        return std::ptr::null_mut();
    }

    let data_slice = slice::from_raw_parts(data, len);
    let shape_slice = slice::from_raw_parts(shape, ndim);

    match NDArrayWrapper::from_slice_u32(data_slice, shape_slice) {
        Ok(wrapper) => NdArrayHandle::from_wrapper(Box::new(wrapper)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Create an NDArray from u64 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_create_uint64(
    data: *const u64,
    len: usize,
    shape: *const usize,
    ndim: usize,
) -> *mut NdArrayHandle {
    if data.is_null() || shape.is_null() {
        return std::ptr::null_mut();
    }

    let data_slice = slice::from_raw_parts(data, len);
    let shape_slice = slice::from_raw_parts(shape, ndim);

    match NDArrayWrapper::from_slice_u64(data_slice, shape_slice) {
        Ok(wrapper) => NdArrayHandle::from_wrapper(Box::new(wrapper)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Create an NDArray from f32 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_create_float32(
    data: *const f32,
    len: usize,
    shape: *const usize,
    ndim: usize,
) -> *mut NdArrayHandle {
    if data.is_null() || shape.is_null() {
        return std::ptr::null_mut();
    }

    let data_slice = slice::from_raw_parts(data, len);
    let shape_slice = slice::from_raw_parts(shape, ndim);

    match NDArrayWrapper::from_slice_f32(data_slice, shape_slice) {
        Ok(wrapper) => NdArrayHandle::from_wrapper(Box::new(wrapper)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Create an NDArray from f64 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_create_float64(
    data: *const f64,
    len: usize,
    shape: *const usize,
    ndim: usize,
) -> *mut NdArrayHandle {
    if data.is_null() || shape.is_null() {
        return std::ptr::null_mut();
    }

    let data_slice = slice::from_raw_parts(data, len);
    let shape_slice = slice::from_raw_parts(shape, ndim);

    match NDArrayWrapper::from_slice_f64(data_slice, shape_slice) {
        Ok(wrapper) => NdArrayHandle::from_wrapper(Box::new(wrapper)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Create an NDArray from bool data (passed as u8: 0 = false, non-zero = true).
#[no_mangle]
pub unsafe extern "C" fn ndarray_create_bool(
    data: *const u8,
    len: usize,
    shape: *const usize,
    ndim: usize,
) -> *mut NdArrayHandle {
    if data.is_null() || shape.is_null() {
        return std::ptr::null_mut();
    }

    let data_slice = slice::from_raw_parts(data, len);
    let shape_slice = slice::from_raw_parts(shape, ndim);

    match NDArrayWrapper::from_slice_bool(data_slice, shape_slice) {
        Ok(wrapper) => NdArrayHandle::from_wrapper(Box::new(wrapper)),
        Err(_) => std::ptr::null_mut(),
    }
}
