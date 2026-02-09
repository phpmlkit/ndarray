//! Array data access FFI functions.

use crate::ffi::NdArrayHandle;
use std::slice;

// ============================================================================
// Array Data Access Functions
// ============================================================================
// Note: These accessors return 0 if the internal array type does not match
// the requested type. This ensures type safety at the FFI boundary.

/// Get int8 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_data_int8(
    handle: *const NdArrayHandle,
    out_data: *mut i8,
    max_len: usize,
) -> usize {
    if handle.is_null() || out_data.is_null() {
        return 0;
    }

    let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
    if let Some(data) = wrapper.to_int8_vec() {
        let len = data.len().min(max_len);
        let out_slice = slice::from_raw_parts_mut(out_data, len);
        out_slice.copy_from_slice(&data[..len]);
        data.len()
    } else {
        0
    }
}

/// Get int16 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_data_int16(
    handle: *const NdArrayHandle,
    out_data: *mut i16,
    max_len: usize,
) -> usize {
    if handle.is_null() || out_data.is_null() {
        return 0;
    }

    let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
    if let Some(data) = wrapper.to_int16_vec() {
        let len = data.len().min(max_len);
        let out_slice = slice::from_raw_parts_mut(out_data, len);
        out_slice.copy_from_slice(&data[..len]);
        data.len()
    } else {
        0
    }
}

/// Get int32 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_data_int32(
    handle: *const NdArrayHandle,
    out_data: *mut i32,
    max_len: usize,
) -> usize {
    if handle.is_null() || out_data.is_null() {
        return 0;
    }

    let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
    if let Some(data) = wrapper.to_int32_vec() {
        let len = data.len().min(max_len);
        let out_slice = slice::from_raw_parts_mut(out_data, len);
        out_slice.copy_from_slice(&data[..len]);
        data.len()
    } else {
        0
    }
}

/// Get int64 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_data_int64(
    handle: *const NdArrayHandle,
    out_data: *mut i64,
    max_len: usize,
) -> usize {
    if handle.is_null() || out_data.is_null() {
        return 0;
    }

    let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
    if let Some(data) = wrapper.to_int64_vec() {
        let len = data.len().min(max_len);
        let out_slice = slice::from_raw_parts_mut(out_data, len);
        out_slice.copy_from_slice(&data[..len]);
        data.len()
    } else {
        0
    }
}

/// Get uint8 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_data_uint8(
    handle: *const NdArrayHandle,
    out_data: *mut u8,
    max_len: usize,
) -> usize {
    if handle.is_null() || out_data.is_null() {
        return 0;
    }

    let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
    if let Some(data) = wrapper.to_uint8_vec() {
        let len = data.len().min(max_len);
        let out_slice = slice::from_raw_parts_mut(out_data, len);
        out_slice.copy_from_slice(&data[..len]);
        data.len()
    } else {
        0
    }
}

/// Get uint16 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_data_uint16(
    handle: *const NdArrayHandle,
    out_data: *mut u16,
    max_len: usize,
) -> usize {
    if handle.is_null() || out_data.is_null() {
        return 0;
    }

    let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
    if let Some(data) = wrapper.to_uint16_vec() {
        let len = data.len().min(max_len);
        let out_slice = slice::from_raw_parts_mut(out_data, len);
        out_slice.copy_from_slice(&data[..len]);
        data.len()
    } else {
        0
    }
}

/// Get uint32 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_data_uint32(
    handle: *const NdArrayHandle,
    out_data: *mut u32,
    max_len: usize,
) -> usize {
    if handle.is_null() || out_data.is_null() {
        return 0;
    }

    let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
    if let Some(data) = wrapper.to_uint32_vec() {
        let len = data.len().min(max_len);
        let out_slice = slice::from_raw_parts_mut(out_data, len);
        out_slice.copy_from_slice(&data[..len]);
        data.len()
    } else {
        0
    }
}

/// Get uint64 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_data_uint64(
    handle: *const NdArrayHandle,
    out_data: *mut u64,
    max_len: usize,
) -> usize {
    if handle.is_null() || out_data.is_null() {
        return 0;
    }

    let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
    if let Some(data) = wrapper.to_uint64_vec() {
        let len = data.len().min(max_len);
        let out_slice = slice::from_raw_parts_mut(out_data, len);
        out_slice.copy_from_slice(&data[..len]);
        data.len()
    } else {
        0
    }
}

/// Get float32 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_data_float32(
    handle: *const NdArrayHandle,
    out_data: *mut f32,
    max_len: usize,
) -> usize {
    if handle.is_null() || out_data.is_null() {
        return 0;
    }

    let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
    if let Some(data) = wrapper.to_float32_vec() {
        let len = data.len().min(max_len);
        let out_slice = slice::from_raw_parts_mut(out_data, len);
        out_slice.copy_from_slice(&data[..len]);
        data.len()
    } else {
        0
    }
}

/// Get float64 data.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_data_float64(
    handle: *const NdArrayHandle,
    out_data: *mut f64,
    max_len: usize,
) -> usize {
    if handle.is_null() || out_data.is_null() {
        return 0;
    }

    let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
    if let Some(data) = wrapper.to_float64_vec() {
        let len = data.len().min(max_len);
        let out_slice = slice::from_raw_parts_mut(out_data, len);
        out_slice.copy_from_slice(&data[..len]);
        data.len()
    } else {
        0
    }
}

/// Get bool data (as u8).
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_data_bool(
    handle: *const NdArrayHandle,
    out_data: *mut u8,
    max_len: usize,
) -> usize {
    if handle.is_null() || out_data.is_null() {
        return 0;
    }

    let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
    if let Some(data) = wrapper.to_bool_vec() {
        let len = data.len().min(max_len);
        let out_slice = slice::from_raw_parts_mut(out_data, len);
        out_slice.copy_from_slice(&data[..len]);
        data.len()
    } else {
        0
    }
}
