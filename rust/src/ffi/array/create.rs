//! Array creation FFI functions.

use std::ffi::c_void;
use std::slice;

use crate::core::NDArrayWrapper;
use crate::dtype::DType;
use crate::error::{self, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::ffi::{NdArrayHandle, ViewMetadata};

/// Create an NDArray from raw data with specified dtype.
///
/// # Arguments
/// * `data` - Pointer to raw data (type depends on dtype)
/// * `len` - Number of elements
/// * `shape` - Pointer to shape array
/// * `ndim` - Number of dimensions
/// * `dtype` - Data type enum value
/// * `out_handle` - Output handle pointer
#[no_mangle]
pub unsafe extern "C" fn ndarray_create(
    data: *const c_void,
    len: usize,
    shape: *const usize,
    ndim: usize,
    dtype: i32,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if data.is_null() || shape.is_null() || out_handle.is_null() {
        return ERR_GENERIC;
    }

    let dtype = match DType::from_u8(dtype as u8) {
        Some(d) => d,
        None => return ERR_GENERIC,
    };

    crate::ffi_guard!({
        let shape_slice = slice::from_raw_parts(shape, ndim);

        let result = match dtype {
            DType::Int8 => {
                let data_slice = slice::from_raw_parts(data as *const i8, len);
                NDArrayWrapper::from_slice_i8(data_slice, shape_slice)
            }
            DType::Int16 => {
                let data_slice = slice::from_raw_parts(data as *const i16, len);
                NDArrayWrapper::from_slice_i16(data_slice, shape_slice)
            }
            DType::Int32 => {
                let data_slice = slice::from_raw_parts(data as *const i32, len);
                NDArrayWrapper::from_slice_i32(data_slice, shape_slice)
            }
            DType::Int64 => {
                let data_slice = slice::from_raw_parts(data as *const i64, len);
                NDArrayWrapper::from_slice_i64(data_slice, shape_slice)
            }
            DType::Uint8 => {
                let data_slice = slice::from_raw_parts(data as *const u8, len);
                NDArrayWrapper::from_slice_u8(data_slice, shape_slice)
            }
            DType::Uint16 => {
                let data_slice = slice::from_raw_parts(data as *const u16, len);
                NDArrayWrapper::from_slice_u16(data_slice, shape_slice)
            }
            DType::Uint32 => {
                let data_slice = slice::from_raw_parts(data as *const u32, len);
                NDArrayWrapper::from_slice_u32(data_slice, shape_slice)
            }
            DType::Uint64 => {
                let data_slice = slice::from_raw_parts(data as *const u64, len);
                NDArrayWrapper::from_slice_u64(data_slice, shape_slice)
            }
            DType::Float32 => {
                let data_slice = slice::from_raw_parts(data as *const f32, len);
                NDArrayWrapper::from_slice_f32(data_slice, shape_slice)
            }
            DType::Float64 => {
                let data_slice = slice::from_raw_parts(data as *const f64, len);
                NDArrayWrapper::from_slice_f64(data_slice, shape_slice)
            }
            DType::Bool => {
                let data_slice = slice::from_raw_parts(data as *const u8, len);
                NDArrayWrapper::from_slice_bool(data_slice, shape_slice)
            }
        };

        match result {
            Ok(wrapper) => {
                *out_handle = NdArrayHandle::from_wrapper(Box::new(wrapper));
                SUCCESS
            }
            Err(e) => {
                error::set_last_error(e);
                ERR_SHAPE
            }
        }
    })
}

/// Create a deep copy of an array view.
#[no_mangle]
pub unsafe extern "C" fn ndarray_copy(
    handle: *const NdArrayHandle,
    meta: *const ViewMetadata,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || meta.is_null() || out_handle.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let meta = &*meta;
       
        let result = match wrapper.dtype {
            DType::Int8 => wrapper.copy_view_i8(meta),
            DType::Int16 => wrapper.copy_view_i16(meta),
            DType::Int32 => wrapper.copy_view_i32(meta),
            DType::Int64 => wrapper.copy_view_i64(meta),
            DType::Uint8 => wrapper.copy_view_u8(meta),
            DType::Uint16 => wrapper.copy_view_u16(meta),
            DType::Uint32 => wrapper.copy_view_u32(meta),
            DType::Uint64 => wrapper.copy_view_u64(meta),
            DType::Float32 => wrapper.copy_view_f32(meta),
            DType::Float64 => wrapper.copy_view_f64(meta),
            DType::Bool => wrapper.copy_view_bool(meta),
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
