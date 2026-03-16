//! Array creation FFI functions.

use std::ffi::c_void;
use std::slice;

use crate::core::NDArrayWrapper;
use crate::core::dtype::DType;
use crate::core::error::{self, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::ffi::NdArrayHandle;

/// Create an NDArray from raw data with specified dtype.
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
