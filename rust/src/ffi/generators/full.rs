//! Create an array filled with a specific value.

use ndarray::{ArrayD, IxDyn};
use parking_lot::RwLock;
use std::os::raw::c_void;
use std::sync::Arc;

use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{ERR_DTYPE, ERR_GENERIC, SUCCESS};
use crate::ffi::NdArrayHandle;
use std::slice;

/// Create an array filled with a specific value.
#[no_mangle]
pub unsafe extern "C" fn ndarray_full(
    shape: *const usize,
    ndim: usize,
    value: *const c_void,
    dtype: u8,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if shape.is_null() || value.is_null() || out_handle.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let shape_slice = slice::from_raw_parts(shape, ndim);
        let dtype_enum = match DType::from_u8(dtype) {
            Some(d) => d,
            None => return ERR_DTYPE,
        };

        let wrapper = match dtype_enum {
            DType::Int8 => {
                let val = *(value as *const i8);
                let arr = ArrayD::<i8>::from_elem(IxDyn(shape_slice), val);
                NDArrayWrapper {
                    data: ArrayData::Int8(Arc::new(RwLock::new(arr))),
                    dtype: DType::Int8,
                }
            }
            DType::Int16 => {
                let val = *(value as *const i16);
                let arr = ArrayD::<i16>::from_elem(IxDyn(shape_slice), val);
                NDArrayWrapper {
                    data: ArrayData::Int16(Arc::new(RwLock::new(arr))),
                    dtype: DType::Int16,
                }
            }
            DType::Int32 => {
                let val = *(value as *const i32);
                let arr = ArrayD::<i32>::from_elem(IxDyn(shape_slice), val);
                NDArrayWrapper {
                    data: ArrayData::Int32(Arc::new(RwLock::new(arr))),
                    dtype: DType::Int32,
                }
            }
            DType::Int64 => {
                let val = *(value as *const i64);
                let arr = ArrayD::<i64>::from_elem(IxDyn(shape_slice), val);
                NDArrayWrapper {
                    data: ArrayData::Int64(Arc::new(RwLock::new(arr))),
                    dtype: DType::Int64,
                }
            }
            DType::Uint8 => {
                let val = *(value as *const u8);
                let arr = ArrayD::<u8>::from_elem(IxDyn(shape_slice), val);
                NDArrayWrapper {
                    data: ArrayData::Uint8(Arc::new(RwLock::new(arr))),
                    dtype: DType::Uint8,
                }
            }
            DType::Uint16 => {
                let val = *(value as *const u16);
                let arr = ArrayD::<u16>::from_elem(IxDyn(shape_slice), val);
                NDArrayWrapper {
                    data: ArrayData::Uint16(Arc::new(RwLock::new(arr))),
                    dtype: DType::Uint16,
                }
            }
            DType::Uint32 => {
                let val = *(value as *const u32);
                let arr = ArrayD::<u32>::from_elem(IxDyn(shape_slice), val);
                NDArrayWrapper {
                    data: ArrayData::Uint32(Arc::new(RwLock::new(arr))),
                    dtype: DType::Uint32,
                }
            }
            DType::Uint64 => {
                let val = *(value as *const u64);
                let arr = ArrayD::<u64>::from_elem(IxDyn(shape_slice), val);
                NDArrayWrapper {
                    data: ArrayData::Uint64(Arc::new(RwLock::new(arr))),
                    dtype: DType::Uint64,
                }
            }
            DType::Float32 => {
                let val = *(value as *const f32);
                let arr = ArrayD::<f32>::from_elem(IxDyn(shape_slice), val);
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(arr))),
                    dtype: DType::Float32,
                }
            }
            DType::Float64 => {
                let val = *(value as *const f64);
                let arr = ArrayD::<f64>::from_elem(IxDyn(shape_slice), val);
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(arr))),
                    dtype: DType::Float64,
                }
            }
            DType::Bool => {
                let val = *(value as *const u8);
                let arr = ArrayD::<u8>::from_elem(IxDyn(shape_slice), val);
                NDArrayWrapper {
                    data: ArrayData::Bool(Arc::new(RwLock::new(arr))),
                    dtype: DType::Bool,
                }
            }
        };

        *out_handle = NdArrayHandle::from_wrapper(Box::new(wrapper));
        SUCCESS
    })
}
