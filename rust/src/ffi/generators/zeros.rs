//! Create an array filled with zeros.

use ndarray::{ArrayD, IxDyn};
use parking_lot::RwLock;
use std::sync::Arc;

use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{ERR_DTYPE, ERR_GENERIC, SUCCESS};
use crate::ffi::NdArrayHandle;
use std::slice;

/// Create an array filled with zeros.
#[no_mangle]
pub unsafe extern "C" fn ndarray_zeros(
    shape: *const usize,
    ndim: usize,
    dtype: u8,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if shape.is_null() || out_handle.is_null() {
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
                let arr = ArrayD::<i8>::zeros(IxDyn(shape_slice));
                NDArrayWrapper {
                    data: ArrayData::Int8(Arc::new(RwLock::new(arr))),
                    dtype: DType::Int8,
                }
            }
            DType::Int16 => {
                let arr = ArrayD::<i16>::zeros(IxDyn(shape_slice));
                NDArrayWrapper {
                    data: ArrayData::Int16(Arc::new(RwLock::new(arr))),
                    dtype: DType::Int16,
                }
            }
            DType::Int32 => {
                let arr = ArrayD::<i32>::zeros(IxDyn(shape_slice));
                NDArrayWrapper {
                    data: ArrayData::Int32(Arc::new(RwLock::new(arr))),
                    dtype: DType::Int32,
                }
            }
            DType::Int64 => {
                let arr = ArrayD::<i64>::zeros(IxDyn(shape_slice));
                NDArrayWrapper {
                    data: ArrayData::Int64(Arc::new(RwLock::new(arr))),
                    dtype: DType::Int64,
                }
            }
            DType::Uint8 => {
                let arr = ArrayD::<u8>::zeros(IxDyn(shape_slice));
                NDArrayWrapper {
                    data: ArrayData::Uint8(Arc::new(RwLock::new(arr))),
                    dtype: DType::Uint8,
                }
            }
            DType::Uint16 => {
                let arr = ArrayD::<u16>::zeros(IxDyn(shape_slice));
                NDArrayWrapper {
                    data: ArrayData::Uint16(Arc::new(RwLock::new(arr))),
                    dtype: DType::Uint16,
                }
            }
            DType::Uint32 => {
                let arr = ArrayD::<u32>::zeros(IxDyn(shape_slice));
                NDArrayWrapper {
                    data: ArrayData::Uint32(Arc::new(RwLock::new(arr))),
                    dtype: DType::Uint32,
                }
            }
            DType::Uint64 => {
                let arr = ArrayD::<u64>::zeros(IxDyn(shape_slice));
                NDArrayWrapper {
                    data: ArrayData::Uint64(Arc::new(RwLock::new(arr))),
                    dtype: DType::Uint64,
                }
            }
            DType::Float32 => {
                let arr = ArrayD::<f32>::zeros(IxDyn(shape_slice));
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(arr))),
                    dtype: DType::Float32,
                }
            }
            DType::Float64 => {
                let arr = ArrayD::<f64>::zeros(IxDyn(shape_slice));
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(arr))),
                    dtype: DType::Float64,
                }
            }
            DType::Bool => {
                // bool zeros = false
                let arr = ArrayD::<u8>::zeros(IxDyn(shape_slice));
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
