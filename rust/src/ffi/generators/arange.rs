//! Create evenly spaced values within a given interval.

use ndarray::{ArrayD, IxDyn};
use parking_lot::RwLock;
use std::sync::Arc;

use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;

/// Create evenly spaced values within a given interval.
///
/// For integer types, uses native arithmetic. For float types, manual calculation.
/// Returns error if step is zero.
#[no_mangle]
pub unsafe extern "C" fn ndarray_arange(
    start: f64,
    stop: f64,
    step: f64,
    dtype: u8,
    out_handle: *mut *mut crate::ffi::NdArrayHandle,
) -> i32 {
    if out_handle.is_null() || step == 0.0 {
        return crate::error::ERR_GENERIC;
    }

    crate::ffi_guard!({
        let dtype_enum = match DType::from_u8(dtype) {
            Some(d) => d,
            None => return crate::error::ERR_DTYPE,
        };

        // Calculate number of elements
        let n = if step > 0.0 {
            if start >= stop {
                0
            } else {
                ((stop - start) / step).ceil() as usize
            }
        } else {
            if start <= stop {
                0
            } else {
                ((start - stop) / -step).ceil() as usize
            }
        };

        if n == 0 {
            // Empty array
            let wrapper = match dtype_enum {
                DType::Int8 => NDArrayWrapper {
                    data: ArrayData::Int8(Arc::new(RwLock::new(ArrayD::<i8>::zeros(IxDyn(&[0]))))),
                    dtype: DType::Int8,
                },
                DType::Int16 => NDArrayWrapper {
                    data: ArrayData::Int16(Arc::new(RwLock::new(ArrayD::<i16>::zeros(IxDyn(&[
                        0,
                    ]))))),
                    dtype: DType::Int16,
                },
                DType::Int32 => NDArrayWrapper {
                    data: ArrayData::Int32(Arc::new(RwLock::new(ArrayD::<i32>::zeros(IxDyn(&[
                        0,
                    ]))))),
                    dtype: DType::Int32,
                },
                DType::Int64 => NDArrayWrapper {
                    data: ArrayData::Int64(Arc::new(RwLock::new(ArrayD::<i64>::zeros(IxDyn(&[
                        0,
                    ]))))),
                    dtype: DType::Int64,
                },
                DType::Uint8 => NDArrayWrapper {
                    data: ArrayData::Uint8(Arc::new(RwLock::new(ArrayD::<u8>::zeros(IxDyn(&[0]))))),
                    dtype: DType::Uint8,
                },
                DType::Uint16 => NDArrayWrapper {
                    data: ArrayData::Uint16(Arc::new(RwLock::new(ArrayD::<u16>::zeros(IxDyn(&[
                        0,
                    ]))))),
                    dtype: DType::Uint16,
                },
                DType::Uint32 => NDArrayWrapper {
                    data: ArrayData::Uint32(Arc::new(RwLock::new(ArrayD::<u32>::zeros(IxDyn(&[
                        0,
                    ]))))),
                    dtype: DType::Uint32,
                },
                DType::Uint64 => NDArrayWrapper {
                    data: ArrayData::Uint64(Arc::new(RwLock::new(ArrayD::<u64>::zeros(IxDyn(&[
                        0,
                    ]))))),
                    dtype: DType::Uint64,
                },
                DType::Float32 => NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(ArrayD::<f32>::zeros(IxDyn(
                        &[0],
                    ))))),
                    dtype: DType::Float32,
                },
                DType::Float64 => NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(ArrayD::<f64>::zeros(IxDyn(
                        &[0],
                    ))))),
                    dtype: DType::Float64,
                },
                DType::Bool => NDArrayWrapper {
                    data: ArrayData::Bool(Arc::new(RwLock::new(ArrayD::<u8>::zeros(IxDyn(&[0]))))),
                    dtype: DType::Bool,
                },
            };
            *out_handle = crate::ffi::NdArrayHandle::from_wrapper(Box::new(wrapper));
            return crate::error::SUCCESS;
        }

        let wrapper = match dtype_enum {
            DType::Int8 => {
                let s = step as i8;
                if s == 0 {
                    return crate::error::ERR_GENERIC;
                }
                let data: Vec<i8> = (0..n).map(|i| (start as i8) + (i as i8) * s).collect();
                let arr = ArrayD::<i8>::from_shape_vec(IxDyn(&[n]), data)
                    .expect("Shape mismatch should not happen");
                NDArrayWrapper {
                    data: ArrayData::Int8(Arc::new(RwLock::new(arr))),
                    dtype: DType::Int8,
                }
            }
            DType::Int16 => {
                let s = step as i16;
                if s == 0 {
                    return crate::error::ERR_GENERIC;
                }
                let data: Vec<i16> = (0..n).map(|i| (start as i16) + (i as i16) * s).collect();
                let arr = ArrayD::<i16>::from_shape_vec(IxDyn(&[n]), data)
                    .expect("Shape mismatch should not happen");
                NDArrayWrapper {
                    data: ArrayData::Int16(Arc::new(RwLock::new(arr))),
                    dtype: DType::Int16,
                }
            }
            DType::Int32 => {
                let s = step as i32;
                if s == 0 {
                    return crate::error::ERR_GENERIC;
                }
                let data: Vec<i32> = (0..n).map(|i| (start as i32) + (i as i32) * s).collect();
                let arr = ArrayD::<i32>::from_shape_vec(IxDyn(&[n]), data)
                    .expect("Shape mismatch should not happen");
                NDArrayWrapper {
                    data: ArrayData::Int32(Arc::new(RwLock::new(arr))),
                    dtype: DType::Int32,
                }
            }
            DType::Int64 => {
                let s = step as i64;
                if s == 0 {
                    return crate::error::ERR_GENERIC;
                }
                let data: Vec<i64> = (0..n).map(|i| (start as i64) + (i as i64) * s).collect();
                let arr = ArrayD::<i64>::from_shape_vec(IxDyn(&[n]), data)
                    .expect("Shape mismatch should not happen");
                NDArrayWrapper {
                    data: ArrayData::Int64(Arc::new(RwLock::new(arr))),
                    dtype: DType::Int64,
                }
            }
            DType::Uint8 => {
                let s = step as u8;
                if s == 0 {
                    return crate::error::ERR_GENERIC;
                }
                let data: Vec<u8> = (0..n).map(|i| (start as u8) + (i as u8) * s).collect();
                let arr = ArrayD::<u8>::from_shape_vec(IxDyn(&[n]), data)
                    .expect("Shape mismatch should not happen");
                NDArrayWrapper {
                    data: ArrayData::Uint8(Arc::new(RwLock::new(arr))),
                    dtype: DType::Uint8,
                }
            }
            DType::Uint16 => {
                let s = step as u16;
                if s == 0 {
                    return crate::error::ERR_GENERIC;
                }
                let data: Vec<u16> = (0..n).map(|i| (start as u16) + (i as u16) * s).collect();
                let arr = ArrayD::<u16>::from_shape_vec(IxDyn(&[n]), data)
                    .expect("Shape mismatch should not happen");
                NDArrayWrapper {
                    data: ArrayData::Uint16(Arc::new(RwLock::new(arr))),
                    dtype: DType::Uint16,
                }
            }
            DType::Uint32 => {
                let s = step as u32;
                if s == 0 {
                    return crate::error::ERR_GENERIC;
                }
                let data: Vec<u32> = (0..n).map(|i| (start as u32) + (i as u32) * s).collect();
                let arr = ArrayD::<u32>::from_shape_vec(IxDyn(&[n]), data)
                    .expect("Shape mismatch should not happen");
                NDArrayWrapper {
                    data: ArrayData::Uint32(Arc::new(RwLock::new(arr))),
                    dtype: DType::Uint32,
                }
            }
            DType::Uint64 => {
                let s = step as u64;
                if s == 0 {
                    return crate::error::ERR_GENERIC;
                }
                let data: Vec<u64> = (0..n).map(|i| (start as u64) + (i as u64) * s).collect();
                let arr = ArrayD::<u64>::from_shape_vec(IxDyn(&[n]), data)
                    .expect("Shape mismatch should not happen");
                NDArrayWrapper {
                    data: ArrayData::Uint64(Arc::new(RwLock::new(arr))),
                    dtype: DType::Uint64,
                }
            }
            DType::Float32 => {
                let s = step as f32;
                let data: Vec<f32> = (0..n).map(|i| (start as f32) + (i as f32) * s).collect();
                let arr = ArrayD::<f32>::from_shape_vec(IxDyn(&[n]), data)
                    .expect("Shape mismatch should not happen");
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(arr))),
                    dtype: DType::Float32,
                }
            }
            DType::Float64 => {
                let s = step;
                let data: Vec<f64> = (0..n).map(|i| start + (i as f64) * s).collect();
                let arr = ArrayD::<f64>::from_shape_vec(IxDyn(&[n]), data)
                    .expect("Shape mismatch should not happen");
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(arr))),
                    dtype: DType::Float64,
                }
            }
            DType::Bool => {
                return crate::error::ERR_DTYPE;
            }
        };

        *out_handle = crate::ffi::NdArrayHandle::from_wrapper(Box::new(wrapper));
        crate::error::SUCCESS
    })
}
