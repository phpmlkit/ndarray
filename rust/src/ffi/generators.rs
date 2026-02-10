//! Array generation FFI functions (zeros, ones, full, eye).

use ndarray::{s, Array2, ArrayD, IxDyn};
use parking_lot::RwLock;
use std::os::raw::c_void;
use std::slice;
use std::sync::Arc;

use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{ERR_DTYPE, ERR_GENERIC, SUCCESS};
use crate::ffi::NdArrayHandle;

// ============================================================================
// Array Generators
// ============================================================================

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

/// Create an array filled with ones.
#[no_mangle]
pub unsafe extern "C" fn ndarray_ones(
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
                let arr = ArrayD::<i8>::ones(IxDyn(shape_slice));
                NDArrayWrapper {
                    data: ArrayData::Int8(Arc::new(RwLock::new(arr))),
                    dtype: DType::Int8,
                }
            }
            DType::Int16 => {
                let arr = ArrayD::<i16>::ones(IxDyn(shape_slice));
                NDArrayWrapper {
                    data: ArrayData::Int16(Arc::new(RwLock::new(arr))),
                    dtype: DType::Int16,
                }
            }
            DType::Int32 => {
                let arr = ArrayD::<i32>::ones(IxDyn(shape_slice));
                NDArrayWrapper {
                    data: ArrayData::Int32(Arc::new(RwLock::new(arr))),
                    dtype: DType::Int32,
                }
            }
            DType::Int64 => {
                let arr = ArrayD::<i64>::ones(IxDyn(shape_slice));
                NDArrayWrapper {
                    data: ArrayData::Int64(Arc::new(RwLock::new(arr))),
                    dtype: DType::Int64,
                }
            }
            DType::Uint8 => {
                let arr = ArrayD::<u8>::ones(IxDyn(shape_slice));
                NDArrayWrapper {
                    data: ArrayData::Uint8(Arc::new(RwLock::new(arr))),
                    dtype: DType::Uint8,
                }
            }
            DType::Uint16 => {
                let arr = ArrayD::<u16>::ones(IxDyn(shape_slice));
                NDArrayWrapper {
                    data: ArrayData::Uint16(Arc::new(RwLock::new(arr))),
                    dtype: DType::Uint16,
                }
            }
            DType::Uint32 => {
                let arr = ArrayD::<u32>::ones(IxDyn(shape_slice));
                NDArrayWrapper {
                    data: ArrayData::Uint32(Arc::new(RwLock::new(arr))),
                    dtype: DType::Uint32,
                }
            }
            DType::Uint64 => {
                let arr = ArrayD::<u64>::ones(IxDyn(shape_slice));
                NDArrayWrapper {
                    data: ArrayData::Uint64(Arc::new(RwLock::new(arr))),
                    dtype: DType::Uint64,
                }
            }
            DType::Float32 => {
                let arr = ArrayD::<f32>::ones(IxDyn(shape_slice));
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(arr))),
                    dtype: DType::Float32,
                }
            }
            DType::Float64 => {
                let arr = ArrayD::<f64>::ones(IxDyn(shape_slice));
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(arr))),
                    dtype: DType::Float64,
                }
            }
            DType::Bool => {
                // bool ones = true (1)
                let arr = ArrayD::<u8>::ones(IxDyn(shape_slice));
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

/// Create a 2D identity matrix.
#[no_mangle]
pub unsafe extern "C" fn ndarray_eye(
    n: usize,
    m: usize,
    k: isize,
    dtype: u8,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if out_handle.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let dtype_enum = match DType::from_u8(dtype) {
            Some(d) => d,
            None => return ERR_DTYPE,
        };

        // Helper to fill diagonal using slice logic
        fn fill_eye<T>(mut arr: Array2<T>, k: isize, one: T) -> Array2<T>
        where
            T: Clone,
        {
            let (rows, cols) = arr.dim();

            // Determine slice range based on k
            if k >= 0 {
                // Upper diagonal
                if (k as usize) < cols {
                    // Safe handling of slicing
                    let mut slice = arr.slice_mut(s![.., k as usize..]);
                    slice.diag_mut().fill(one);
                }
            } else {
                // Lower diagonal
                let k_abs = (-k) as usize;
                if k_abs < rows {
                    let mut slice = arr.slice_mut(s![k_abs.., ..]);
                    slice.diag_mut().fill(one);
                }
            }
            arr
        }

        let wrapper = match dtype_enum {
            DType::Int8 => {
                let arr = Array2::<i8>::zeros((n, m));
                let arr = fill_eye(arr, k, 1);
                NDArrayWrapper {
                    data: ArrayData::Int8(Arc::new(RwLock::new(arr.into_dyn()))),
                    dtype: DType::Int8,
                }
            }
            DType::Int16 => {
                let arr = Array2::<i16>::zeros((n, m));
                let arr = fill_eye(arr, k, 1);
                NDArrayWrapper {
                    data: ArrayData::Int16(Arc::new(RwLock::new(arr.into_dyn()))),
                    dtype: DType::Int16,
                }
            }
            DType::Int32 => {
                let arr = Array2::<i32>::zeros((n, m));
                let arr = fill_eye(arr, k, 1);
                NDArrayWrapper {
                    data: ArrayData::Int32(Arc::new(RwLock::new(arr.into_dyn()))),
                    dtype: DType::Int32,
                }
            }
            DType::Int64 => {
                let arr = Array2::<i64>::zeros((n, m));
                let arr = fill_eye(arr, k, 1);
                NDArrayWrapper {
                    data: ArrayData::Int64(Arc::new(RwLock::new(arr.into_dyn()))),
                    dtype: DType::Int64,
                }
            }
            DType::Uint8 => {
                let arr = Array2::<u8>::zeros((n, m));
                let arr = fill_eye(arr, k, 1);
                NDArrayWrapper {
                    data: ArrayData::Uint8(Arc::new(RwLock::new(arr.into_dyn()))),
                    dtype: DType::Uint8,
                }
            }
            DType::Uint16 => {
                let arr = Array2::<u16>::zeros((n, m));
                let arr = fill_eye(arr, k, 1);
                NDArrayWrapper {
                    data: ArrayData::Uint16(Arc::new(RwLock::new(arr.into_dyn()))),
                    dtype: DType::Uint16,
                }
            }
            DType::Uint32 => {
                let arr = Array2::<u32>::zeros((n, m));
                let arr = fill_eye(arr, k, 1);
                NDArrayWrapper {
                    data: ArrayData::Uint32(Arc::new(RwLock::new(arr.into_dyn()))),
                    dtype: DType::Uint32,
                }
            }
            DType::Uint64 => {
                let arr = Array2::<u64>::zeros((n, m));
                let arr = fill_eye(arr, k, 1);
                NDArrayWrapper {
                    data: ArrayData::Uint64(Arc::new(RwLock::new(arr.into_dyn()))),
                    dtype: DType::Uint64,
                }
            }
            DType::Float32 => {
                let arr = Array2::<f32>::zeros((n, m));
                let arr = fill_eye(arr, k, 1.0);
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(arr.into_dyn()))),
                    dtype: DType::Float32,
                }
            }
            DType::Float64 => {
                let arr = Array2::<f64>::zeros((n, m));
                let arr = fill_eye(arr, k, 1.0);
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(arr.into_dyn()))),
                    dtype: DType::Float64,
                }
            }
            DType::Bool => {
                let arr = Array2::<u8>::zeros((n, m));
                let arr = fill_eye(arr, k, 1);
                NDArrayWrapper {
                    data: ArrayData::Bool(Arc::new(RwLock::new(arr.into_dyn()))),
                    dtype: DType::Bool,
                }
            }
        };

        *out_handle = NdArrayHandle::from_wrapper(Box::new(wrapper));
        SUCCESS
    })
}

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
    out_handle: *mut *mut NdArrayHandle,
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
            *out_handle = NdArrayHandle::from_wrapper(Box::new(wrapper));
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

        *out_handle = NdArrayHandle::from_wrapper(Box::new(wrapper));
        crate::error::SUCCESS
    })
}

/// Create evenly spaced numbers over a specified interval.
///
/// Only supports Float32 and Float64 dtypes.
/// The endpoint parameter controls whether stop is included.
#[no_mangle]
pub unsafe extern "C" fn ndarray_linspace(
    start: f64,
    stop: f64,
    num: usize,
    endpoint: bool,
    dtype: u8,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if out_handle.is_null() || num == 0 {
        return crate::error::ERR_GENERIC;
    }

    crate::ffi_guard!({
        let dtype_enum = match DType::from_u8(dtype) {
            Some(d) => d,
            None => return crate::error::ERR_DTYPE,
        };

        match dtype_enum {
            DType::Float32 => {
                let data: Vec<f32> = if num == 1 {
                    vec![start as f32]
                } else {
                    let step = (stop - start)
                        / (if endpoint {
                            (num - 1) as f64
                        } else {
                            num as f64
                        });
                    (0..num).map(|i| (start + step * i as f64) as f32).collect()
                };
                let arr = ArrayD::<f32>::from_shape_vec(IxDyn(&[num]), data)
                    .expect("Shape mismatch should not happen");
                let wrapper = NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(arr))),
                    dtype: DType::Float32,
                };
                *out_handle = NdArrayHandle::from_wrapper(Box::new(wrapper));
            }
            DType::Float64 => {
                let data: Vec<f64> = if num == 1 {
                    vec![start]
                } else {
                    let step = (stop - start)
                        / (if endpoint {
                            (num - 1) as f64
                        } else {
                            num as f64
                        });
                    (0..num).map(|i| start + step * i as f64).collect()
                };
                let arr = ArrayD::<f64>::from_shape_vec(IxDyn(&[num]), data)
                    .expect("Shape mismatch should not happen");
                let wrapper = NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(arr))),
                    dtype: DType::Float64,
                };
                *out_handle = NdArrayHandle::from_wrapper(Box::new(wrapper));
            }
            _ => return crate::error::ERR_DTYPE,
        }

        crate::error::SUCCESS
    })
}

/// Create numbers spaced evenly on a log scale.
///
/// Only supports Float32 and Float64 dtypes.
/// Returns `base.powf(start)` to `base.powf(stop)` with `num` points.
#[no_mangle]
pub unsafe extern "C" fn ndarray_logspace(
    start: f64,
    stop: f64,
    num: usize,
    base: f64,
    dtype: u8,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if out_handle.is_null() || num == 0 {
        return crate::error::ERR_GENERIC;
    }

    crate::ffi_guard!({
        let dtype_enum = match DType::from_u8(dtype) {
            Some(d) => d,
            None => return crate::error::ERR_DTYPE,
        };

        match dtype_enum {
            DType::Float32 => {
                let data: Vec<f32> = if num == 1 {
                    vec![base.powf(start) as f32]
                } else {
                    let step = (stop - start) / ((num - 1) as f64);
                    (0..num)
                        .map(|i| base.powf(start + step * i as f64) as f32)
                        .collect()
                };
                let arr = ArrayD::<f32>::from_shape_vec(IxDyn(&[num]), data)
                    .expect("Shape mismatch should not happen");
                let wrapper = NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(arr))),
                    dtype: DType::Float32,
                };
                *out_handle = NdArrayHandle::from_wrapper(Box::new(wrapper));
            }
            DType::Float64 => {
                let data: Vec<f64> = if num == 1 {
                    vec![base.powf(start)]
                } else {
                    let step = (stop - start) / ((num - 1) as f64);
                    (0..num)
                        .map(|i| base.powf(start + step * i as f64))
                        .collect()
                };
                let arr = ArrayD::<f64>::from_shape_vec(IxDyn(&[num]), data)
                    .expect("Shape mismatch should not happen");
                let wrapper = NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(arr))),
                    dtype: DType::Float64,
                };
                *out_handle = NdArrayHandle::from_wrapper(Box::new(wrapper));
            }
            _ => return crate::error::ERR_DTYPE,
        }

        crate::error::SUCCESS
    })
}

/// Create numbers spaced geometrically from start to stop.
///
/// Only supports Float32 and Float64 dtypes.
/// Returns error if start and stop have different signs or if either is zero.
#[no_mangle]
pub unsafe extern "C" fn ndarray_geomspace(
    start: f64,
    stop: f64,
    num: usize,
    dtype: u8,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if out_handle.is_null() || num == 0 {
        return crate::error::ERR_GENERIC;
    }

    crate::ffi_guard!({
        let dtype_enum = match DType::from_u8(dtype) {
            Some(d) => d,
            None => return crate::error::ERR_DTYPE,
        };

        // Validate: start and stop must have same sign and neither can be zero
        if start == 0.0 || stop == 0.0 || (start > 0.0) != (stop > 0.0) {
            return crate::error::ERR_GENERIC;
        }

        match dtype_enum {
            DType::Float32 => {
                let data: Vec<f32> = if num == 1 {
                    vec![start as f32]
                } else {
                    let ratio = (stop / start).powf(1.0 / ((num - 1) as f64));
                    (0..num)
                        .map(|i| (start * ratio.powi(i as i32)) as f32)
                        .collect()
                };
                let arr = ArrayD::<f32>::from_shape_vec(IxDyn(&[num]), data)
                    .expect("Shape mismatch should not happen");
                let wrapper = NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(arr))),
                    dtype: DType::Float32,
                };
                *out_handle = NdArrayHandle::from_wrapper(Box::new(wrapper));
            }
            DType::Float64 => {
                let data: Vec<f64> = if num == 1 {
                    vec![start]
                } else {
                    let ratio = (stop / start).powf(1.0 / ((num - 1) as f64));
                    (0..num).map(|i| start * ratio.powi(i as i32)).collect()
                };
                let arr = ArrayD::<f64>::from_shape_vec(IxDyn(&[num]), data)
                    .expect("Shape mismatch should not happen");
                let wrapper = NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(arr))),
                    dtype: DType::Float64,
                };
                *out_handle = NdArrayHandle::from_wrapper(Box::new(wrapper));
            }
            _ => return crate::error::ERR_DTYPE,
        }

        crate::error::SUCCESS
    })
}
