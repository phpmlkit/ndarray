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

/// Create an identity matrix (2D).
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
