//! Create a 2D identity matrix.

use ndarray::{s, Array2};
use num_complex::Complex;
use parking_lot::RwLock;
use std::sync::Arc;

use crate::helpers::error::{ERR_DTYPE, ERR_GENERIC, SUCCESS};
use crate::types::dtype::DType;
use crate::types::{ArrayData, NDArrayWrapper, NdArrayHandle};

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
            DType::Complex64 => {
                let arr = Array2::<Complex<f32>>::zeros((n, m));
                let arr = fill_eye(arr, k, Complex::new(1.0, 0.0));
                NDArrayWrapper {
                    data: ArrayData::Complex64(Arc::new(RwLock::new(arr.into_dyn()))),
                    dtype: DType::Complex64,
                }
            }
            DType::Complex128 => {
                let arr = Array2::<Complex<f64>>::zeros((n, m));
                let arr = fill_eye(arr, k, Complex::new(1.0, 0.0));
                NDArrayWrapper {
                    data: ArrayData::Complex128(Arc::new(RwLock::new(arr.into_dyn()))),
                    dtype: DType::Complex128,
                }
            }
        };

        *out_handle = NdArrayHandle::from_wrapper(Box::new(wrapper));
        SUCCESS
    })
}
