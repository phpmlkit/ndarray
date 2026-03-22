//! Create evenly spaced numbers over a specified interval.

use ndarray::{Array, ArrayD};
use parking_lot::RwLock;
use std::sync::Arc;

use crate::helpers::error::{set_last_error, ERR_DTYPE, ERR_GENERIC, SUCCESS};
use crate::types::dtype::DType;
use crate::types::{ArrayData, NDArrayWrapper, NdArrayHandle};

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
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let dtype_enum = match DType::from_u8(dtype) {
            Some(d) => d,
            None => return ERR_DTYPE,
        };

        let result_wrapper = match dtype_enum {
            DType::Float32 => {
                let adjusted_stop = if endpoint {
                    stop as f32
                } else {
                    let step = ((stop - start) / (num as f64)) as f32;
                    (start as f32) + step * ((num - 1) as f32)
                };

                let arr: ArrayD<f32> = Array::linspace(start as f32, adjusted_stop, num).into_dyn();

                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(arr))),
                    dtype: DType::Float32,
                }
            }
            DType::Float64 => {
                let adjusted_stop = if endpoint {
                    stop
                } else {
                    let step = (stop - start) / (num as f64);
                    start + step * ((num - 1) as f64)
                };

                let arr: ArrayD<f64> = Array::linspace(start, adjusted_stop, num).into_dyn();

                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(arr))),
                    dtype: DType::Float64,
                }
            }
            _ => {
                set_last_error("linspace() requires float type (Float64 or Float32)".to_string());
                return ERR_DTYPE;
            }
        };
        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));

        SUCCESS
    })
}
