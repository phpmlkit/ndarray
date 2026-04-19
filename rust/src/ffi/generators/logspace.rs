//! Create numbers spaced evenly on a log scale.

use ndarray::{Array, ArrayD};
use parking_lot::RwLock;
use std::sync::Arc;

use crate::helpers::error::{set_last_error, ERR_DTYPE, ERR_GENERIC, SUCCESS};
use crate::types::dtype::DType;
use crate::types::{ArrayData, NDArrayWrapper, NdArrayHandle};

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
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let dtype_enum = match DType::from_u8(dtype) {
            Some(d) => d,
            None => return ERR_DTYPE,
        };

        let result_wrapper = match dtype_enum {
            DType::Float32 => {
                let arr: ArrayD<f32> =
                    Array::logspace(base as f32, start as f32, stop as f32, num).into_dyn();
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(arr))),
                    dtype: DType::Float32,
                }
            }
            DType::Float64 => {
                let arr: ArrayD<f64> = Array::logspace(base, start, stop, num).into_dyn();
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(arr))),
                    dtype: DType::Float64,
                }
            }
            DType::Complex64 | DType::Complex128 => {
                set_last_error("logspace() not supported for complex dtype".to_string());
                return ERR_GENERIC;
            }
            _ => {
                set_last_error("logspace() requires float type (Float64 or Float32)".to_string());
                return ERR_DTYPE;
            }
        };
        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));

        SUCCESS
    })
}
