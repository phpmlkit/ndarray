//! Create numbers spaced geometrically from start to stop.

use ndarray::{Array, ArrayD};
use parking_lot::RwLock;
use std::sync::Arc;

use crate::helpers::error::{set_last_error, ERR_DTYPE, ERR_GENERIC, SUCCESS};
use crate::types::dtype::DType;
use crate::types::{ArrayData, NDArrayWrapper, NdArrayHandle};

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
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let dtype_enum = match DType::from_u8(dtype) {
            Some(d) => d,
            None => return ERR_DTYPE,
        };

        let result_wrapper = match dtype_enum {
            DType::Float32 => {
                let arr: ArrayD<f32> = match Array::geomspace(start as f32, stop as f32, num) {
                    Some(arr1) => arr1.into_dyn(),
                    None => {
                        set_last_error("geomspace requires start and stop to have the same sign and be non-zero".to_string());
                        return ERR_GENERIC;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(arr))),
                    dtype: DType::Float32,
                }
            }
            DType::Float64 => {
                let arr: ArrayD<f64> = match Array::geomspace(start, stop, num) {
                    Some(arr1) => arr1.into_dyn(),
                    None => {
                        set_last_error("geomspace requires start and stop to have the same sign and be non-zero".to_string());
                        return ERR_GENERIC;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(arr))),
                    dtype: DType::Float64,
                }
            }
            _ => {
                set_last_error("geomspace() requires float type (Float64 or Float32)".to_string());
                return ERR_DTYPE;
            }
        };
        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));

        SUCCESS
    })
}
