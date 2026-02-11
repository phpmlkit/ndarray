//! Create evenly spaced numbers over a specified interval.

use ndarray::{ArrayD, IxDyn};
use parking_lot::RwLock;
use std::sync::Arc;

use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;

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
    out_handle: *mut *mut crate::ffi::NdArrayHandle,
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
                *out_handle = crate::ffi::NdArrayHandle::from_wrapper(Box::new(wrapper));
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
                *out_handle = crate::ffi::NdArrayHandle::from_wrapper(Box::new(wrapper));
            }
            _ => return crate::error::ERR_DTYPE,
        }

        crate::error::SUCCESS
    })
}
