//! Create arrays with random uniform values in [low, high).

use ndarray::{ArrayD, IxDyn};
use parking_lot::RwLock;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::sync::Arc;

use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{ERR_DTYPE, ERR_GENERIC, SUCCESS};
use crate::ffi::NdArrayHandle;
use std::slice;

fn shape_len(shape: &[usize]) -> Result<usize, String> {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| "Shape product overflow".to_string())
}

fn build_rng(has_seed: bool, seed: u64) -> StdRng {
    if has_seed {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::seed_from_u64(rand::random::<u64>())
    }
}

/// Create an array of random values sampled uniformly from [low, high).
///
/// Supports Float32 and Float64 only.
#[no_mangle]
pub unsafe extern "C" fn ndarray_uniform(
    low: f64,
    high: f64,
    shape: *const usize,
    ndim: usize,
    dtype: u8,
    has_seed: bool,
    seed: u64,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if shape.is_null() || out_handle.is_null() {
        return ERR_GENERIC;
    }
    if high <= low {
        crate::error::set_last_error(format!(
            "uniform requires high > low, got [{}, {})",
            low, high
        ));
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let shape_slice = slice::from_raw_parts(shape, ndim);
        let len = match shape_len(shape_slice) {
            Ok(v) => v,
            Err(e) => {
                crate::error::set_last_error(e);
                return ERR_GENERIC;
            }
        };

        let dtype_enum = match DType::from_u8(dtype) {
            Some(d) => d,
            None => return ERR_DTYPE,
        };

        let mut rng = build_rng(has_seed, seed);

        let wrapper = match dtype_enum {
            DType::Float32 => {
                let lo = low as f32;
                let hi = high as f32;
                let data: Vec<f32> = (0..len).map(|_| rng.random_range(lo..hi)).collect();
                let arr = ArrayD::<f32>::from_shape_vec(IxDyn(shape_slice), data)
                    .expect("Shape mismatch should not happen");
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(arr))),
                    dtype: DType::Float32,
                }
            }
            DType::Float64 => {
                let data: Vec<f64> = (0..len).map(|_| rng.random_range(low..high)).collect();
                let arr = ArrayD::<f64>::from_shape_vec(IxDyn(shape_slice), data)
                    .expect("Shape mismatch should not happen");
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(arr))),
                    dtype: DType::Float64,
                }
            }
            _ => return ERR_DTYPE,
        };

        *out_handle = NdArrayHandle::from_wrapper(Box::new(wrapper));
        SUCCESS
    })
}
