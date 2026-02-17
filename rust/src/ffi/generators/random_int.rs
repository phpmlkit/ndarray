//! Create arrays with random integer values in [low, high).

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

fn bounds_i8(low: i64, high: i64) -> Result<(i8, i8), String> {
    let lo = i8::try_from(low).map_err(|_| "low is out of range for Int8".to_string())?;
    let hi = i8::try_from(high).map_err(|_| "high is out of range for Int8".to_string())?;
    if hi <= lo {
        return Err("randomInt requires high > low".to_string());
    }
    Ok((lo, hi))
}

fn bounds_i16(low: i64, high: i64) -> Result<(i16, i16), String> {
    let lo = i16::try_from(low).map_err(|_| "low is out of range for Int16".to_string())?;
    let hi = i16::try_from(high).map_err(|_| "high is out of range for Int16".to_string())?;
    if hi <= lo {
        return Err("randomInt requires high > low".to_string());
    }
    Ok((lo, hi))
}

fn bounds_i32(low: i64, high: i64) -> Result<(i32, i32), String> {
    let lo = i32::try_from(low).map_err(|_| "low is out of range for Int32".to_string())?;
    let hi = i32::try_from(high).map_err(|_| "high is out of range for Int32".to_string())?;
    if hi <= lo {
        return Err("randomInt requires high > low".to_string());
    }
    Ok((lo, hi))
}

fn bounds_i64(low: i64, high: i64) -> Result<(i64, i64), String> {
    if high <= low {
        return Err("randomInt requires high > low".to_string());
    }
    Ok((low, high))
}

fn bounds_u8(low: i64, high: i64) -> Result<(u8, u8), String> {
    if low < 0 || high < 0 {
        return Err("randomInt bounds must be >= 0 for unsigned dtypes".to_string());
    }
    let lo = u8::try_from(low).map_err(|_| "low is out of range for Uint8".to_string())?;
    let hi = u8::try_from(high).map_err(|_| "high is out of range for Uint8".to_string())?;
    if hi <= lo {
        return Err("randomInt requires high > low".to_string());
    }
    Ok((lo, hi))
}

fn bounds_u16(low: i64, high: i64) -> Result<(u16, u16), String> {
    if low < 0 || high < 0 {
        return Err("randomInt bounds must be >= 0 for unsigned dtypes".to_string());
    }
    let lo = u16::try_from(low).map_err(|_| "low is out of range for Uint16".to_string())?;
    let hi = u16::try_from(high).map_err(|_| "high is out of range for Uint16".to_string())?;
    if hi <= lo {
        return Err("randomInt requires high > low".to_string());
    }
    Ok((lo, hi))
}

fn bounds_u32(low: i64, high: i64) -> Result<(u32, u32), String> {
    if low < 0 || high < 0 {
        return Err("randomInt bounds must be >= 0 for unsigned dtypes".to_string());
    }
    let lo = u32::try_from(low).map_err(|_| "low is out of range for Uint32".to_string())?;
    let hi = u32::try_from(high).map_err(|_| "high is out of range for Uint32".to_string())?;
    if hi <= lo {
        return Err("randomInt requires high > low".to_string());
    }
    Ok((lo, hi))
}

fn bounds_u64(low: i64, high: i64) -> Result<(u64, u64), String> {
    if low < 0 || high < 0 {
        return Err("randomInt bounds must be >= 0 for unsigned dtypes".to_string());
    }
    let lo = low as u64;
    let hi = high as u64;
    if hi <= lo {
        return Err("randomInt requires high > low".to_string());
    }
    Ok((lo, hi))
}

/// Create an array of random integers sampled uniformly from [low, high).
///
/// Supports signed and unsigned integer dtypes only.
#[no_mangle]
pub unsafe extern "C" fn ndarray_random_int(
    low: i64,
    high: i64,
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
            DType::Int8 => {
                let (lo, hi) = match bounds_i8(low, high) {
                    Ok(v) => v,
                    Err(e) => {
                        crate::error::set_last_error(e);
                        return ERR_GENERIC;
                    }
                };
                let data: Vec<i8> = (0..len).map(|_| rng.random_range(lo..hi)).collect();
                let arr = ArrayD::<i8>::from_shape_vec(IxDyn(shape_slice), data)
                    .expect("Shape mismatch should not happen");
                NDArrayWrapper {
                    data: ArrayData::Int8(Arc::new(RwLock::new(arr))),
                    dtype: DType::Int8,
                }
            }
            DType::Int16 => {
                let (lo, hi) = match bounds_i16(low, high) {
                    Ok(v) => v,
                    Err(e) => {
                        crate::error::set_last_error(e);
                        return ERR_GENERIC;
                    }
                };
                let data: Vec<i16> = (0..len).map(|_| rng.random_range(lo..hi)).collect();
                let arr = ArrayD::<i16>::from_shape_vec(IxDyn(shape_slice), data)
                    .expect("Shape mismatch should not happen");
                NDArrayWrapper {
                    data: ArrayData::Int16(Arc::new(RwLock::new(arr))),
                    dtype: DType::Int16,
                }
            }
            DType::Int32 => {
                let (lo, hi) = match bounds_i32(low, high) {
                    Ok(v) => v,
                    Err(e) => {
                        crate::error::set_last_error(e);
                        return ERR_GENERIC;
                    }
                };
                let data: Vec<i32> = (0..len).map(|_| rng.random_range(lo..hi)).collect();
                let arr = ArrayD::<i32>::from_shape_vec(IxDyn(shape_slice), data)
                    .expect("Shape mismatch should not happen");
                NDArrayWrapper {
                    data: ArrayData::Int32(Arc::new(RwLock::new(arr))),
                    dtype: DType::Int32,
                }
            }
            DType::Int64 => {
                let (lo, hi) = match bounds_i64(low, high) {
                    Ok(v) => v,
                    Err(e) => {
                        crate::error::set_last_error(e);
                        return ERR_GENERIC;
                    }
                };
                let data: Vec<i64> = (0..len).map(|_| rng.random_range(lo..hi)).collect();
                let arr = ArrayD::<i64>::from_shape_vec(IxDyn(shape_slice), data)
                    .expect("Shape mismatch should not happen");
                NDArrayWrapper {
                    data: ArrayData::Int64(Arc::new(RwLock::new(arr))),
                    dtype: DType::Int64,
                }
            }
            DType::Uint8 => {
                let (lo, hi) = match bounds_u8(low, high) {
                    Ok(v) => v,
                    Err(e) => {
                        crate::error::set_last_error(e);
                        return ERR_GENERIC;
                    }
                };
                let data: Vec<u8> = (0..len).map(|_| rng.random_range(lo..hi)).collect();
                let arr = ArrayD::<u8>::from_shape_vec(IxDyn(shape_slice), data)
                    .expect("Shape mismatch should not happen");
                NDArrayWrapper {
                    data: ArrayData::Uint8(Arc::new(RwLock::new(arr))),
                    dtype: DType::Uint8,
                }
            }
            DType::Uint16 => {
                let (lo, hi) = match bounds_u16(low, high) {
                    Ok(v) => v,
                    Err(e) => {
                        crate::error::set_last_error(e);
                        return ERR_GENERIC;
                    }
                };
                let data: Vec<u16> = (0..len).map(|_| rng.random_range(lo..hi)).collect();
                let arr = ArrayD::<u16>::from_shape_vec(IxDyn(shape_slice), data)
                    .expect("Shape mismatch should not happen");
                NDArrayWrapper {
                    data: ArrayData::Uint16(Arc::new(RwLock::new(arr))),
                    dtype: DType::Uint16,
                }
            }
            DType::Uint32 => {
                let (lo, hi) = match bounds_u32(low, high) {
                    Ok(v) => v,
                    Err(e) => {
                        crate::error::set_last_error(e);
                        return ERR_GENERIC;
                    }
                };
                let data: Vec<u32> = (0..len).map(|_| rng.random_range(lo..hi)).collect();
                let arr = ArrayD::<u32>::from_shape_vec(IxDyn(shape_slice), data)
                    .expect("Shape mismatch should not happen");
                NDArrayWrapper {
                    data: ArrayData::Uint32(Arc::new(RwLock::new(arr))),
                    dtype: DType::Uint32,
                }
            }
            DType::Uint64 => {
                let (lo, hi) = match bounds_u64(low, high) {
                    Ok(v) => v,
                    Err(e) => {
                        crate::error::set_last_error(e);
                        return ERR_GENERIC;
                    }
                };
                let data: Vec<u64> = (0..len).map(|_| rng.random_range(lo..hi)).collect();
                let arr = ArrayD::<u64>::from_shape_vec(IxDyn(shape_slice), data)
                    .expect("Shape mismatch should not happen");
                NDArrayWrapper {
                    data: ArrayData::Uint64(Arc::new(RwLock::new(arr))),
                    dtype: DType::Uint64,
                }
            }
            _ => return ERR_DTYPE,
        };

        *out_handle = NdArrayHandle::from_wrapper(Box::new(wrapper));
        SUCCESS
    })
}
