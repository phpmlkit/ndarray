//! Softmax along axis: exp(x - max) / sum(exp(x - max)).
//!
//! Numerically stable implementation. Float-only. Output shape equals input shape.

use crate::core::view_helpers::{extract_view_f32, extract_view_f64};
use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::reductions::helpers::validate_axis;
use crate::ffi::NdArrayHandle;
use ndarray::Axis;
use parking_lot::RwLock;
use std::slice;
use std::sync::Arc;

/// Softmax along axis. Numerically stable: subtract max before exp.
#[no_mangle]
pub unsafe extern "C" fn ndarray_softmax(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
    axis: i32,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || out_handle.is_null() || shape.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = slice::from_raw_parts(shape, ndim);
        let strides_slice = slice::from_raw_parts(strides, ndim);

        let axis_usize = match validate_axis(shape_slice, axis) {
            Ok(a) => a,
            Err(e) => {
                crate::error::set_last_error(e);
                return ERR_GENERIC;
            }
        };

        let result_wrapper = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                let mut result = view.to_owned();
                for (in_lane, mut out_lane) in view
                        .lanes(Axis(axis_usize))
                        .into_iter()
                        .zip(result.lanes_mut(Axis(axis_usize)))
                    {
                        let max = in_lane
                            .iter()
                            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                        let sum_exp: f64 = in_lane.iter().map(|&x| (x - max).exp()).sum();
                        if sum_exp > 0.0 && sum_exp.is_finite() {
                            for (i, &x) in in_lane.iter().enumerate() {
                                out_lane[i] = (x - max).exp() / sum_exp;
                            }
                        } else {
                            let n = in_lane.len() as f64;
                            for i in 0..in_lane.len() {
                                out_lane[i] = 1.0 / n;
                            }
                        }
                    }
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                let mut result = view.to_owned();
                for (in_lane, mut out_lane) in view
                        .lanes(Axis(axis_usize))
                        .into_iter()
                        .zip(result.lanes_mut(Axis(axis_usize)))
                    {
                        let max = in_lane
                            .iter()
                            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                        let sum_exp: f32 = in_lane.iter().map(|&x| (x - max).exp()).sum();
                        if sum_exp > 0.0 && sum_exp.is_finite() {
                            for (i, &x) in in_lane.iter().enumerate() {
                                out_lane[i] = (x - max).exp() / sum_exp;
                            }
                        } else {
                            let n = in_lane.len() as f32;
                            for i in 0..in_lane.len() {
                                out_lane[i] = 1.0_f32 / n;
                            }
                        }
                    }
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(result))),
                    dtype: DType::Float32,
                }
            }
            _ => {
                crate::error::set_last_error(
                    "softmax() requires float type (Float64 or Float32)".to_string(),
                );
                return ERR_GENERIC;
            }
        };

        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
