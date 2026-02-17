//! Sigmoid operation: 1 / (1 + exp(-x)).
//!
//! ndarray has no built-in sigmoid; uses mapv with the standard formula.

use crate::core::view_helpers::{extract_view_f32, extract_view_f64};
use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::NdArrayHandle;
use parking_lot::RwLock;
use std::slice;
use std::sync::Arc;

/// Compute sigmoid element-wise: 1 / (1 + exp(-x)).
#[no_mangle]
pub unsafe extern "C" fn ndarray_sigmoid(
    a: *const NdArrayHandle,
    a_offset: usize,
    a_shape: *const usize,
    a_strides: *const usize,
    ndim: usize,
    out: *mut *mut NdArrayHandle,
) -> i32 {
    if a.is_null() || out.is_null() || a_shape.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let a_wrapper = NdArrayHandle::as_wrapper(a as *mut _);
        let a_shape_slice = slice::from_raw_parts(a_shape, ndim);
        let a_strides_slice = slice::from_raw_parts(a_strides, ndim);

        let result_wrapper = match a_wrapper.dtype {
            DType::Float64 => {
                let Some(view) =
                    extract_view_f64(a_wrapper, a_offset, a_shape_slice, a_strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.mapv(|x: f64| 1.0 / (1.0 + (-x).exp()));
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let Some(view) =
                    extract_view_f32(a_wrapper, a_offset, a_shape_slice, a_strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.mapv(|x: f32| 1.0_f32 / (1.0_f32 + (-x).exp()));
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(result))),
                    dtype: DType::Float32,
                }
            }
            _ => {
                crate::error::set_last_error(
                    "sigmoid() requires float type (Float64 or Float32)".to_string(),
                );
                return ERR_GENERIC;
            }
        };

        *out = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
