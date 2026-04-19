//! Compute phase angle element-wise.
//!
//! Returns the counterclockwise angle from the positive real axis on the complex plane.
//! Always returns Float64 regardless of input dtype.
//! For real arrays: 0.0 for non-negative, π for negative.
//! For complex arrays: atan2(imag, real).

use crate::helpers::error::{set_last_error, ERR_GENERIC, SUCCESS};
use crate::helpers::write_output_metadata;
use crate::helpers::{
    extract_view_bool, extract_view_c128, extract_view_c64, extract_view_f32, extract_view_f64,
    extract_view_i16, extract_view_i32, extract_view_i64, extract_view_i8, extract_view_u16,
    extract_view_u32, extract_view_u64, extract_view_u8,
};
use crate::types::dtype::DType;
use crate::types::{ArrayData, ArrayMetadata, NDArrayWrapper, NdArrayHandle};
use ndarray::ArrayD;
use parking_lot::RwLock;
use std::sync::Arc;

/// Compute phase angle element-wise (always returns Float64).
#[no_mangle]
pub unsafe extern "C" fn ndarray_angle(
    a: *const NdArrayHandle,
    meta: *const ArrayMetadata,
    deg: u8,
    out: *mut *mut NdArrayHandle,
    out_dtype: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if a.is_null()
        || meta.is_null()
        || out.is_null()
        || out_dtype.is_null()
        || out_ndim.is_null()
        || out_shape.is_null()
    {
        return ERR_GENERIC;
    }

    let scale = if deg != 0 {
        180.0 / std::f64::consts::PI
    } else {
        1.0
    };

    crate::ffi_guard!({
        let meta = &*meta;
        let a_wrapper = NdArrayHandle::as_wrapper(a as *mut _);

        let result_wrapper = match a_wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(a_wrapper, meta) else {
                    set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.mapv(|x| {
                    let angle = if x >= 0.0 { 0.0 } else { std::f64::consts::PI };
                    angle * scale
                });
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(a_wrapper, meta) else {
                    set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.mapv(|x| {
                    let angle = if x >= 0.0 { 0.0 } else { std::f64::consts::PI };
                    angle * scale
                });
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
                    dtype: DType::Float64,
                }
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(a_wrapper, meta) else {
                    set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.mapv(|x| {
                    let angle = if x >= 0 { 0.0 } else { std::f64::consts::PI };
                    angle * scale
                });
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
                    dtype: DType::Float64,
                }
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(a_wrapper, meta) else {
                    set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.mapv(|x| {
                    let angle = if x >= 0 { 0.0 } else { std::f64::consts::PI };
                    angle * scale
                });
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
                    dtype: DType::Float64,
                }
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(a_wrapper, meta) else {
                    set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.mapv(|x| {
                    let angle = if x >= 0 { 0.0 } else { std::f64::consts::PI };
                    angle * scale
                });
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
                    dtype: DType::Float64,
                }
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(a_wrapper, meta) else {
                    set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.mapv(|x| {
                    let angle = if x >= 0 { 0.0 } else { std::f64::consts::PI };
                    angle * scale
                });
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
                    dtype: DType::Float64,
                }
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(a_wrapper, meta) else {
                    set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = ArrayD::zeros(view.raw_dim());
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
                    dtype: DType::Float64,
                }
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(a_wrapper, meta) else {
                    set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                let result = ArrayD::zeros(view.raw_dim());
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
                    dtype: DType::Float64,
                }
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(a_wrapper, meta) else {
                    set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                let result = ArrayD::zeros(view.raw_dim());
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
                    dtype: DType::Float64,
                }
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(a_wrapper, meta) else {
                    set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                let result = ArrayD::zeros(view.raw_dim());
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
                    dtype: DType::Float64,
                }
            }
            DType::Bool => {
                let Some(view) = extract_view_bool(a_wrapper, meta) else {
                    set_last_error("Failed to extract bool view".to_string());
                    return ERR_GENERIC;
                };
                let result = ArrayD::zeros(view.raw_dim());
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
                    dtype: DType::Float64,
                }
            }
            DType::Complex64 => {
                let Some(view) = extract_view_c64(a_wrapper, meta) else {
                    set_last_error("Failed to extract Complex64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.mapv(|x| {
                    let angle = x.arg() as f64;
                    angle * scale
                });
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
                    dtype: DType::Float64,
                }
            }
            DType::Complex128 => {
                let Some(view) = extract_view_c128(a_wrapper, meta) else {
                    set_last_error("Failed to extract Complex128 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.mapv(|x| {
                    let angle = x.arg();
                    angle * scale
                });
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
                    dtype: DType::Float64,
                }
            }
        };

        if let Err(e) =
            write_output_metadata(&result_wrapper, out_dtype, out_ndim, out_shape, max_ndim)
        {
            set_last_error(e);
            return ERR_GENERIC;
        }
        *out = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
