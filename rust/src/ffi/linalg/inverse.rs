//! Matrix Inverse
//!
//! Compute the inverse of a square matrix using LU decomposition.

use std::sync::Arc;

use ndarray::Ix2;
use ndarray_linalg::Inverse;
use parking_lot::RwLock;

use crate::helpers::error::{self, ERR_DTYPE, ERR_GENERIC, ERR_MATH, ERR_SHAPE, SUCCESS};
use crate::helpers::write_output_metadata;
use crate::helpers::{extract_view_c128, extract_view_c64, extract_view_f32, extract_view_f64};
use crate::types::{ArrayData, ArrayMetadata, DType, NDArrayWrapper, NdArrayHandle};

/// Compute the inverse of a square matrix.
#[no_mangle]
pub unsafe extern "C" fn ndarray_inv(
    a: *const NdArrayHandle,
    a_meta: *const ArrayMetadata,
    out_handle: *mut *mut NdArrayHandle,
    out_dtype: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if a.is_null() || a_meta.is_null() || out_handle.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let a_meta_ref = &*a_meta;
        let a_wrapper = NdArrayHandle::as_wrapper(a as *mut _);

        if a_meta_ref.ndim != 2 {
            error::set_last_error("Inverse requires a 2D matrix".to_string());
            return ERR_SHAPE;
        }

        let shape = a_meta_ref.shape_slice();
        if shape[0] != shape[1] {
            error::set_last_error("Inverse requires a square matrix".to_string());
            return ERR_SHAPE;
        }

        let result_wrapper = match a_wrapper.dtype {
            DType::Float64 => {
                let Some(a_view_dyn) = extract_view_f64(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract f64 view for inverse".to_string());
                    return ERR_GENERIC;
                };
                let a_view_2d = match a_view_dyn.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                let result = match a_view_2d.inv() {
                    Ok(r) => r,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_MATH;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result.into_dyn()))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let Some(a_view_dyn) = extract_view_f32(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract f32 view for inverse".to_string());
                    return ERR_GENERIC;
                };
                let a_view_2d = match a_view_dyn.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                let result = match a_view_2d.inv() {
                    Ok(r) => r,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_MATH;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(result.into_dyn()))),
                    dtype: DType::Float32,
                }
            }
            DType::Complex64 => {
                let Some(a_view_dyn) = extract_view_c64(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract c64 view for inverse".to_string());
                    return ERR_GENERIC;
                };
                let a_view_2d = match a_view_dyn.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                let result = match a_view_2d.inv() {
                    Ok(r) => r,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_MATH;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Complex64(Arc::new(RwLock::new(result.into_dyn()))),
                    dtype: DType::Complex64,
                }
            }
            DType::Complex128 => {
                let Some(a_view_dyn) = extract_view_c128(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract c128 view for inverse".to_string());
                    return ERR_GENERIC;
                };
                let a_view_2d = match a_view_dyn.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                let result = match a_view_2d.inv() {
                    Ok(r) => r,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_MATH;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Complex128(Arc::new(RwLock::new(result.into_dyn()))),
                    dtype: DType::Complex128,
                }
            }
            _ => {
                error::set_last_error(
                    "Inverse only supports Float32, Float64, Complex64, and Complex128".to_string(),
                );
                return ERR_DTYPE;
            }
        };

        if let Err(e) =
            write_output_metadata(&result_wrapper, out_dtype, out_ndim, out_shape, max_ndim)
        {
            error::set_last_error(e);
            return ERR_GENERIC;
        }
        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
