//! Cholesky Decomposition
//!
//! For a Hermitian positive-definite matrix A, decomposes A = L * L^H
//! where L is lower triangular, or A = U^H * U where U is upper triangular.

use std::sync::Arc;

use ndarray::Ix2;
use ndarray_linalg::{cholesky::UPLO, Cholesky};
use parking_lot::RwLock;

use crate::helpers::error::{self, ERR_DTYPE, ERR_GENERIC, ERR_MATH, ERR_SHAPE, SUCCESS};
use crate::helpers::write_output_metadata;
use crate::helpers::{extract_view_f32, extract_view_f64};
use crate::types::{ArrayData, ArrayMetadata, DType, NDArrayWrapper, NdArrayHandle};

/// Compute Cholesky decomposition of a Hermitian positive-definite matrix.
///
/// `upper` determines which triangular factor to return:
/// - `upper = 0` (false): returns L such that A = L * L^H
/// - `upper = 1` (true): returns U such that A = U^H * U
#[no_mangle]
pub unsafe extern "C" fn ndarray_cholesky(
    a: *const NdArrayHandle,
    a_meta: *const ArrayMetadata,
    upper: u8,
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
            error::set_last_error("Cholesky requires a 2D matrix".to_string());
            return ERR_SHAPE;
        }

        let shape = a_meta_ref.shape_slice();
        if shape[0] != shape[1] {
            error::set_last_error("Cholesky requires a square matrix".to_string());
            return ERR_SHAPE;
        }

        let uplo = if upper != 0 { UPLO::Upper } else { UPLO::Lower };

        let result_wrapper = match a_wrapper.dtype {
            DType::Float64 => {
                let Some(a_view_dyn) = extract_view_f64(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract f64 view for Cholesky".to_string());
                    return ERR_GENERIC;
                };
                let a_view_2d = match a_view_dyn.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                let result = match a_view_2d.cholesky(uplo) {
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
                    error::set_last_error("Failed to extract f32 view for Cholesky".to_string());
                    return ERR_GENERIC;
                };
                let a_view_2d = match a_view_dyn.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                let result = match a_view_2d.cholesky(uplo) {
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
            _ => {
                error::set_last_error("Cholesky only supports Float32 and Float64".to_string());
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
