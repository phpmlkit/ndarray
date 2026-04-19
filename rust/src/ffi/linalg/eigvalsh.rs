//! Hermitian / symmetric eigenvalues only (no eigenvectors).

use std::sync::Arc;

use ndarray::Ix2;
use ndarray_linalg::EigValsh;
use parking_lot::RwLock;

use crate::helpers::error::{self, ERR_DTYPE, ERR_GENERIC, ERR_MATH, ERR_SHAPE, SUCCESS};
use crate::helpers::validation::{adjust_uplo_for_layout, uplo_from_int, validate_square_matrix};
use crate::helpers::write_output_metadata;
use crate::helpers::{extract_view_c128, extract_view_c64, extract_view_f32, extract_view_f64};
use crate::types::{ArrayData, ArrayMetadata, DType, NDArrayWrapper, NdArrayHandle};

/// Compute eigenvalues only (no eigenvectors) for a Hermitian/symmetric matrix.
///
/// Eigenvalues are always real.
///
/// # Arguments
/// * `a` - Input matrix handle (n x n, 2D, square, Hermitian/symmetric)
/// * `a_meta` - Array metadata
/// * `uplo` - 0 for Lower triangle, 1 for Upper triangle
/// * `out_eigvals` - Output eigenvalues vector handle (length n, real)
/// * `out_dtype_eigvals` - Eigenvalues dtype output
/// * `out_ndim_eigvals` - Eigenvalues ndim output
/// * `out_shape_eigvals` - Eigenvalues shape output
/// * `max_ndim` - Maximum number of dimensions
#[no_mangle]
pub unsafe extern "C" fn ndarray_eigvalsh(
    a: *const NdArrayHandle,
    a_meta: *const ArrayMetadata,
    uplo: u8,
    out_eigvals: *mut *mut NdArrayHandle,
    out_dtype_eigvals: *mut u8,
    out_ndim_eigvals: *mut usize,
    out_shape_eigvals: *mut usize,
    max_ndim: usize,
) -> i32 {
    if a.is_null() || a_meta.is_null() || out_eigvals.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let a_meta_ref = &*a_meta;
        let a_wrapper = NdArrayHandle::as_wrapper(a as *mut _);

        let status = validate_square_matrix(a_meta_ref, "EigValsH");
        if status != SUCCESS {
            return status;
        }

        let uplo = uplo_from_int(uplo);

        let eigvals_wrapper = match a_wrapper.dtype {
            DType::Float64 => {
                let Some(a_view_dyn) = extract_view_f64(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract f64 view for EigValsH".to_string());
                    return ERR_GENERIC;
                };
                let a_view_2d = match a_view_dyn.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                let uplo = adjust_uplo_for_layout(&a_view_2d, uplo);
                let eigvals = match a_view_2d.eigvalsh(uplo) {
                    Ok(vals) => vals,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_MATH;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(eigvals.into_dyn()))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let Some(a_view_dyn) = extract_view_f32(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract f32 view for EigValsH".to_string());
                    return ERR_GENERIC;
                };
                let a_view_2d = match a_view_dyn.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                let uplo = adjust_uplo_for_layout(&a_view_2d, uplo);
                let eigvals = match a_view_2d.eigvalsh(uplo) {
                    Ok(vals) => vals,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_MATH;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(eigvals.into_dyn()))),
                    dtype: DType::Float32,
                }
            }
            DType::Complex64 => {
                let Some(a_view_dyn) = extract_view_c64(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract c64 view for EigValsH".to_string());
                    return ERR_GENERIC;
                };
                let a_view_2d = match a_view_dyn.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                let uplo = adjust_uplo_for_layout(&a_view_2d, uplo);
                let eigvals = match a_view_2d.eigvalsh(uplo) {
                    Ok(vals) => vals,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_MATH;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(eigvals.into_dyn()))),
                    dtype: DType::Float32,
                }
            }
            DType::Complex128 => {
                let Some(a_view_dyn) = extract_view_c128(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract c128 view for EigValsH".to_string());
                    return ERR_GENERIC;
                };
                let a_view_2d = match a_view_dyn.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                let uplo = adjust_uplo_for_layout(&a_view_2d, uplo);
                let eigvals = match a_view_2d.eigvalsh(uplo) {
                    Ok(vals) => vals,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_MATH;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(eigvals.into_dyn()))),
                    dtype: DType::Float64,
                }
            }
            _ => {
                error::set_last_error(
                    "EigValsH only supports Float32, Float64, Complex64, and Complex128"
                        .to_string(),
                );
                return ERR_DTYPE;
            }
        };

        if let Err(e) = write_output_metadata(
            &eigvals_wrapper,
            out_dtype_eigvals,
            out_ndim_eigvals,
            out_shape_eigvals,
            max_ndim,
        ) {
            error::set_last_error(e);
            return ERR_GENERIC;
        }
        *out_eigvals = NdArrayHandle::from_wrapper(Box::new(eigvals_wrapper));

        SUCCESS
    })
}
