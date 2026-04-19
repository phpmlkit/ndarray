//! Hermitian / symmetric eigenvalue decomposition (eigenvalues and eigenvectors).
//!
//! Eigenvalues are real. Eigenvectors match the input element type. The `uplo` argument
//! selects the stored triangle; it is adjusted when the matrix is C-contiguous (see
//! [`crate::helpers::validation::adjust_uplo_for_layout`]).

use std::sync::Arc;

use ndarray::Ix2;
use ndarray_linalg::Eigh;
use parking_lot::RwLock;

use crate::helpers::error::{self, ERR_DTYPE, ERR_GENERIC, ERR_MATH, ERR_SHAPE, SUCCESS};
use crate::helpers::validation::{adjust_uplo_for_layout, uplo_from_int, validate_square_matrix};
use crate::helpers::write_output_metadata;
use crate::helpers::{extract_view_c128, extract_view_c64, extract_view_f32, extract_view_f64};
use crate::types::{ArrayData, ArrayMetadata, DType, NDArrayWrapper, NdArrayHandle};

/// Compute eigenvalue decomposition for Hermitian/symmetric matrix.
///
/// Eigenvalues are always real. Eigenvectors have the same type as input.
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
/// * `out_eigvecs` - Output eigenvectors matrix handle (n x n)
/// * `out_dtype_eigvecs` - Eigenvectors dtype output
/// * `out_ndim_eigvecs` - Eigenvectors ndim output
/// * `out_shape_eigvecs` - Eigenvectors shape output
#[no_mangle]
pub unsafe extern "C" fn ndarray_eigh(
    a: *const NdArrayHandle,
    a_meta: *const ArrayMetadata,
    uplo: u8,
    out_eigvals: *mut *mut NdArrayHandle,
    out_dtype_eigvals: *mut u8,
    out_ndim_eigvals: *mut usize,
    out_shape_eigvals: *mut usize,
    max_ndim: usize,
    out_eigvecs: *mut *mut NdArrayHandle,
    out_dtype_eigvecs: *mut u8,
    out_ndim_eigvecs: *mut usize,
    out_shape_eigvecs: *mut usize,
) -> i32 {
    if a.is_null() || a_meta.is_null() || out_eigvals.is_null() || out_eigvecs.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let a_meta_ref = &*a_meta;
        let a_wrapper = NdArrayHandle::as_wrapper(a as *mut _);

        let status = validate_square_matrix(a_meta_ref, "Eigh");
        if status != SUCCESS {
            return status;
        }

        let uplo = uplo_from_int(uplo);

        let (eigvals_wrapper, eigvecs_wrapper) = match a_wrapper.dtype {
            DType::Float64 => {
                let Some(a_view_dyn) = extract_view_f64(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract f64 view for Eigh".to_string());
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
                let (eigvals, eigvecs) = match a_view_2d.eigh(uplo) {
                    Ok(pair) => pair,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_MATH;
                    }
                };
                let eigvecs = eigvecs.as_standard_layout().to_owned();
                let eigvals_wrapper = NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(eigvals.into_dyn()))),
                    dtype: DType::Float64,
                };
                let eigvecs_wrapper = NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(eigvecs.into_dyn()))),
                    dtype: DType::Float64,
                };
                (eigvals_wrapper, eigvecs_wrapper)
            }
            DType::Float32 => {
                let Some(a_view_dyn) = extract_view_f32(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract f32 view for Eigh".to_string());
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
                let (eigvals, eigvecs) = match a_view_2d.eigh(uplo) {
                    Ok(pair) => pair,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_MATH;
                    }
                };
                let eigvecs = eigvecs.as_standard_layout().to_owned();
                let eigvals_wrapper = NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(eigvals.into_dyn()))),
                    dtype: DType::Float32,
                };
                let eigvecs_wrapper = NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(eigvecs.into_dyn()))),
                    dtype: DType::Float32,
                };
                (eigvals_wrapper, eigvecs_wrapper)
            }
            DType::Complex64 => {
                let Some(a_view_dyn) = extract_view_c64(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract c64 view for Eigh".to_string());
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
                let (eigvals, eigvecs) = match a_view_2d.eigh(uplo) {
                    Ok(pair) => pair,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_MATH;
                    }
                };
                let eigvecs = eigvecs.as_standard_layout().to_owned();
                let eigvals_wrapper = NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(eigvals.into_dyn()))),
                    dtype: DType::Float32,
                };
                let eigvecs_wrapper = NDArrayWrapper {
                    data: ArrayData::Complex64(Arc::new(RwLock::new(eigvecs.into_dyn()))),
                    dtype: DType::Complex64,
                };
                (eigvals_wrapper, eigvecs_wrapper)
            }
            DType::Complex128 => {
                let Some(a_view_dyn) = extract_view_c128(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract c128 view for Eigh".to_string());
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
                let (eigvals, eigvecs) = match a_view_2d.eigh(uplo) {
                    Ok(pair) => pair,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_MATH;
                    }
                };
                let eigvecs = eigvecs.as_standard_layout().to_owned();
                let eigvals_wrapper = NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(eigvals.into_dyn()))),
                    dtype: DType::Float64,
                };
                let eigvecs_wrapper = NDArrayWrapper {
                    data: ArrayData::Complex128(Arc::new(RwLock::new(eigvecs.into_dyn()))),
                    dtype: DType::Complex128,
                };
                (eigvals_wrapper, eigvecs_wrapper)
            }
            _ => {
                error::set_last_error(
                    "Eigh only supports Float32, Float64, Complex64, and Complex128".to_string(),
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

        if let Err(e) = write_output_metadata(
            &eigvecs_wrapper,
            out_dtype_eigvecs,
            out_ndim_eigvecs,
            out_shape_eigvecs,
            max_ndim,
        ) {
            error::set_last_error(e);
            return ERR_GENERIC;
        }
        *out_eigvecs = NdArrayHandle::from_wrapper(Box::new(eigvecs_wrapper));

        SUCCESS
    })
}
