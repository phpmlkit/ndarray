//! Matrix Determinant
//!
//! Compute the determinant of a square matrix.

use std::ffi::c_void;

use ndarray::Ix2;
use ndarray_linalg::Determinant;

use crate::helpers::error::{self, ERR_DTYPE, ERR_GENERIC, ERR_MATH, ERR_SHAPE, SUCCESS};
use crate::helpers::{extract_view_c128, extract_view_c64, extract_view_f32, extract_view_f64};
use crate::types::{ArrayMetadata, DType, NdArrayHandle};

/// Compute the determinant of a square matrix.
#[no_mangle]
pub unsafe extern "C" fn ndarray_det(
    a: *const NdArrayHandle,
    a_meta: *const ArrayMetadata,
    out_value: *mut c_void,
    out_dtype: *mut u8,
) -> i32 {
    if a.is_null() || a_meta.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let a_meta_ref = &*a_meta;
        let a_wrapper = NdArrayHandle::as_wrapper(a as *mut _);

        if a_meta_ref.ndim != 2 {
            error::set_last_error("Determinant requires a 2D matrix".to_string());
            return ERR_SHAPE;
        }

        let shape = a_meta_ref.shape_slice();
        if shape[0] != shape[1] {
            error::set_last_error("Determinant requires a square matrix".to_string());
            return ERR_MATH;
        }

        match a_wrapper.dtype {
            DType::Float64 => {
                let Some(a_view_dyn) = extract_view_f64(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract f64 view for determinant".to_string());
                    return ERR_GENERIC;
                };
                let a_view_2d = match a_view_dyn.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                let det = match a_view_2d.det() {
                    Ok(d) => d,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_MATH;
                    }
                };
                if !out_value.is_null() {
                    *(out_value as *mut f64) = det;
                }
                if !out_dtype.is_null() {
                    *out_dtype = DType::Float64 as u8;
                }
            }
            DType::Float32 => {
                let Some(a_view_dyn) = extract_view_f32(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract f32 view for determinant".to_string());
                    return ERR_GENERIC;
                };
                let a_view_2d = match a_view_dyn.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                let det = match a_view_2d.det() {
                    Ok(d) => d,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_MATH;
                    }
                };
                if !out_value.is_null() {
                    *(out_value as *mut f32) = det;
                }
                if !out_dtype.is_null() {
                    *out_dtype = DType::Float32 as u8;
                }
            }
            DType::Complex64 => {
                let Some(a_view_dyn) = extract_view_c64(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract c64 view for determinant".to_string());
                    return ERR_GENERIC;
                };
                let a_view_2d = match a_view_dyn.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                let det = match a_view_2d.det() {
                    Ok(d) => d,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_MATH;
                    }
                };
                if !out_value.is_null() {
                    *(out_value as *mut num_complex::Complex<f32>) = det;
                }
                if !out_dtype.is_null() {
                    *out_dtype = DType::Complex64 as u8;
                }
            }
            DType::Complex128 => {
                let Some(a_view_dyn) = extract_view_c128(a_wrapper, a_meta_ref) else {
                    error::set_last_error(
                        "Failed to extract c128 view for determinant".to_string(),
                    );
                    return ERR_GENERIC;
                };
                let a_view_2d = match a_view_dyn.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                let det = match a_view_2d.det() {
                    Ok(d) => d,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_MATH;
                    }
                };
                if !out_value.is_null() {
                    *(out_value as *mut num_complex::Complex<f64>) = det;
                }
                if !out_dtype.is_null() {
                    *out_dtype = DType::Complex128 as u8;
                }
            }
            _ => {
                error::set_last_error(
                    "Determinant only supports Float32, Float64, Complex64, and Complex128"
                        .to_string(),
                );
                return ERR_DTYPE;
            }
        }

        SUCCESS
    })
}
