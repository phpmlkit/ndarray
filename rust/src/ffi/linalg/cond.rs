//! Condition Number
//!
//! Compute the 2-norm condition number using SVD.

use std::ffi::c_void;

use ndarray::Ix2;
use ndarray_linalg::SVD;

use crate::ffi::reductions::helpers::{write_reduction_scalar, ReductionScalar};
use crate::helpers::error::{self, ERR_DTYPE, ERR_GENERIC, ERR_MATH, ERR_SHAPE};
use crate::helpers::{extract_view_c128, extract_view_c64, extract_view_f32, extract_view_f64};
use crate::types::{ArrayMetadata, DType, NdArrayHandle};

/// Compute the condition number of a matrix.
#[no_mangle]
pub unsafe extern "C" fn ndarray_cond(
    a: *const NdArrayHandle,
    a_meta: *const ArrayMetadata,
    out_value: *mut c_void,
    out_dtype: *mut u8,
) -> i32 {
    if a.is_null() || a_meta.is_null() || out_value.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let a_meta_ref = &*a_meta;
        let a_wrapper = NdArrayHandle::as_wrapper(a as *mut _);

        if a_meta_ref.ndim != 2 {
            error::set_last_error("Cond requires a 2D matrix".to_string());
            return ERR_SHAPE;
        }

        match a_wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract f64 view for cond".to_string());
                    return ERR_GENERIC;
                };
                let a_arr = match view.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(format!("Cond: failed to convert to 2D: {}", e));
                        return ERR_SHAPE;
                    }
                };
                let (_, s, _) = match a_arr.svd(false, false) {
                    Ok(r) => r,
                    Err(e) => {
                        error::set_last_error(format!("Cond SVD failed: {:?}", e));
                        return ERR_MATH;
                    }
                };
                let s_max = s.iter().cloned().fold(0.0f64, f64::max);
                let s_min = s.iter().cloned().fold(s_max, f64::min);
                let cond_val = if s_min <= f64::EPSILON * s_max {
                    f64::INFINITY
                } else {
                    s_max / s_min
                };
                write_reduction_scalar(out_value, out_dtype, ReductionScalar::F64(cond_val));
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract f32 view for cond".to_string());
                    return ERR_GENERIC;
                };
                let a_arr = match view.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(format!("Cond: failed to convert to 2D: {}", e));
                        return ERR_SHAPE;
                    }
                };
                let (_, s, _) = match a_arr.svd(false, false) {
                    Ok(r) => r,
                    Err(e) => {
                        error::set_last_error(format!("Cond SVD failed: {:?}", e));
                        return ERR_MATH;
                    }
                };
                let s_max = s.iter().cloned().fold(0.0f32, f32::max);
                let s_min = s.iter().cloned().fold(s_max, f32::min);
                let cond_val = if s_min <= f32::EPSILON * s_max {
                    f32::INFINITY
                } else {
                    s_max / s_min
                };
                write_reduction_scalar(out_value, out_dtype, ReductionScalar::F32(cond_val));
            }
            DType::Complex64 => {
                let Some(view) = extract_view_c64(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract c64 view for cond".to_string());
                    return ERR_GENERIC;
                };
                let a_arr = match view.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(format!("Cond: failed to convert to 2D: {}", e));
                        return ERR_SHAPE;
                    }
                };
                let (_, s, _) = match a_arr.svd(false, false) {
                    Ok(r) => r,
                    Err(e) => {
                        error::set_last_error(format!("Cond SVD failed: {:?}", e));
                        return ERR_MATH;
                    }
                };
                let s_max = s.iter().cloned().fold(0.0f32, f32::max);
                let s_min = s.iter().cloned().fold(s_max, f32::min);
                let cond_val = if s_min <= f32::EPSILON * s_max {
                    f32::INFINITY
                } else {
                    s_max / s_min
                };
                write_reduction_scalar(out_value, out_dtype, ReductionScalar::F32(cond_val));
            }
            DType::Complex128 => {
                let Some(view) = extract_view_c128(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract c128 view for cond".to_string());
                    return ERR_GENERIC;
                };
                let a_arr = match view.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(format!("Cond: failed to convert to 2D: {}", e));
                        return ERR_SHAPE;
                    }
                };
                let (_, s, _) = match a_arr.svd(false, false) {
                    Ok(r) => r,
                    Err(e) => {
                        error::set_last_error(format!("Cond SVD failed: {:?}", e));
                        return ERR_MATH;
                    }
                };
                let s_max = s.iter().cloned().fold(0.0f64, f64::max);
                let s_min = s.iter().cloned().fold(s_max, f64::min);
                let cond_val = if s_min <= f64::EPSILON * s_max {
                    f64::INFINITY
                } else {
                    s_max / s_min
                };
                write_reduction_scalar(out_value, out_dtype, ReductionScalar::F64(cond_val));
            }
            _ => {
                error::set_last_error(
                    "Cond only supports Float32, Float64, Complex64, and Complex128".to_string(),
                );
                return ERR_DTYPE;
            }
        }

        crate::helpers::error::SUCCESS
    })
}
