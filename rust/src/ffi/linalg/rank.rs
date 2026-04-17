//! Matrix Rank
//!
//! Compute the rank of a matrix using SVD.

use ndarray::Ix2;
use ndarray_linalg::SVD;

use crate::helpers::error::{self, ERR_DTYPE, ERR_GENERIC, ERR_MATH, ERR_SHAPE};
use crate::helpers::{extract_view_f32, extract_view_f64};
use crate::types::{ArrayMetadata, DType, NdArrayHandle};

/// Compute the rank of a matrix.
///
/// `tol` is passed as a pointer. If null, the default tolerance is used:
/// max(m, n) * eps * max(singular_value)
#[no_mangle]
pub unsafe extern "C" fn ndarray_rank(
    a: *const NdArrayHandle,
    a_meta: *const ArrayMetadata,
    tol: *const f64,
    out_rank: *mut i32,
) -> i32 {
    if a.is_null() || a_meta.is_null() || out_rank.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let a_meta_ref = &*a_meta;
        let a_wrapper = NdArrayHandle::as_wrapper(a as *mut _);

        if a_meta_ref.ndim != 2 {
            error::set_last_error("Rank requires a 2D matrix".to_string());
            return ERR_SHAPE;
        }

        match a_wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract f64 view for rank".to_string());
                    return ERR_GENERIC;
                };
                let a_arr = match view.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(format!("Rank: failed to convert to 2D: {}", e));
                        return ERR_SHAPE;
                    }
                };
                let (_, s, _) = match a_arr.svd(false, false) {
                    Ok(r) => r,
                    Err(e) => {
                        error::set_last_error(format!("Rank SVD failed: {:?}", e));
                        return ERR_MATH;
                    }
                };
                let m = a_arr.nrows();
                let n = a_arr.ncols();
                let s_max = s.iter().cloned().fold(0.0f64, f64::max);
                let tolerance = if tol.is_null() {
                    (m.max(n) as f64) * f64::EPSILON * s_max
                } else {
                    *tol
                };
                let rank = s.iter().filter(|&&si| si > tolerance).count() as i32;
                *out_rank = rank;
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract f32 view for rank".to_string());
                    return ERR_GENERIC;
                };
                let a_arr = match view.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(format!("Rank: failed to convert to 2D: {}", e));
                        return ERR_SHAPE;
                    }
                };
                let (_, s, _) = match a_arr.svd(false, false) {
                    Ok(r) => r,
                    Err(e) => {
                        error::set_last_error(format!("Rank SVD failed: {:?}", e));
                        return ERR_MATH;
                    }
                };
                let m = a_arr.nrows();
                let n = a_arr.ncols();
                let s_max = s.iter().cloned().fold(0.0f32, f32::max);
                let tolerance = if tol.is_null() {
                    (m.max(n) as f32) * f32::EPSILON * s_max
                } else {
                    *tol as f32
                };
                let rank = s.iter().filter(|&&si| si > tolerance).count() as i32;
                *out_rank = rank;
            }
            _ => {
                error::set_last_error("Rank only supports Float32 and Float64".to_string());
                return ERR_DTYPE;
            }
        }

        crate::helpers::error::SUCCESS
    })
}
