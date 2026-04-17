//! Pseudo-inverse
//!
//! Compute the Moore-Penrose pseudo-inverse using SVD.

use std::sync::Arc;

use ndarray::{Array2, Ix2};
use ndarray_linalg::SVD;
use parking_lot::RwLock;
use std::ffi::c_void;

use crate::helpers::error::{self, ERR_DTYPE, ERR_GENERIC, ERR_MATH, ERR_SHAPE, SUCCESS};
use crate::helpers::write_output_metadata;
use crate::helpers::{extract_view_f32, extract_view_f64};
use crate::types::{ArrayData, ArrayMetadata, DType, NDArrayWrapper, NdArrayHandle};

fn pinv_f64(a: ndarray::ArrayView2<f64>, rcond: Option<f64>) -> Result<Array2<f64>, String> {
    let (u, s, vt) = a
        .svd(true, true)
        .map_err(|e| format!("SVD failed for pinv: {:?}", e))?;
    let u = u.ok_or("SVD did not return U")?;
    let vt = vt.ok_or("SVD did not return VT")?;

    let m = a.nrows();
    let n = a.ncols();
    let max_mn = m.max(n) as f64;
    let tol =
        rcond.unwrap_or_else(|| max_mn * f64::EPSILON) * s.iter().cloned().fold(0.0, f64::max);

    let k = s.len();
    let mut s_inv = Array2::<f64>::zeros((k, k));
    for (i, &si) in s.iter().enumerate() {
        if si > tol {
            s_inv[(i, i)] = 1.0 / si;
        }
    }

    // ndarray-linalg returns full U (m x m) and VT (n x n); slice to thin forms
    let u_thin = u.slice(ndarray::s![.., 0..k]);
    let vt_thin = vt.slice(ndarray::s![0..k, ..]);
    let v = vt_thin.t();
    let ut = u_thin.t();
    Ok(v.dot(&s_inv).dot(&ut))
}

fn pinv_f32(a: ndarray::ArrayView2<f32>, rcond: Option<f32>) -> Result<Array2<f32>, String> {
    let (u, s, vt) = a
        .svd(true, true)
        .map_err(|e| format!("SVD failed for pinv: {:?}", e))?;
    let u = u.ok_or("SVD did not return U")?;
    let vt = vt.ok_or("SVD did not return VT")?;

    let m = a.nrows();
    let n = a.ncols();
    let max_mn = m.max(n) as f32;
    let tol =
        rcond.unwrap_or_else(|| max_mn * f32::EPSILON) * s.iter().cloned().fold(0.0, f32::max);

    let k = s.len();
    let mut s_inv = Array2::<f32>::zeros((k, k));
    for (i, &si) in s.iter().enumerate() {
        if si > tol {
            s_inv[(i, i)] = 1.0 / si;
        }
    }

    let u_thin = u.slice(ndarray::s![.., 0..k]);
    let vt_thin = vt.slice(ndarray::s![0..k, ..]);
    let v = vt_thin.t();
    let ut = u_thin.t();
    Ok(v.dot(&s_inv).dot(&ut))
}

/// Compute the Moore-Penrose pseudo-inverse of a matrix.
///
/// `rcond` is passed as a pointer. If it is null, the default tolerance is used.
#[no_mangle]
pub unsafe extern "C" fn ndarray_pinv(
    a: *const NdArrayHandle,
    a_meta: *const ArrayMetadata,
    rcond: *const c_void,
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
            error::set_last_error("Pinv requires a 2D matrix".to_string());
            return ERR_SHAPE;
        }

        let result_wrapper = match a_wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract f64 view for pinv".to_string());
                    return ERR_GENERIC;
                };
                let a_arr = match view.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(format!("Pinv: failed to convert to 2D: {}", e));
                        return ERR_SHAPE;
                    }
                };
                let rcond_val = if rcond.is_null() {
                    None
                } else {
                    Some(*(rcond as *const f64))
                };
                let result = match pinv_f64(a_arr, rcond_val) {
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
                let Some(view) = extract_view_f32(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract f32 view for pinv".to_string());
                    return ERR_GENERIC;
                };
                let a_arr = match view.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(format!("Pinv: failed to convert to 2D: {}", e));
                        return ERR_SHAPE;
                    }
                };
                let rcond_val = if rcond.is_null() {
                    None
                } else {
                    Some(*(rcond as *const f32))
                };
                let result = match pinv_f32(a_arr, rcond_val) {
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
                error::set_last_error("Pinv only supports Float32 and Float64".to_string());
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
