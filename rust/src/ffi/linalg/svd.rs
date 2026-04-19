//! Singular Value Decomposition (SVD)
//!
//! Computes the singular value decomposition of a matrix: A = U * S * V^T

use std::os::raw::c_uchar;
use std::sync::Arc;

use ndarray::Ix2;
use ndarray_linalg::SVD;
use parking_lot::RwLock;

use crate::helpers::error::{self, ERR_DTYPE, ERR_GENERIC, ERR_MATH, ERR_SHAPE, SUCCESS};
use crate::helpers::write_output_metadata;
use crate::helpers::{extract_view_c128, extract_view_c64, extract_view_f32, extract_view_f64};
use crate::types::{ArrayData, ArrayMetadata, DType, NDArrayWrapper, NdArrayHandle};

/// Compute SVD: A = U * S * V^T
///
/// # Arguments
/// * `a` - Input matrix handle (m x n, 2D)
/// * `a_meta` - Array metadata
/// * `calc_u` - If non-zero, compute U matrix
/// * `calc_vt` - If non-zero, compute V^T matrix
/// * `out_u` - Output U matrix handle (only written if calc_u is non-zero)
/// * `out_dtype_u` - U dtype output
/// * `out_ndim_u` - U ndim output
/// * `out_shape_u` - U shape output
/// * `max_ndim` - Maximum number of dimensions
/// * `out_s` - Output singular values vector handle
/// * `out_dtype_s` - S dtype output
/// * `out_ndim_s` - S ndim output
/// * `out_shape_s` - S shape output
/// * `out_vt` - Output V^T matrix handle (only written if calc_vt is non-zero)
/// * `out_dtype_vt` - VT dtype output
/// * `out_ndim_vt` - VT ndim output
/// * `out_shape_vt` - VT shape output
///
/// # Returns
/// * 0 on success, negative error code on failure
#[no_mangle]
pub unsafe extern "C" fn ndarray_svd(
    a: *const NdArrayHandle,
    a_meta: *const ArrayMetadata,
    calc_u: c_uchar,
    calc_vt: c_uchar,
    out_u: *mut *mut NdArrayHandle,
    out_dtype_u: *mut u8,
    out_ndim_u: *mut usize,
    out_shape_u: *mut usize,
    max_ndim: usize,
    out_s: *mut *mut NdArrayHandle,
    out_dtype_s: *mut u8,
    out_ndim_s: *mut usize,
    out_shape_s: *mut usize,
    out_vt: *mut *mut NdArrayHandle,
    out_dtype_vt: *mut u8,
    out_ndim_vt: *mut usize,
    out_shape_vt: *mut usize,
) -> i32 {
    if a.is_null() || a_meta.is_null() || out_s.is_null() {
        return ERR_GENERIC;
    }

    let calc_u_bool = calc_u != 0;
    let calc_vt_bool = calc_vt != 0;

    crate::ffi_guard!({
        let a_meta_ref = &*a_meta;
        let a_wrapper = NdArrayHandle::as_wrapper(a as *mut _);

        if a_meta_ref.ndim != 2 {
            error::set_last_error("SVD requires 2D matrix".to_string());
            return ERR_SHAPE;
        }

        let (u_wrapper_opt, s_wrapper, vt_wrapper_opt) = match a_wrapper.dtype {
            DType::Float64 => {
                let Some(a_view_dyn) = extract_view_f64(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract f64 view for SVD".to_string());
                    return ERR_GENERIC;
                };
                let a_view_2d = match a_view_dyn.into_dimensionality::<Ix2>() {
                    Ok(a) => a,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                let (u_opt, s, vt_opt) = match a_view_2d.svd(calc_u_bool, calc_vt_bool) {
                    Ok(triple) => triple,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_MATH;
                    }
                };

                let u_wrapper = u_opt.map(|u| NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(u.into_dyn()))),
                    dtype: DType::Float64,
                });

                let s_wrapper = NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(s.into_dyn()))),
                    dtype: DType::Float64,
                };

                let vt_wrapper = vt_opt.map(|vt| NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(vt.into_dyn()))),
                    dtype: DType::Float64,
                });

                (u_wrapper, s_wrapper, vt_wrapper)
            }
            DType::Float32 => {
                let Some(a_view_dyn) = extract_view_f32(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract f32 view for SVD".to_string());
                    return ERR_GENERIC;
                };
                let a_view_2d = match a_view_dyn.into_dimensionality::<Ix2>() {
                    Ok(a) => a,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                let (u_opt, s, vt_opt) = match a_view_2d.svd(calc_u_bool, calc_vt_bool) {
                    Ok(triple) => triple,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_MATH;
                    }
                };

                let u_wrapper = u_opt.map(|u| NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(u.into_dyn()))),
                    dtype: DType::Float32,
                });

                let s_wrapper = NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(s.into_dyn()))),
                    dtype: DType::Float32,
                };

                let vt_wrapper = vt_opt.map(|vt| NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(vt.into_dyn()))),
                    dtype: DType::Float32,
                });

                (u_wrapper, s_wrapper, vt_wrapper)
            }
            DType::Complex64 => {
                let Some(a_view_dyn) = extract_view_c64(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract c64 view for SVD".to_string());
                    return ERR_GENERIC;
                };
                let a_view_2d = match a_view_dyn.into_dimensionality::<Ix2>() {
                    Ok(a) => a,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                let (u_opt, s, vt_opt) = match a_view_2d.svd(calc_u_bool, calc_vt_bool) {
                    Ok(triple) => triple,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_MATH;
                    }
                };

                let u_wrapper = u_opt.map(|u| NDArrayWrapper {
                    data: ArrayData::Complex64(Arc::new(RwLock::new(u.into_dyn()))),
                    dtype: DType::Complex64,
                });

                let s_wrapper = NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(s.into_dyn()))),
                    dtype: DType::Float32,
                };

                let vt_wrapper = vt_opt.map(|vt| NDArrayWrapper {
                    data: ArrayData::Complex64(Arc::new(RwLock::new(vt.into_dyn()))),
                    dtype: DType::Complex64,
                });

                (u_wrapper, s_wrapper, vt_wrapper)
            }
            DType::Complex128 => {
                let Some(a_view_dyn) = extract_view_c128(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract c128 view for SVD".to_string());
                    return ERR_GENERIC;
                };
                let a_view_2d = match a_view_dyn.into_dimensionality::<Ix2>() {
                    Ok(a) => a,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                let (u_opt, s, vt_opt) = match a_view_2d.svd(calc_u_bool, calc_vt_bool) {
                    Ok(triple) => triple,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_MATH;
                    }
                };

                let u_wrapper = u_opt.map(|u| NDArrayWrapper {
                    data: ArrayData::Complex128(Arc::new(RwLock::new(u.into_dyn()))),
                    dtype: DType::Complex128,
                });

                let s_wrapper = NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(s.into_dyn()))),
                    dtype: DType::Float64,
                };

                let vt_wrapper = vt_opt.map(|vt| NDArrayWrapper {
                    data: ArrayData::Complex128(Arc::new(RwLock::new(vt.into_dyn()))),
                    dtype: DType::Complex128,
                });

                (u_wrapper, s_wrapper, vt_wrapper)
            }
            _ => {
                error::set_last_error(
                    "SVD only supports Float32, Float64, Complex64, and Complex128".to_string(),
                );
                return ERR_DTYPE;
            }
        };

        // Write S metadata
        if let Err(e) =
            write_output_metadata(&s_wrapper, out_dtype_s, out_ndim_s, out_shape_s, max_ndim)
        {
            error::set_last_error(e);
            return ERR_GENERIC;
        }
        *out_s = NdArrayHandle::from_wrapper(Box::new(s_wrapper));

        // Write U metadata if requested
        if calc_u_bool {
            if let Some(u_wrapper) = u_wrapper_opt {
                if let Err(e) = write_output_metadata(
                    &u_wrapper,
                    out_dtype_u,
                    out_ndim_u,
                    out_shape_u,
                    max_ndim,
                ) {
                    error::set_last_error(e);
                    return ERR_GENERIC;
                }
                *out_u = NdArrayHandle::from_wrapper(Box::new(u_wrapper));
            }
        }

        // Write VT metadata if requested
        if calc_vt_bool {
            if let Some(vt_wrapper) = vt_wrapper_opt {
                if let Err(e) = write_output_metadata(
                    &vt_wrapper,
                    out_dtype_vt,
                    out_ndim_vt,
                    out_shape_vt,
                    max_ndim,
                ) {
                    error::set_last_error(e);
                    return ERR_GENERIC;
                }
                *out_vt = NdArrayHandle::from_wrapper(Box::new(vt_wrapper));
            }
        }

        SUCCESS
    })
}
