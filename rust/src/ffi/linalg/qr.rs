//! QR Decomposition
//!
//! Decomposes matrix A into Q * R where Q is orthogonal and R is upper triangular.

use std::sync::Arc;

use ndarray::Ix2;
use ndarray_linalg::QR;
use parking_lot::RwLock;

use crate::helpers::error::{self, ERR_DTYPE, ERR_GENERIC, ERR_MATH, ERR_SHAPE, SUCCESS};
use crate::helpers::write_output_metadata;
use crate::helpers::{extract_view_c128, extract_view_c64, extract_view_f32, extract_view_f64};
use crate::types::{ArrayData, ArrayMetadata, DType, NDArrayWrapper, NdArrayHandle};

/// Compute QR decomposition: A = Q * R.
#[no_mangle]
pub unsafe extern "C" fn ndarray_qr(
    a: *const NdArrayHandle,
    a_meta: *const ArrayMetadata,
    out_q: *mut *mut NdArrayHandle,
    out_dtype_q: *mut u8,
    out_ndim_q: *mut usize,
    out_shape_q: *mut usize,
    max_ndim: usize,
    out_r: *mut *mut NdArrayHandle,
    out_dtype_r: *mut u8,
    out_ndim_r: *mut usize,
    out_shape_r: *mut usize,
) -> i32 {
    if a.is_null() || a_meta.is_null() || out_q.is_null() || out_r.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let a_meta_ref = &*a_meta;
        let a_wrapper = NdArrayHandle::as_wrapper(a as *mut _);

        if a_meta_ref.ndim != 2 {
            error::set_last_error("QR requires a 2D matrix".to_string());
            return ERR_SHAPE;
        }

        let (q_wrapper, r_wrapper) = match a_wrapper.dtype {
            DType::Float64 => {
                let Some(a_view_dyn) = extract_view_f64(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract f64 view for QR".to_string());
                    return ERR_GENERIC;
                };
                let a_view_2d = match a_view_dyn.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                let (q, r) = match a_view_2d.qr() {
                    Ok(pair) => pair,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_MATH;
                    }
                };
                let q_wrapper = NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(q.into_dyn()))),
                    dtype: DType::Float64,
                };
                let r_wrapper = NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(r.into_dyn()))),
                    dtype: DType::Float64,
                };
                (q_wrapper, r_wrapper)
            }
            DType::Float32 => {
                let Some(a_view_dyn) = extract_view_f32(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract f32 view for QR".to_string());
                    return ERR_GENERIC;
                };
                let a_view_2d = match a_view_dyn.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                let (q, r) = match a_view_2d.qr() {
                    Ok(pair) => pair,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_MATH;
                    }
                };
                let q_wrapper = NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(q.into_dyn()))),
                    dtype: DType::Float32,
                };
                let r_wrapper = NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(r.into_dyn()))),
                    dtype: DType::Float32,
                };
                (q_wrapper, r_wrapper)
            }
            DType::Complex64 => {
                let Some(a_view_dyn) = extract_view_c64(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract c64 view for QR".to_string());
                    return ERR_GENERIC;
                };
                let a_view_2d = match a_view_dyn.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                let (q, r) = match a_view_2d.qr() {
                    Ok(pair) => pair,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_MATH;
                    }
                };
                let q_wrapper = NDArrayWrapper {
                    data: ArrayData::Complex64(Arc::new(RwLock::new(q.into_dyn()))),
                    dtype: DType::Complex64,
                };
                let r_wrapper = NDArrayWrapper {
                    data: ArrayData::Complex64(Arc::new(RwLock::new(r.into_dyn()))),
                    dtype: DType::Complex64,
                };
                (q_wrapper, r_wrapper)
            }
            DType::Complex128 => {
                let Some(a_view_dyn) = extract_view_c128(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract c128 view for QR".to_string());
                    return ERR_GENERIC;
                };
                let a_view_2d = match a_view_dyn.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                let (q, r) = match a_view_2d.qr() {
                    Ok(pair) => pair,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_MATH;
                    }
                };
                let q_wrapper = NDArrayWrapper {
                    data: ArrayData::Complex128(Arc::new(RwLock::new(q.into_dyn()))),
                    dtype: DType::Complex128,
                };
                let r_wrapper = NDArrayWrapper {
                    data: ArrayData::Complex128(Arc::new(RwLock::new(r.into_dyn()))),
                    dtype: DType::Complex128,
                };
                (q_wrapper, r_wrapper)
            }
            _ => {
                error::set_last_error(
                    "QR only supports Float32, Float64, Complex64, and Complex128".to_string(),
                );
                return ERR_DTYPE;
            }
        };

        if let Err(e) =
            write_output_metadata(&q_wrapper, out_dtype_q, out_ndim_q, out_shape_q, max_ndim)
        {
            error::set_last_error(e);
            return ERR_GENERIC;
        }
        *out_q = NdArrayHandle::from_wrapper(Box::new(q_wrapper));

        if let Err(e) =
            write_output_metadata(&r_wrapper, out_dtype_r, out_ndim_r, out_shape_r, max_ndim)
        {
            error::set_last_error(e);
            return ERR_GENERIC;
        }
        *out_r = NdArrayHandle::from_wrapper(Box::new(r_wrapper));

        SUCCESS
    })
}
