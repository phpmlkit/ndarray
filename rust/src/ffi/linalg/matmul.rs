//! Matrix multiplication.
//!
//! Only supports Float32 and Float64 types.

use std::sync::Arc;

use ndarray::linalg::Dot;
use ndarray::Ix2;
use parking_lot::RwLock;

use crate::helpers::error::{self, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::helpers::write_output_metadata;
use crate::helpers::{extract_view_c128, extract_view_c64, extract_view_f32, extract_view_f64};
use crate::types::{ArrayData, ArrayMetadata, DType, NDArrayWrapper, NdArrayHandle};

/// Matrix multiplication.
///
/// Only supports Float32 and Float64 types (2D arrays).
/// When BLAS is enabled, automatically uses BLAS gemm for f32/f64.
#[no_mangle]
pub unsafe extern "C" fn ndarray_matmul(
    a: *const NdArrayHandle,
    a_meta: *const ArrayMetadata,
    b: *const NdArrayHandle,
    b_meta: *const ArrayMetadata,
    out_handle: *mut *mut NdArrayHandle,
    out_dtype_ptr: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if a.is_null()
        || b.is_null()
        || out_handle.is_null()
        || out_dtype_ptr.is_null()
        || out_ndim.is_null()
        || out_shape.is_null()
        || a_meta.is_null()
        || b_meta.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let a_meta_ref = &*a_meta;
        let b_meta_ref = &*b_meta;
        let a_wrapper = NdArrayHandle::as_wrapper(a as *mut _);
        let b_wrapper = NdArrayHandle::as_wrapper(b as *mut _);

        if a_wrapper.dtype != b_wrapper.dtype {
            error::set_last_error(
                "Matrix multiplication requires both arrays to have the same dtype".to_string(),
            );
            return ERR_GENERIC;
        }

        if a_meta_ref.ndim < 2 || b_meta_ref.ndim < 2 {
            error::set_last_error("Matmul requires at least 2D arrays".to_string());
            return ERR_SHAPE;
        }

        let result_wrapper = match a_wrapper.dtype {
            DType::Float64 => {
                let Some(a_view) = extract_view_f64(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract f64 view for array a".to_string());
                    return ERR_GENERIC;
                };
                let Some(b_view) = extract_view_f64(b_wrapper, b_meta_ref) else {
                    error::set_last_error("Failed to extract f64 view for array b".to_string());
                    return ERR_GENERIC;
                };

                let a_view_2d = match a_view.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(format!("Failed to convert to 2D: {}", e));
                        return ERR_SHAPE;
                    }
                };
                let b_view_2d = match b_view.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(format!("Failed to convert to 2D: {}", e));
                        return ERR_SHAPE;
                    }
                };

                let result = a_view_2d.dot(&b_view_2d);

                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result.into_dyn()))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let Some(a_view) = extract_view_f32(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract f32 view for array a".to_string());
                    return ERR_GENERIC;
                };
                let Some(b_view) = extract_view_f32(b_wrapper, b_meta_ref) else {
                    error::set_last_error("Failed to extract f32 view for array b".to_string());
                    return ERR_GENERIC;
                };

                let a_view_2d = match a_view.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(format!("Failed to convert to 2D: {}", e));
                        return ERR_SHAPE;
                    }
                };
                let b_view_2d = match b_view.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(format!("Failed to convert to 2D: {}", e));
                        return ERR_SHAPE;
                    }
                };

                let result = a_view_2d.dot(&b_view_2d);

                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(result.into_dyn()))),
                    dtype: DType::Float32,
                }
            }
            DType::Complex64 => {
                let Some(a_view) = extract_view_c64(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract c64 view for array a".to_string());
                    return ERR_GENERIC;
                };
                let Some(b_view) = extract_view_c64(b_wrapper, b_meta_ref) else {
                    error::set_last_error("Failed to extract c64 view for array b".to_string());
                    return ERR_GENERIC;
                };

                let a_view_2d = match a_view.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(format!("Failed to convert to 2D: {}", e));
                        return ERR_SHAPE;
                    }
                };
                let b_view_2d = match b_view.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(format!("Failed to convert to 2D: {}", e));
                        return ERR_SHAPE;
                    }
                };

                let result = a_view_2d.dot(&b_view_2d);

                NDArrayWrapper {
                    data: ArrayData::Complex64(Arc::new(RwLock::new(result.into_dyn()))),
                    dtype: DType::Complex64,
                }
            }
            DType::Complex128 => {
                let Some(a_view) = extract_view_c128(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract c128 view for array a".to_string());
                    return ERR_GENERIC;
                };
                let Some(b_view) = extract_view_c128(b_wrapper, b_meta_ref) else {
                    error::set_last_error("Failed to extract c128 view for array b".to_string());
                    return ERR_GENERIC;
                };

                let a_view_2d = match a_view.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(format!("Failed to convert to 2D: {}", e));
                        return ERR_SHAPE;
                    }
                };
                let b_view_2d = match b_view.into_dimensionality::<Ix2>() {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(format!("Failed to convert to 2D: {}", e));
                        return ERR_SHAPE;
                    }
                };

                let result = a_view_2d.dot(&b_view_2d);

                NDArrayWrapper {
                    data: ArrayData::Complex128(Arc::new(RwLock::new(result.into_dyn()))),
                    dtype: DType::Complex128,
                }
            }
            _ => {
                error::set_last_error(
                    "Matrix multiplication only supports Float64, Float32, Complex64, and Complex128 types".to_string(),
                );
                return ERR_GENERIC;
            }
        };

        if let Err(e) = write_output_metadata(
            &result_wrapper,
            out_dtype_ptr,
            out_ndim,
            out_shape,
            max_ndim,
        ) {
            error::set_last_error(e);
            return ERR_GENERIC;
        }
        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
