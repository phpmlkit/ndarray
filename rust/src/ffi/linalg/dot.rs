//! Dot product operation.
//!
//! Only supports Float32 and Float64 types.

use std::sync::Arc;

use ndarray::linalg::Dot;
use ndarray::{ArrayD, ArrayViewD, Ix1, Ix2, IxDyn, LinalgScalar};
use parking_lot::RwLock;

use crate::helpers::error::{self, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::helpers::write_output_metadata;
use crate::helpers::{extract_view_c128, extract_view_c64, extract_view_f32, extract_view_f64};
use crate::types::{ArrayData, ArrayMetadata, DType, NDArrayWrapper, NdArrayHandle};

/// Compute dot product of two arrays.
#[no_mangle]
pub unsafe extern "C" fn ndarray_dot(
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
                "Dot product requires both arrays to have the same dtype".to_string(),
            );
            return ERR_GENERIC;
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

                let result = match dot_dispatch(a_view, b_view) {
                    Ok(r) => r,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };

                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
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

                let result = match dot_dispatch(a_view, b_view) {
                    Ok(r) => r,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };

                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(result))),
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

                let result = match dot_dispatch(a_view, b_view) {
                    Ok(r) => r,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };

                NDArrayWrapper {
                    data: ArrayData::Complex64(Arc::new(RwLock::new(result))),
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

                let result = match dot_dispatch(a_view, b_view) {
                    Ok(r) => r,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };

                NDArrayWrapper {
                    data: ArrayData::Complex128(Arc::new(RwLock::new(result))),
                    dtype: DType::Complex128,
                }
            }
            _ => {
                error::set_last_error(
                    "Dot product only supports Float64, Float32, Complex64, and Complex128 types"
                        .to_string(),
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

/// Dispatch dot product for dynamic views.
///
/// Returns `ArrayD<A>` where the result is 0D (scalar), 1D, or 2D depending on inputs.
fn dot_dispatch<A>(a: ArrayViewD<A>, b: ArrayViewD<A>) -> Result<ArrayD<A>, String>
where
    A: LinalgScalar,
{
    match (a.ndim(), b.ndim()) {
        (1, 1) => {
            let a1 = a.into_dimensionality::<Ix1>().map_err(|e| e.to_string())?;
            let b1 = b.into_dimensionality::<Ix1>().map_err(|e| e.to_string())?;
            if a1.len() != b1.len() {
                return Err(format!("Shape mismatch: {} and {}", a1.len(), b1.len()));
            }
            Ok(ArrayD::from_elem(vec![], a1.dot(&b1)))
        }
        (1, 2) => {
            let a1 = a.into_dimensionality::<Ix1>().map_err(|e| e.to_string())?;
            let b2 = b.into_dimensionality::<Ix2>().map_err(|e| e.to_string())?;
            Ok(a1.dot(&b2).into_dimensionality::<IxDyn>().unwrap())
        }
        (2, 1) => {
            let a2 = a.into_dimensionality::<Ix2>().map_err(|e| e.to_string())?;
            let b1 = b.into_dimensionality::<Ix1>().map_err(|e| e.to_string())?;
            Ok(a2.dot(&b1).into_dimensionality::<IxDyn>().unwrap())
        }
        (2, 2) => {
            let a2 = a.into_dimensionality::<Ix2>().map_err(|e| e.to_string())?;
            let b2 = b.into_dimensionality::<Ix2>().map_err(|e| e.to_string())?;
            Ok(a2.dot(&b2).into_dimensionality::<IxDyn>().unwrap())
        }
        _ => Err(format!(
            "Dot product not supported for dimensions {}D @ {}D",
            a.ndim(),
            b.ndim()
        )),
    }
}
