//! Dot product operation.
//!
//! Only supports Float32 and Float64 types.

use std::sync::Arc;

use ndarray::linalg::Dot;
use ndarray::{Ix1, Ix2};
use parking_lot::RwLock;

use crate::helpers::error::{self, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::helpers::write_output_metadata;
use crate::helpers::{extract_view_f32, extract_view_f64};
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

                match (a_meta_ref.ndim, b_meta_ref.ndim) {
                    (1, 1) => {
                        // Validate shape lengths match for 1D @ 1D
                        let a_shape = unsafe { a_meta_ref.shape_slice() };
                        let b_shape = unsafe { b_meta_ref.shape_slice() };
                        if a_shape[0] != b_shape[0] {
                            error::set_last_error(format!(
                                "Shape mismatch: {} and {}",
                                a_shape[0], b_shape[0]
                            ));
                            return ERR_SHAPE;
                        }

                        let a_arr = match a_view.into_dimensionality::<Ix1>() {
                            Ok(v) => v.into_owned(),
                            Err(e) => {
                                error::set_last_error(format!("Failed to convert: {}", e));
                                return ERR_SHAPE;
                            }
                        };
                        let b_arr = match b_view.into_dimensionality::<Ix1>() {
                            Ok(v) => v.into_owned(),
                            Err(e) => {
                                error::set_last_error(format!("Failed to convert: {}", e));
                                return ERR_SHAPE;
                            }
                        };
                        let result = a_arr.dot(&b_arr);
                        match NDArrayWrapper::from_slice_f64(&[result], &[]) {
                            Ok(w) => w,
                            Err(e) => {
                                error::set_last_error(e);
                                return ERR_GENERIC;
                            }
                        }
                    }
                    (1, 2) => {
                        let a_arr = match a_view.into_dimensionality::<Ix1>() {
                            Ok(v) => v.into_owned(),
                            Err(e) => {
                                error::set_last_error(format!("Failed to convert: {}", e));
                                return ERR_SHAPE;
                            }
                        };
                        let b_arr = match b_view.into_dimensionality::<Ix2>() {
                            Ok(v) => v.into_owned(),
                            Err(e) => {
                                error::set_last_error(format!("Failed to convert: {}", e));
                                return ERR_SHAPE;
                            }
                        };
                        let result = a_arr.dot(&b_arr);
                        NDArrayWrapper {
                            data: ArrayData::Float64(Arc::new(RwLock::new(result.into_dyn()))),
                            dtype: DType::Float64,
                        }
                    }
                    (2, 1) => {
                        let a_arr = match a_view.into_dimensionality::<Ix2>() {
                            Ok(v) => v.into_owned(),
                            Err(e) => {
                                error::set_last_error(format!("Failed to convert: {}", e));
                                return ERR_SHAPE;
                            }
                        };
                        let b_arr = match b_view.into_dimensionality::<Ix1>() {
                            Ok(v) => v.into_owned(),
                            Err(e) => {
                                error::set_last_error(format!("Failed to convert: {}", e));
                                return ERR_SHAPE;
                            }
                        };
                        let result = a_arr.dot(&b_arr);
                        NDArrayWrapper {
                            data: ArrayData::Float64(Arc::new(RwLock::new(result.into_dyn()))),
                            dtype: DType::Float64,
                        }
                    }
                    (2, 2) => {
                        let a_arr = match a_view.into_dimensionality::<Ix2>() {
                            Ok(v) => v.into_owned(),
                            Err(e) => {
                                error::set_last_error(format!("Failed to convert: {}", e));
                                return ERR_SHAPE;
                            }
                        };
                        let b_arr = match b_view.into_dimensionality::<Ix2>() {
                            Ok(v) => v.into_owned(),
                            Err(e) => {
                                error::set_last_error(format!("Failed to convert: {}", e));
                                return ERR_SHAPE;
                            }
                        };
                        let result = a_arr.dot(&b_arr);
                        NDArrayWrapper {
                            data: ArrayData::Float64(Arc::new(RwLock::new(result.into_dyn()))),
                            dtype: DType::Float64,
                        }
                    }
                    _ => {
                        error::set_last_error(format!(
                            "Dot product not supported for dimensions {}D @ {}D",
                            a_meta_ref.ndim, b_meta_ref.ndim
                        ));
                        return ERR_SHAPE;
                    }
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

                match (a_meta_ref.ndim, b_meta_ref.ndim) {
                    (1, 1) => {
                        // Validate shape lengths match for 1D @ 1D
                        let a_shape = unsafe { a_meta_ref.shape_slice() };
                        let b_shape = unsafe { b_meta_ref.shape_slice() };
                        if a_shape[0] != b_shape[0] {
                            error::set_last_error(format!(
                                "Shape mismatch: {} and {}",
                                a_shape[0], b_shape[0]
                            ));
                            return ERR_SHAPE;
                        }

                        let a_arr = match a_view.into_dimensionality::<Ix1>() {
                            Ok(v) => v.into_owned(),
                            Err(e) => {
                                error::set_last_error(format!("Failed to convert: {}", e));
                                return ERR_SHAPE;
                            }
                        };
                        let b_arr = match b_view.into_dimensionality::<Ix1>() {
                            Ok(v) => v.into_owned(),
                            Err(e) => {
                                error::set_last_error(format!("Failed to convert: {}", e));
                                return ERR_SHAPE;
                            }
                        };
                        let result = a_arr.dot(&b_arr);
                        match NDArrayWrapper::from_slice_f32(&[result], &[]) {
                            Ok(w) => w,
                            Err(e) => {
                                error::set_last_error(e);
                                return ERR_GENERIC;
                            }
                        }
                    }
                    (1, 2) => {
                        let a_arr = match a_view.into_dimensionality::<Ix1>() {
                            Ok(v) => v.into_owned(),
                            Err(e) => {
                                error::set_last_error(format!("Failed to convert: {}", e));
                                return ERR_SHAPE;
                            }
                        };
                        let b_arr = match b_view.into_dimensionality::<Ix2>() {
                            Ok(v) => v.into_owned(),
                            Err(e) => {
                                error::set_last_error(format!("Failed to convert: {}", e));
                                return ERR_SHAPE;
                            }
                        };
                        let result = a_arr.dot(&b_arr);
                        NDArrayWrapper {
                            data: ArrayData::Float32(Arc::new(RwLock::new(result.into_dyn()))),
                            dtype: DType::Float32,
                        }
                    }
                    (2, 1) => {
                        let a_arr = match a_view.into_dimensionality::<Ix2>() {
                            Ok(v) => v.into_owned(),
                            Err(e) => {
                                error::set_last_error(format!("Failed to convert: {}", e));
                                return ERR_SHAPE;
                            }
                        };
                        let b_arr = match b_view.into_dimensionality::<Ix1>() {
                            Ok(v) => v.into_owned(),
                            Err(e) => {
                                error::set_last_error(format!("Failed to convert: {}", e));
                                return ERR_SHAPE;
                            }
                        };
                        let result = a_arr.dot(&b_arr);
                        NDArrayWrapper {
                            data: ArrayData::Float32(Arc::new(RwLock::new(result.into_dyn()))),
                            dtype: DType::Float32,
                        }
                    }
                    (2, 2) => {
                        let a_arr = match a_view.into_dimensionality::<Ix2>() {
                            Ok(v) => v.into_owned(),
                            Err(e) => {
                                error::set_last_error(format!("Failed to convert: {}", e));
                                return ERR_SHAPE;
                            }
                        };
                        let b_arr = match b_view.into_dimensionality::<Ix2>() {
                            Ok(v) => v.into_owned(),
                            Err(e) => {
                                error::set_last_error(format!("Failed to convert: {}", e));
                                return ERR_SHAPE;
                            }
                        };
                        let result = a_arr.dot(&b_arr);
                        NDArrayWrapper {
                            data: ArrayData::Float32(Arc::new(RwLock::new(result.into_dyn()))),
                            dtype: DType::Float32,
                        }
                    }
                    _ => {
                        error::set_last_error(format!(
                            "Dot product not supported for dimensions {}D @ {}D",
                            a_meta_ref.ndim, b_meta_ref.ndim
                        ));
                        return ERR_SHAPE;
                    }
                }
            }
            _ => {
                error::set_last_error(
                    "Dot product only supports Float64 and Float32 types".to_string(),
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
