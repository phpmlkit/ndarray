//! Dot product with dtype promotion between operands.

use std::sync::Arc;

use ndarray::linalg::Dot;
use ndarray::{ArrayBase, ArrayD, Data, Ix0, Ix1, Ix2, IxDyn, LinalgScalar};
use parking_lot::RwLock;

use crate::helpers::error::{self, ERR_DTYPE, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::helpers::write_output_metadata;
use crate::helpers::{
    extract_view_as_c128, extract_view_as_c64, extract_view_as_f32, extract_view_as_f64,
    extract_view_c128, extract_view_c64, extract_view_f32, extract_view_f64,
    linalg_computation_dtype,
};
use crate::types::dtype::DType;
use crate::types::{ArrayData, ArrayMetadata, NDArrayWrapper, NdArrayHandle};

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

        let promoted = DType::promote(a_wrapper.dtype, b_wrapper.dtype);
        let Some(comp_dtype) = linalg_computation_dtype(promoted) else {
            error::set_last_error(
                "Dot product supports floating-point and complex dtypes".to_string(),
            );
            return ERR_DTYPE;
        };

        let same_native = a_wrapper.dtype == b_wrapper.dtype && a_wrapper.dtype == comp_dtype;

        let result_wrapper = match comp_dtype {
            DType::Float64 => {
                let result = if same_native {
                    let Some(a_v) = extract_view_f64(a_wrapper, a_meta_ref) else {
                        error::set_last_error(
                            "Failed to prepare Float64 operand a for dot".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    let Some(b_v) = extract_view_f64(b_wrapper, b_meta_ref) else {
                        error::set_last_error(
                            "Failed to prepare Float64 operand b for dot".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    match dot_dispatch(&a_v, &b_v) {
                        Ok(r) => r,
                        Err(e) => {
                            error::set_last_error(e);
                            return ERR_SHAPE;
                        }
                    }
                } else {
                    let Some(a_arr) = extract_view_as_f64(a_wrapper, a_meta_ref) else {
                        error::set_last_error(
                            "Failed to prepare Float64 operand a for dot".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    let Some(b_arr) = extract_view_as_f64(b_wrapper, b_meta_ref) else {
                        error::set_last_error(
                            "Failed to prepare Float64 operand b for dot".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    match dot_dispatch(&a_arr, &b_arr) {
                        Ok(r) => r,
                        Err(e) => {
                            error::set_last_error(e);
                            return ERR_SHAPE;
                        }
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let result = if same_native {
                    let Some(a_v) = extract_view_f32(a_wrapper, a_meta_ref) else {
                        error::set_last_error(
                            "Failed to prepare Float32 operand a for dot".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    let Some(b_v) = extract_view_f32(b_wrapper, b_meta_ref) else {
                        error::set_last_error(
                            "Failed to prepare Float32 operand b for dot".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    match dot_dispatch(&a_v, &b_v) {
                        Ok(r) => r,
                        Err(e) => {
                            error::set_last_error(e);
                            return ERR_SHAPE;
                        }
                    }
                } else {
                    let Some(a_arr) = extract_view_as_f32(a_wrapper, a_meta_ref) else {
                        error::set_last_error(
                            "Failed to prepare Float32 operand a for dot".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    let Some(b_arr) = extract_view_as_f32(b_wrapper, b_meta_ref) else {
                        error::set_last_error(
                            "Failed to prepare Float32 operand b for dot".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    match dot_dispatch(&a_arr, &b_arr) {
                        Ok(r) => r,
                        Err(e) => {
                            error::set_last_error(e);
                            return ERR_SHAPE;
                        }
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(result))),
                    dtype: DType::Float32,
                }
            }
            DType::Complex64 => {
                let result = if same_native {
                    let Some(a_v) = extract_view_c64(a_wrapper, a_meta_ref) else {
                        error::set_last_error(
                            "Failed to prepare Complex64 operand a for dot".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    let Some(b_v) = extract_view_c64(b_wrapper, b_meta_ref) else {
                        error::set_last_error(
                            "Failed to prepare Complex64 operand b for dot".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    match dot_dispatch(&a_v, &b_v) {
                        Ok(r) => r,
                        Err(e) => {
                            error::set_last_error(e);
                            return ERR_SHAPE;
                        }
                    }
                } else {
                    let Some(a_arr) = extract_view_as_c64(a_wrapper, a_meta_ref) else {
                        error::set_last_error(
                            "Failed to prepare Complex64 operand a for dot".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    let Some(b_arr) = extract_view_as_c64(b_wrapper, b_meta_ref) else {
                        error::set_last_error(
                            "Failed to prepare Complex64 operand b for dot".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    match dot_dispatch(&a_arr, &b_arr) {
                        Ok(r) => r,
                        Err(e) => {
                            error::set_last_error(e);
                            return ERR_SHAPE;
                        }
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Complex64(Arc::new(RwLock::new(result))),
                    dtype: DType::Complex64,
                }
            }
            DType::Complex128 => {
                let result = if same_native {
                    let Some(a_v) = extract_view_c128(a_wrapper, a_meta_ref) else {
                        error::set_last_error(
                            "Failed to prepare Complex128 operand a for dot".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    let Some(b_v) = extract_view_c128(b_wrapper, b_meta_ref) else {
                        error::set_last_error(
                            "Failed to prepare Complex128 operand b for dot".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    match dot_dispatch(&a_v, &b_v) {
                        Ok(r) => r,
                        Err(e) => {
                            error::set_last_error(e);
                            return ERR_SHAPE;
                        }
                    }
                } else {
                    let Some(a_arr) = extract_view_as_c128(a_wrapper, a_meta_ref) else {
                        error::set_last_error(
                            "Failed to prepare Complex128 operand a for dot".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    let Some(b_arr) = extract_view_as_c128(b_wrapper, b_meta_ref) else {
                        error::set_last_error(
                            "Failed to prepare Complex128 operand b for dot".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    match dot_dispatch(&a_arr, &b_arr) {
                        Ok(r) => r,
                        Err(e) => {
                            error::set_last_error(e);
                            return ERR_SHAPE;
                        }
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Complex128(Arc::new(RwLock::new(result))),
                    dtype: DType::Complex128,
                }
            }
            _ => {
                error::set_last_error("Dot product internal dtype error".to_string());
                return ERR_DTYPE;
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

/// Dispatch dot product (shape rules unchanged from legacy `dot`).
fn dot_dispatch<A, Sa, Sb>(
    a: &ArrayBase<Sa, IxDyn>,
    b: &ArrayBase<Sb, IxDyn>,
) -> Result<ArrayD<A>, String>
where
    A: LinalgScalar,
    Sa: Data<Elem = A>,
    Sb: Data<Elem = A>,
{
    let na = a.ndim();
    let nb = b.ndim();
    match (na, nb) {
        (1, 1) => {
            let a1 = a
                .view()
                .into_dimensionality::<Ix1>()
                .map_err(|e| e.to_string())?;
            let b1 = b
                .view()
                .into_dimensionality::<Ix1>()
                .map_err(|e| e.to_string())?;
            if a1.len() != b1.len() {
                return Err(format!("Shape mismatch: {} and {}", a1.len(), b1.len()));
            }
            Ok(ndarray::Array::<A, _>::from_elem(Ix0(), a1.dot(&b1)).into_dyn())
        }
        (1, 2) => {
            let a1 = a
                .view()
                .into_dimensionality::<Ix1>()
                .map_err(|e| e.to_string())?;
            let b2 = b
                .view()
                .into_dimensionality::<Ix2>()
                .map_err(|e| e.to_string())?;
            Ok(a1.dot(&b2).into_dimensionality::<IxDyn>().unwrap())
        }
        (2, 1) => {
            let a2 = a
                .view()
                .into_dimensionality::<Ix2>()
                .map_err(|e| e.to_string())?;
            let b1 = b
                .view()
                .into_dimensionality::<Ix1>()
                .map_err(|e| e.to_string())?;
            Ok(a2.dot(&b1).into_dimensionality::<IxDyn>().unwrap())
        }
        (2, 2) => {
            let a2 = a
                .view()
                .into_dimensionality::<Ix2>()
                .map_err(|e| e.to_string())?;
            let b2 = b
                .view()
                .into_dimensionality::<Ix2>()
                .map_err(|e| e.to_string())?;
            Ok(a2.dot(&b2).into_dimensionality::<IxDyn>().unwrap())
        }
        _ => Err(format!(
            "Dot product not supported for dimensions {}D @ {}D",
            na, nb
        )),
    }
}
