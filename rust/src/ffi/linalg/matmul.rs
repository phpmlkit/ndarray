//! Matrix multiplication (`@`), NumPy `matmul` rules for 1D/2D and dtype promotion.

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

/// Matrix multiply following NumPy `matmul` for 1D and 2D only (no stacks).
///
/// Works on any [`ArrayBase`] with dynamic dimensions (owned arrays or views).
///
/// - `(m,n) @ (n,k)` → `(m,k)`
/// - `(m,n) @ (n,)` → `(m,)`
/// - `(n,) @ (n,k)` → `(k,)`
/// - `(n,) @ (n,)` → scalar (`0`-D array)
fn matmul_nd<A, Sa, Sb>(
    a: &ArrayBase<Sa, IxDyn>,
    b: &ArrayBase<Sb, IxDyn>,
) -> Result<ArrayD<A>, String>
where
    A: LinalgScalar + Clone,
    Sa: Data<Elem = A>,
    Sb: Data<Elem = A>,
{
    let na = a.ndim();
    let nb = b.ndim();
    if na == 0 || nb == 0 || na > 2 || nb > 2 {
        return Err(format!(
            "matmul supports 1D and 2D arrays only (got {}D @ {}D)",
            na, nb
        ));
    }

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
                return Err(format!(
                    "matmul: inner dimensions mismatch ({} and {})",
                    a1.len(),
                    b1.len()
                ));
            }
            let s = a1.dot(&b1);
            Ok(ndarray::Array::<A, _>::from_elem(Ix0(), s).into_dyn())
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
            let n = a2.shape()[1];
            if n != b2.shape()[0] {
                return Err(format!(
                    "matmul: inner dimensions mismatch ({} and {})",
                    n,
                    b2.shape()[0]
                ));
            }
            Ok(a2.dot(&b2).into_dyn())
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
            if a2.shape()[1] != b1.len() {
                return Err(format!(
                    "matmul: inner dimensions mismatch ({} and {})",
                    a2.shape()[1],
                    b1.len()
                ));
            }
            Ok(a2.dot(&b1).into_dyn())
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
            if a1.len() != b2.shape()[0] {
                return Err(format!(
                    "matmul: inner dimensions mismatch ({} and {})",
                    a1.len(),
                    b2.shape()[0]
                ));
            }
            Ok(a1.dot(&b2).into_dyn())
        }
        _ => Err(format!("matmul: unsupported dimensions {}D @ {}D", na, nb)),
    }
}

/// Matrix multiplication with NumPy-style 1D/2D handling and dtype promotion.
/// When BLAS is enabled, automatically uses BLAS gemm.
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

        let promoted = DType::promote(a_wrapper.dtype, b_wrapper.dtype);
        let Some(comp_dtype) = linalg_computation_dtype(promoted) else {
            error::set_last_error("Matmul supports floating-point and complex dtypes".to_string());
            return ERR_DTYPE;
        };

        let same_native = a_wrapper.dtype == b_wrapper.dtype && a_wrapper.dtype == comp_dtype;

        let result_wrapper = match comp_dtype {
            DType::Float64 => {
                let result = if same_native {
                    let Some(a_v) = extract_view_f64(a_wrapper, a_meta_ref) else {
                        error::set_last_error(
                            "Failed to prepare Float64 operand a for matmul".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    let Some(b_v) = extract_view_f64(b_wrapper, b_meta_ref) else {
                        error::set_last_error(
                            "Failed to prepare Float64 operand b for matmul".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    match matmul_nd(&a_v, &b_v) {
                        Ok(r) => r,
                        Err(e) => {
                            error::set_last_error(e);
                            return ERR_SHAPE;
                        }
                    }
                } else {
                    let Some(a_arr) = extract_view_as_f64(a_wrapper, a_meta_ref) else {
                        error::set_last_error(
                            "Failed to prepare Float64 operand a for matmul".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    let Some(b_arr) = extract_view_as_f64(b_wrapper, b_meta_ref) else {
                        error::set_last_error(
                            "Failed to prepare Float64 operand b for matmul".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    match matmul_nd(&a_arr, &b_arr) {
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
                            "Failed to prepare Float32 operand a for matmul".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    let Some(b_v) = extract_view_f32(b_wrapper, b_meta_ref) else {
                        error::set_last_error(
                            "Failed to prepare Float32 operand b for matmul".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    match matmul_nd(&a_v, &b_v) {
                        Ok(r) => r,
                        Err(e) => {
                            error::set_last_error(e);
                            return ERR_SHAPE;
                        }
                    }
                } else {
                    let Some(a_arr) = extract_view_as_f32(a_wrapper, a_meta_ref) else {
                        error::set_last_error(
                            "Failed to prepare Float32 operand a for matmul".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    let Some(b_arr) = extract_view_as_f32(b_wrapper, b_meta_ref) else {
                        error::set_last_error(
                            "Failed to prepare Float32 operand b for matmul".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    match matmul_nd(&a_arr, &b_arr) {
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
                            "Failed to prepare Complex64 operand a for matmul".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    let Some(b_v) = extract_view_c64(b_wrapper, b_meta_ref) else {
                        error::set_last_error(
                            "Failed to prepare Complex64 operand b for matmul".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    match matmul_nd(&a_v, &b_v) {
                        Ok(r) => r,
                        Err(e) => {
                            error::set_last_error(e);
                            return ERR_SHAPE;
                        }
                    }
                } else {
                    let Some(a_arr) = extract_view_as_c64(a_wrapper, a_meta_ref) else {
                        error::set_last_error(
                            "Failed to prepare Complex64 operand a for matmul".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    let Some(b_arr) = extract_view_as_c64(b_wrapper, b_meta_ref) else {
                        error::set_last_error(
                            "Failed to prepare Complex64 operand b for matmul".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    match matmul_nd(&a_arr, &b_arr) {
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
                            "Failed to prepare Complex128 operand a for matmul".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    let Some(b_v) = extract_view_c128(b_wrapper, b_meta_ref) else {
                        error::set_last_error(
                            "Failed to prepare Complex128 operand b for matmul".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    match matmul_nd(&a_v, &b_v) {
                        Ok(r) => r,
                        Err(e) => {
                            error::set_last_error(e);
                            return ERR_SHAPE;
                        }
                    }
                } else {
                    let Some(a_arr) = extract_view_as_c128(a_wrapper, a_meta_ref) else {
                        error::set_last_error(
                            "Failed to prepare Complex128 operand a for matmul".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    let Some(b_arr) = extract_view_as_c128(b_wrapper, b_meta_ref) else {
                        error::set_last_error(
                            "Failed to prepare Complex128 operand b for matmul".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    match matmul_nd(&a_arr, &b_arr) {
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
                error::set_last_error("Matmul internal dtype error".to_string());
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
