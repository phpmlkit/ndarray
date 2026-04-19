//! Solving Linear Systems
//!
//! Solve A * x = b for x using LU decomposition.

use std::sync::Arc;

use ndarray::{ArrayD, ArrayViewD, Axis, Ix1, Ix2};
use ndarray_linalg::{Lapack, Scalar, Solve};
use parking_lot::RwLock;

use crate::helpers::error::{self, ERR_DTYPE, ERR_GENERIC, ERR_MATH, ERR_SHAPE, SUCCESS};
use crate::helpers::write_output_metadata;
use crate::helpers::{extract_view_c128, extract_view_c64, extract_view_f32, extract_view_f64};
use crate::types::{ArrayData, ArrayMetadata, DType, NDArrayWrapper, NdArrayHandle};

enum SolveErr {
    Shape(String),
    Math(String),
    Generic(String),
}

/// Dispatch solve for generic float type.
fn solve_dispatch<A>(
    a: ArrayViewD<A>,
    b: ArrayViewD<A>,
    b_ndim: usize,
) -> Result<ArrayD<A>, SolveErr>
where
    A: Lapack + Scalar + Clone,
{
    let a_2d = a
        .into_dimensionality::<Ix2>()
        .map_err(|e| SolveErr::Shape(format!("Solve: failed to convert A to 2D: {}", e)))?;

    match b_ndim {
        1 => {
            let b_1d = b
                .into_dimensionality::<Ix1>()
                .map_err(|e| SolveErr::Shape(format!("Solve: failed to convert b to 1D: {}", e)))?;
            a_2d.solve(&b_1d)
                .map(|r| r.into_dyn())
                .map_err(|e| SolveErr::Math(format!("Solve computation failed: {:?}", e)))
        }
        2 => {
            let b_2d = b
                .into_dimensionality::<Ix2>()
                .map_err(|e| SolveErr::Shape(format!("Solve: failed to convert b to 2D: {}", e)))?;
            let mut cols: Vec<ndarray::Array1<A>> = Vec::with_capacity(b_2d.ncols());
            for col in b_2d.axis_iter(Axis(1)) {
                let x = a_2d
                    .solve(&col)
                    .map_err(|e| SolveErr::Math(format!("Solve computation failed: {:?}", e)))?;
                cols.push(x);
            }
            let stacked =
                ndarray::stack(Axis(1), &cols.iter().map(|c| c.view()).collect::<Vec<_>>())
                    .map_err(|e| {
                        SolveErr::Generic(format!("Solve failed to stack results: {}", e))
                    })?;
            Ok(stacked.as_standard_layout().to_owned().into_dyn())
        }
        _ => Err(SolveErr::Shape(
            "Solve requires b to be 1D or 2D".to_string(),
        )),
    }
}

/// Solve a linear system A * x = b.
#[no_mangle]
pub unsafe extern "C" fn ndarray_solve(
    a: *const NdArrayHandle,
    a_meta: *const ArrayMetadata,
    b: *const NdArrayHandle,
    b_meta: *const ArrayMetadata,
    out_handle: *mut *mut NdArrayHandle,
    out_dtype: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if a.is_null() || a_meta.is_null() || b.is_null() || b_meta.is_null() || out_handle.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let a_meta_ref = &*a_meta;
        let b_meta_ref = &*b_meta;
        let a_wrapper = NdArrayHandle::as_wrapper(a as *mut _);
        let b_wrapper = NdArrayHandle::as_wrapper(b as *mut _);

        if a_wrapper.dtype != b_wrapper.dtype {
            error::set_last_error("Solve requires both arrays to have the same dtype".to_string());
            return ERR_DTYPE;
        }

        if a_meta_ref.ndim != 2 {
            error::set_last_error("Solve requires coefficient matrix A to be 2D".to_string());
            return ERR_SHAPE;
        }

        let result_wrapper = match a_wrapper.dtype {
            DType::Float64 => {
                let Some(a_view) = extract_view_f64(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract f64 view for A".to_string());
                    return ERR_GENERIC;
                };
                let Some(b_view) = extract_view_f64(b_wrapper, b_meta_ref) else {
                    error::set_last_error("Failed to extract f64 view for b".to_string());
                    return ERR_GENERIC;
                };
                let result = match solve_dispatch(a_view, b_view, b_meta_ref.ndim) {
                    Ok(r) => r,
                    Err(SolveErr::Shape(e)) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                    Err(SolveErr::Math(e)) => {
                        error::set_last_error(e);
                        return ERR_MATH;
                    }
                    Err(SolveErr::Generic(e)) => {
                        error::set_last_error(e);
                        return ERR_GENERIC;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let Some(a_view) = extract_view_f32(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract f32 view for A".to_string());
                    return ERR_GENERIC;
                };
                let Some(b_view) = extract_view_f32(b_wrapper, b_meta_ref) else {
                    error::set_last_error("Failed to extract f32 view for b".to_string());
                    return ERR_GENERIC;
                };
                let result = match solve_dispatch(a_view, b_view, b_meta_ref.ndim) {
                    Ok(r) => r,
                    Err(SolveErr::Shape(e)) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                    Err(SolveErr::Math(e)) => {
                        error::set_last_error(e);
                        return ERR_MATH;
                    }
                    Err(SolveErr::Generic(e)) => {
                        error::set_last_error(e);
                        return ERR_GENERIC;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(result))),
                    dtype: DType::Float32,
                }
            }
            DType::Complex64 => {
                let Some(a_view) = extract_view_c64(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract c64 view for A".to_string());
                    return ERR_GENERIC;
                };
                let Some(b_view) = extract_view_c64(b_wrapper, b_meta_ref) else {
                    error::set_last_error("Failed to extract c64 view for b".to_string());
                    return ERR_GENERIC;
                };
                let result = match solve_dispatch(a_view, b_view, b_meta_ref.ndim) {
                    Ok(r) => r,
                    Err(SolveErr::Shape(e)) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                    Err(SolveErr::Math(e)) => {
                        error::set_last_error(e);
                        return ERR_MATH;
                    }
                    Err(SolveErr::Generic(e)) => {
                        error::set_last_error(e);
                        return ERR_GENERIC;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Complex64(Arc::new(RwLock::new(result))),
                    dtype: DType::Complex64,
                }
            }
            DType::Complex128 => {
                let Some(a_view) = extract_view_c128(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract c128 view for A".to_string());
                    return ERR_GENERIC;
                };
                let Some(b_view) = extract_view_c128(b_wrapper, b_meta_ref) else {
                    error::set_last_error("Failed to extract c128 view for b".to_string());
                    return ERR_GENERIC;
                };
                let result = match solve_dispatch(a_view, b_view, b_meta_ref.ndim) {
                    Ok(r) => r,
                    Err(SolveErr::Shape(e)) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                    Err(SolveErr::Math(e)) => {
                        error::set_last_error(e);
                        return ERR_MATH;
                    }
                    Err(SolveErr::Generic(e)) => {
                        error::set_last_error(e);
                        return ERR_GENERIC;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Complex128(Arc::new(RwLock::new(result))),
                    dtype: DType::Complex128,
                }
            }
            _ => {
                error::set_last_error(
                    "Solve only supports Float32, Float64, Complex64, and Complex128".to_string(),
                );
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
