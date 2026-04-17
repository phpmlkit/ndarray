//! Least Squares
//!
//! Solve the least squares problem min ||Ax - b||_2.

use std::sync::Arc;

use ndarray::{Array1, ArrayD, Ix1, Ix2};
use ndarray_linalg::LeastSquaresSvd;
use parking_lot::RwLock;

use crate::helpers::error::{self, ERR_DTYPE, ERR_GENERIC, ERR_MATH, ERR_SHAPE, SUCCESS};
use crate::helpers::write_output_metadata;
use crate::helpers::{extract_view_f32, extract_view_f64};
use crate::types::{ArrayData, ArrayMetadata, DType, NDArrayWrapper, NdArrayHandle};

enum LstsqErr {
    Shape(String),
    Math(String),
}

fn lstsq_dispatch<A>(
    a: ndarray::ArrayViewD<A>,
    b: ndarray::ArrayViewD<A>,
    b_ndim: usize,
) -> Result<(ArrayD<A>, Option<ArrayD<A::Real>>, i32, Array1<A::Real>), LstsqErr>
where
    A: ndarray_linalg::Scalar + ndarray_linalg::Lapack,
{
    let a_2d = a
        .into_dimensionality::<Ix2>()
        .map_err(|e| LstsqErr::Shape(format!("Lstsq: failed to convert A to 2D: {}", e)))?;

    match b_ndim {
        1 => {
            let b_1d = b
                .into_dimensionality::<Ix1>()
                .map_err(|e| LstsqErr::Shape(format!("Lstsq: failed to convert b to 1D: {}", e)))?;
            let result = a_2d
                .least_squares(&b_1d)
                .map_err(|e| LstsqErr::Math(format!("Lstsq computation failed: {:?}", e)))?;
            let residuals = result.residual_sum_of_squares.map(|r| r.into_dyn());
            Ok((
                result.solution.into_dyn(),
                residuals,
                result.rank,
                result.singular_values,
            ))
        }
        2 => {
            let b_2d = b
                .into_dimensionality::<Ix2>()
                .map_err(|e| LstsqErr::Shape(format!("Lstsq: failed to convert b to 2D: {}", e)))?;
            let result = a_2d
                .least_squares(&b_2d)
                .map_err(|e| LstsqErr::Math(format!("Lstsq computation failed: {:?}", e)))?;
            let residuals = result.residual_sum_of_squares.map(|r| r.into_dyn());
            Ok((
                result.solution.into_dyn(),
                residuals,
                result.rank,
                result.singular_values,
            ))
        }
        _ => Err(LstsqErr::Shape(
            "Lstsq requires b to be 1D or 2D".to_string(),
        )),
    }
}

/// Solve a least-squares problem.
#[no_mangle]
pub unsafe extern "C" fn ndarray_lstsq(
    a: *const NdArrayHandle,
    a_meta: *const ArrayMetadata,
    b: *const NdArrayHandle,
    b_meta: *const ArrayMetadata,
    out_solution: *mut *mut NdArrayHandle,
    out_residuals: *mut *mut NdArrayHandle,
    out_rank: *mut i32,
    out_s: *mut *mut NdArrayHandle,
    out_dtype_sol: *mut u8,
    out_ndim_sol: *mut usize,
    out_shape_sol: *mut usize,
    out_dtype_res: *mut u8,
    out_ndim_res: *mut usize,
    out_shape_res: *mut usize,
    out_dtype_s: *mut u8,
    out_ndim_s: *mut usize,
    out_shape_s: *mut usize,
    max_ndim: usize,
) -> i32 {
    if a.is_null()
        || a_meta.is_null()
        || b.is_null()
        || b_meta.is_null()
        || out_solution.is_null()
        || out_residuals.is_null()
        || out_rank.is_null()
        || out_s.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let a_meta_ref = &*a_meta;
        let b_meta_ref = &*b_meta;
        let a_wrapper = NdArrayHandle::as_wrapper(a as *mut _);
        let b_wrapper = NdArrayHandle::as_wrapper(b as *mut _);

        if a_wrapper.dtype != b_wrapper.dtype {
            error::set_last_error("Lstsq requires both arrays to have the same dtype".to_string());
            return ERR_DTYPE;
        }

        let (solution_wrapper, residuals_wrapper, rank, s_wrapper) = match a_wrapper.dtype {
            DType::Float64 => {
                let Some(a_view) = extract_view_f64(a_wrapper, a_meta_ref) else {
                    error::set_last_error("Failed to extract f64 view for A".to_string());
                    return ERR_GENERIC;
                };
                let Some(b_view) = extract_view_f64(b_wrapper, b_meta_ref) else {
                    error::set_last_error("Failed to extract f64 view for b".to_string());
                    return ERR_GENERIC;
                };
                let (sol, res, rank, s) = match lstsq_dispatch(a_view, b_view, b_meta_ref.ndim) {
                    Ok(r) => r,
                    Err(LstsqErr::Shape(e)) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                    Err(LstsqErr::Math(e)) => {
                        error::set_last_error(e);
                        return ERR_MATH;
                    }
                };
                let sol_wrapper = NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(sol))),
                    dtype: DType::Float64,
                };
                let res_wrapper = res.map(|r| NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(r))),
                    dtype: DType::Float64,
                });
                let s_wrapper = NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(s.into_dyn()))),
                    dtype: DType::Float64,
                };
                (sol_wrapper, res_wrapper, rank, s_wrapper)
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
                let (sol, res, rank, s) = match lstsq_dispatch(a_view, b_view, b_meta_ref.ndim) {
                    Ok(r) => r,
                    Err(LstsqErr::Shape(e)) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                    Err(LstsqErr::Math(e)) => {
                        error::set_last_error(e);
                        return ERR_MATH;
                    }
                };
                let sol_wrapper = NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(sol))),
                    dtype: DType::Float32,
                };
                let res_wrapper = res.map(|r| NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(r))),
                    dtype: DType::Float32,
                });
                let s_wrapper = NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(s.into_dyn()))),
                    dtype: DType::Float32,
                };
                (sol_wrapper, res_wrapper, rank, s_wrapper)
            }
            _ => {
                error::set_last_error("Lstsq only supports Float32 and Float64".to_string());
                return ERR_DTYPE;
            }
        };

        if let Err(e) = write_output_metadata(
            &solution_wrapper,
            out_dtype_sol,
            out_ndim_sol,
            out_shape_sol,
            max_ndim,
        ) {
            error::set_last_error(e);
            return ERR_GENERIC;
        }
        *out_solution = NdArrayHandle::from_wrapper(Box::new(solution_wrapper));

        if let Some(res) = residuals_wrapper {
            if let Err(e) =
                write_output_metadata(&res, out_dtype_res, out_ndim_res, out_shape_res, max_ndim)
            {
                error::set_last_error(e);
                return ERR_GENERIC;
            }
            *out_residuals = NdArrayHandle::from_wrapper(Box::new(res));
        } else {
            *out_residuals = std::ptr::null_mut();
        }

        *out_rank = rank;

        if let Err(e) =
            write_output_metadata(&s_wrapper, out_dtype_s, out_ndim_s, out_shape_s, max_ndim)
        {
            error::set_last_error(e);
            return ERR_GENERIC;
        }
        *out_s = NdArrayHandle::from_wrapper(Box::new(s_wrapper));

        SUCCESS
    })
}
