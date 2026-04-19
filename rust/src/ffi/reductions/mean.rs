//! Mean reduction.

use std::ffi::c_void;

use crate::ffi::reductions::helpers::{
    compute_axis_output_shape, write_reduction_scalar, ReductionScalar,
};
use crate::helpers::error::{set_last_error, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::helpers::normalize_axis;
use crate::helpers::write_output_metadata;
use crate::helpers::{
    extract_view_c128, extract_view_c64, extract_view_f32, extract_view_f64, extract_view_i16,
    extract_view_i32, extract_view_i64, extract_view_i8, extract_view_u16, extract_view_u32,
    extract_view_u64, extract_view_u8,
};
use crate::types::dtype::DType;
use crate::types::{ArrayData, ArrayMetadata, NDArrayWrapper, NdArrayHandle};
use ndarray::Axis;
use ndarray::{ArrayD, IxDyn};
use num_complex::{Complex32, Complex64};
use parking_lot::RwLock;
use std::sync::Arc;

/// Compute the mean of all elements in the array.
///
/// Scalar output dtype matches the computation: `Float32` / `Complex64` / `Complex128` preserve
/// native precision; integer inputs promote to `Float64` (NumPy-style); other reals use `Float64`.
#[no_mangle]
pub unsafe extern "C" fn ndarray_mean(
    handle: *const NdArrayHandle,
    meta: *const ArrayMetadata,
    out_value: *mut c_void,
    out_dtype: *mut u8,
) -> i32 {
    if handle.is_null() || meta.is_null() || out_value.is_null() || out_dtype.is_null() {
        return ERR_GENERIC;
    }

    let meta = &*meta;

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);

        let scalar = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, meta) else {
                    set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                ReductionScalar::F64(view.mean().unwrap_or(0.0))
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, meta) else {
                    set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                ReductionScalar::F32(view.mean().unwrap_or(0.0))
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, meta) else {
                    set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                ReductionScalar::F64(view.mean().map(|x| x as f64).unwrap_or(0.0))
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, meta) else {
                    set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                ReductionScalar::F64(view.mean().map(|x| x as f64).unwrap_or(0.0))
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, meta) else {
                    set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                ReductionScalar::F64(view.mean().map(|x| x as f64).unwrap_or(0.0))
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, meta) else {
                    set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                ReductionScalar::F64(view.mean().map(|x| x as f64).unwrap_or(0.0))
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, meta) else {
                    set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                ReductionScalar::F64(view.mean().map(|x| x as f64).unwrap_or(0.0))
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, meta) else {
                    set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                ReductionScalar::F64(view.mean().map(|x| x as f64).unwrap_or(0.0))
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, meta) else {
                    set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                ReductionScalar::F64(view.mean().map(|x| x as f64).unwrap_or(0.0))
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, meta) else {
                    set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                ReductionScalar::F64(view.mean().map(|x| x as f64).unwrap_or(0.0))
            }
            DType::Complex64 => {
                let Some(view) = extract_view_c64(wrapper, meta) else {
                    set_last_error("Failed to extract Complex64 view".to_string());
                    return ERR_GENERIC;
                };
                let n = view.len() as f32;
                let sum = view.sum();
                let mean = sum / Complex32::new(n, 0.0);
                ReductionScalar::C64(mean)
            }
            DType::Complex128 => {
                let Some(view) = extract_view_c128(wrapper, meta) else {
                    set_last_error("Failed to extract Complex128 view".to_string());
                    return ERR_GENERIC;
                };
                let n = view.len() as f64;
                let sum = view.sum();
                let mean = sum / Complex64::new(n, 0.0);
                ReductionScalar::C128(mean)
            }
            DType::Bool => {
                set_last_error("mean() not supported for Bool type".to_string());
                return ERR_GENERIC;
            }
        };

        write_reduction_scalar(out_value, out_dtype, scalar);
        SUCCESS
    })
}

/// Compute the mean along an axis.
#[no_mangle]
pub unsafe extern "C" fn ndarray_mean_axis(
    handle: *const NdArrayHandle,
    meta: *const ArrayMetadata,
    axis: i32,
    keepdims: bool,
    out_handle: *mut *mut NdArrayHandle,
    out_dtype: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if handle.is_null()
        || out_handle.is_null()
        || out_dtype.is_null()
        || out_ndim.is_null()
        || out_shape.is_null()
    {
        return ERR_GENERIC;
    }

    let meta = &*meta;

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = meta.shape_slice();

        // Validate axis
        let axis_usize = match normalize_axis(shape_slice, axis, false) {
            Ok(a) => a,
            Err(e) => {
                set_last_error(e);
                return ERR_SHAPE;
            }
        };

        // Match on dtype, extract view, compute mean along axis, and create result wrapper
        // Mean always returns Float64
        let reduced: ArrayD<f64> = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, meta) else {
                    set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                view.mean_axis(Axis(axis_usize)).unwrap_or_else(|| {
                    let out_shape = compute_axis_output_shape(shape_slice, axis_usize, false);
                    ArrayD::zeros(IxDyn(&out_shape))
                })
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, meta) else {
                    set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                view.mapv(|x| x as f64)
                    .mean_axis(Axis(axis_usize))
                    .unwrap_or_else(|| {
                        let out_shape = compute_axis_output_shape(shape_slice, axis_usize, false);
                        ArrayD::zeros(IxDyn(&out_shape))
                    })
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, meta) else {
                    set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                view.mapv(|x| x as f64)
                    .mean_axis(Axis(axis_usize))
                    .unwrap_or_else(|| {
                        let out_shape = compute_axis_output_shape(shape_slice, axis_usize, false);
                        ArrayD::zeros(IxDyn(&out_shape))
                    })
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, meta) else {
                    set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                view.mapv(|x| x as f64)
                    .mean_axis(Axis(axis_usize))
                    .unwrap_or_else(|| {
                        let out_shape = compute_axis_output_shape(shape_slice, axis_usize, false);
                        ArrayD::zeros(IxDyn(&out_shape))
                    })
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, meta) else {
                    set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                view.mapv(|x| x as f64)
                    .mean_axis(Axis(axis_usize))
                    .unwrap_or_else(|| {
                        let out_shape = compute_axis_output_shape(shape_slice, axis_usize, false);
                        ArrayD::zeros(IxDyn(&out_shape))
                    })
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, meta) else {
                    set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                view.mapv(|x| x as f64)
                    .mean_axis(Axis(axis_usize))
                    .unwrap_or_else(|| {
                        let out_shape = compute_axis_output_shape(shape_slice, axis_usize, false);
                        ArrayD::zeros(IxDyn(&out_shape))
                    })
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, meta) else {
                    set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                view.mapv(|x| x as f64)
                    .mean_axis(Axis(axis_usize))
                    .unwrap_or_else(|| {
                        let out_shape = compute_axis_output_shape(shape_slice, axis_usize, false);
                        ArrayD::zeros(IxDyn(&out_shape))
                    })
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, meta) else {
                    set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                view.mapv(|x| x as f64)
                    .mean_axis(Axis(axis_usize))
                    .unwrap_or_else(|| {
                        let out_shape = compute_axis_output_shape(shape_slice, axis_usize, false);
                        ArrayD::zeros(IxDyn(&out_shape))
                    })
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, meta) else {
                    set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                view.mapv(|x| x as f64)
                    .mean_axis(Axis(axis_usize))
                    .unwrap_or_else(|| {
                        let out_shape = compute_axis_output_shape(shape_slice, axis_usize, false);
                        ArrayD::zeros(IxDyn(&out_shape))
                    })
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, meta) else {
                    set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                view.mapv(|x| x as f64)
                    .mean_axis(Axis(axis_usize))
                    .unwrap_or_else(|| {
                        let out_shape = compute_axis_output_shape(shape_slice, axis_usize, false);
                        ArrayD::zeros(IxDyn(&out_shape))
                    })
            }
            DType::Complex64 => {
                let Some(view) = extract_view_c64(wrapper, meta) else {
                    set_last_error("Failed to extract Complex64 view".to_string());
                    return ERR_GENERIC;
                };
                let count = shape_slice[axis_usize] as f32;
                let sum = view.sum_axis(Axis(axis_usize));
                let reduced = sum.mapv(|x| x / Complex32::new(count, 0.0));
                let final_arr = if keepdims {
                    reduced.insert_axis(Axis(axis_usize))
                } else {
                    reduced
                };
                let result_wrapper = NDArrayWrapper {
                    data: ArrayData::Complex64(Arc::new(RwLock::new(final_arr))),
                    dtype: DType::Complex64,
                };
                if let Err(e) =
                    write_output_metadata(&result_wrapper, out_dtype, out_ndim, out_shape, max_ndim)
                {
                    set_last_error(e);
                    return ERR_GENERIC;
                }
                *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
                return SUCCESS;
            }
            DType::Complex128 => {
                let Some(view) = extract_view_c128(wrapper, meta) else {
                    set_last_error("Failed to extract Complex128 view".to_string());
                    return ERR_GENERIC;
                };
                let count = shape_slice[axis_usize] as f64;
                let sum = view.sum_axis(Axis(axis_usize));
                let reduced = sum.mapv(|x| x / Complex64::new(count, 0.0));
                let final_arr = if keepdims {
                    reduced.insert_axis(Axis(axis_usize))
                } else {
                    reduced
                };
                let result_wrapper = NDArrayWrapper {
                    data: ArrayData::Complex128(Arc::new(RwLock::new(final_arr))),
                    dtype: DType::Complex128,
                };
                if let Err(e) =
                    write_output_metadata(&result_wrapper, out_dtype, out_ndim, out_shape, max_ndim)
                {
                    set_last_error(e);
                    return ERR_GENERIC;
                }
                *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
                return SUCCESS;
            }
            DType::Bool => {
                set_last_error("mean_axis() not supported for Bool type".to_string());
                return ERR_GENERIC;
            }
        };

        let final_arr = if keepdims {
            reduced.insert_axis(Axis(axis_usize))
        } else {
            reduced
        };

        let result_wrapper = NDArrayWrapper {
            data: ArrayData::Float64(Arc::new(RwLock::new(final_arr))),
            dtype: DType::Float64,
        };

        if let Err(e) =
            write_output_metadata(&result_wrapper, out_dtype, out_ndim, out_shape, max_ndim)
        {
            set_last_error(e);
            return ERR_GENERIC;
        }
        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
