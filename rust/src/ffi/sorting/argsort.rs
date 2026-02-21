//! Argsort along an axis.

use crate::core::view_helpers::{
    extract_view_bool, extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32,
    extract_view_i64, extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64,
    extract_view_u8,
};
use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::ffi::reductions::helpers::validate_axis;
use crate::ffi::sorting::helpers::{
    argsort_axis_generic, argsort_flat_generic, cmp_f32_asc_nan_last, cmp_f64_asc_nan_last,
    SortKind,
};
use crate::ffi::{write_output_metadata, NdArrayHandle, ViewMetadata};
use parking_lot::RwLock;
use std::sync::Arc;

/// Compute the argsort along an axis in the array.
#[no_mangle]
pub unsafe extern "C" fn ndarray_argsort_axis(
    handle: *const NdArrayHandle,
    meta: *const ViewMetadata,
    axis: i32,
    kind: i32,
    out_handle: *mut *mut NdArrayHandle,
    out_dtype: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if handle.is_null()
        || out_handle.is_null()
        || meta.is_null()
        || out_dtype.is_null()
        || out_ndim.is_null()
        || out_shape.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let meta = &*meta;

        let axis_usize = match validate_axis(meta.shape_slice(), axis) {
            Ok(a) => a,
            Err(e) => {
                crate::error::set_last_error(e);
                return ERR_SHAPE;
            }
        };

        let sort_kind = match SortKind::from_i32(kind) {
            Ok(k) => k,
            Err(e) => {
                crate::error::set_last_error(e);
                return ERR_GENERIC;
            }
        };

        let result = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, meta) else {
                    crate::error::set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_axis_generic(view, axis_usize, sort_kind, cmp_f64_asc_nan_last)
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, meta) else {
                    crate::error::set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_axis_generic(view, axis_usize, sort_kind, cmp_f32_asc_nan_last)
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, meta) else {
                    crate::error::set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_axis_generic(view, axis_usize, sort_kind, |a, b| a.cmp(b))
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, meta) else {
                    crate::error::set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_axis_generic(view, axis_usize, sort_kind, |a, b| a.cmp(b))
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, meta) else {
                    crate::error::set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_axis_generic(view, axis_usize, sort_kind, |a, b| a.cmp(b))
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, meta) else {
                    crate::error::set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_axis_generic(view, axis_usize, sort_kind, |a, b| a.cmp(b))
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, meta) else {
                    crate::error::set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_axis_generic(view, axis_usize, sort_kind, |a, b| a.cmp(b))
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, meta) else {
                    crate::error::set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_axis_generic(view, axis_usize, sort_kind, |a, b| a.cmp(b))
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, meta) else {
                    crate::error::set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_axis_generic(view, axis_usize, sort_kind, |a, b| a.cmp(b))
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, meta) else {
                    crate::error::set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_axis_generic(view, axis_usize, sort_kind, |a, b| a.cmp(b))
            }
            DType::Bool => {
                let Some(view) = extract_view_bool(wrapper, meta) else {
                    crate::error::set_last_error("Failed to extract bool view".to_string());
                    return ERR_GENERIC;
                };
                argsort_axis_generic(view, axis_usize, sort_kind, |a, b| a.cmp(b))
            }
        };

        let result_wrapper = NDArrayWrapper {
            data: ArrayData::Int64(Arc::new(RwLock::new(result))),
            dtype: DType::Int64,
        };

        if let Err(e) =
            write_output_metadata(&result_wrapper, out_dtype, out_ndim, out_shape, max_ndim)
        {
            crate::error::set_last_error(e);
            return ERR_GENERIC;
        }
        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}

/// Compute the argsort of the flattened array.
#[no_mangle]
pub unsafe extern "C" fn ndarray_argsort_flat(
    handle: *const NdArrayHandle,
    meta: *const ViewMetadata,
    kind: i32,
    out_handle: *mut *mut NdArrayHandle,
    out_dtype: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if handle.is_null()
        || out_handle.is_null()
        || meta.is_null()
        || out_dtype.is_null()
        || out_ndim.is_null()
        || out_shape.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let meta = &*meta;

        let sort_kind = match SortKind::from_i32(kind) {
            Ok(k) => k,
            Err(e) => {
                crate::error::set_last_error(e);
                return ERR_GENERIC;
            }
        };

        let result = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, meta) else {
                    crate::error::set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_flat_generic(view, sort_kind, cmp_f64_asc_nan_last)
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, meta) else {
                    crate::error::set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_flat_generic(view, sort_kind, cmp_f32_asc_nan_last)
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, meta) else {
                    crate::error::set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_flat_generic(view, sort_kind, |a, b| a.cmp(b))
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, meta) else {
                    crate::error::set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_flat_generic(view, sort_kind, |a, b| a.cmp(b))
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, meta) else {
                    crate::error::set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_flat_generic(view, sort_kind, |a, b| a.cmp(b))
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, meta) else {
                    crate::error::set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_flat_generic(view, sort_kind, |a, b| a.cmp(b))
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, meta) else {
                    crate::error::set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_flat_generic(view, sort_kind, |a, b| a.cmp(b))
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, meta) else {
                    crate::error::set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_flat_generic(view, sort_kind, |a, b| a.cmp(b))
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, meta) else {
                    crate::error::set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_flat_generic(view, sort_kind, |a, b| a.cmp(b))
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, meta) else {
                    crate::error::set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_flat_generic(view, sort_kind, |a, b| a.cmp(b))
            }
            DType::Bool => {
                let Some(view) = extract_view_bool(wrapper, meta) else {
                    crate::error::set_last_error("Failed to extract bool view".to_string());
                    return ERR_GENERIC;
                };
                argsort_flat_generic(view, sort_kind, |a, b| a.cmp(b))
            }
        };

        let result_wrapper = NDArrayWrapper {
            data: ArrayData::Int64(Arc::new(RwLock::new(result))),
            dtype: DType::Int64,
        };

        if let Err(e) =
            write_output_metadata(&result_wrapper, out_dtype, out_ndim, out_shape, max_ndim)
        {
            crate::error::set_last_error(e);
            return ERR_GENERIC;
        }
        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
