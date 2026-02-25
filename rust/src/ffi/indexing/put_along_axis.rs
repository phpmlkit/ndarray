//! Put values along an axis.

use crate::core::view_helpers::{
    extract_view_bool, extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32,
    extract_view_i64, extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64,
    extract_view_u8,
};
use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{self, ERR_DTYPE, ERR_GENERIC, ERR_INDEX, ERR_SHAPE, SUCCESS};
use crate::ffi::indexing::utils::normalize_index;
use crate::ffi::reductions::helpers::validate_axis;
use crate::ffi::{NdArrayHandle, ViewMetadata};
use ndarray::{ArrayD, Dimension, IxDyn};
use parking_lot::RwLock;
use std::ffi::c_void;
use std::sync::Arc;

fn put_along_axis_impl<T: Copy>(
    view: ndarray::ArrayViewD<'_, T>,
    indices: ndarray::ArrayViewD<'_, i64>,
    axis: usize,
    values: &[T],
    scalar: Option<T>,
) -> Result<ArrayD<T>, String> {
    let in_shape = view.shape();
    let idx_shape = indices.shape();
    if in_shape.len() != idx_shape.len() {
        return Err(format!(
            "indices.ndim ({}) must match input.ndim ({})",
            idx_shape.len(),
            in_shape.len()
        ));
    }
    for dim in 0..in_shape.len() {
        if dim != axis && in_shape[dim] != idx_shape[dim] {
            return Err(format!(
                "Shape mismatch at axis {}: input {} vs indices {}",
                dim, in_shape[dim], idx_shape[dim]
            ));
        }
    }
    if values.is_empty() && scalar.is_none() {
        return Err("putAlongAxis requires scalar or non-empty values".to_string());
    }

    let mut out = view.to_owned();
    for (k, (idx_pos, idx_val)) in indices.indexed_iter().enumerate() {
        let mut dst_pos = idx_pos.slice().to_vec();
        dst_pos[axis] = normalize_index(*idx_val, in_shape[axis])?;
        let v = match scalar {
            Some(s) => s,
            None => values[k % values.len()],
        };
        let dst = out
            .get_mut(IxDyn(&dst_pos))
            .ok_or_else(|| "Failed to write destination in putAlongAxis".to_string())?;
        *dst = v;
    }
    Ok(out)
}

/// Scatter values along an axis and return a mutated copy.
#[no_mangle]
pub unsafe extern "C" fn ndarray_put_along_axis(
    handle: *const NdArrayHandle,
    meta: *const ViewMetadata,
    indices_handle: *const NdArrayHandle,
    indices_meta: *const ViewMetadata,
    axis: i32,
    values: *const c_void,
    values_len: usize,
    scalar_value: f64,
    has_scalar: bool,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null()
        || meta.is_null()
        || indices_handle.is_null()
        || indices_meta.is_null()
        || out_handle.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let indices_wrapper = NdArrayHandle::as_wrapper(indices_handle as *mut _);
        let meta_ref = &*meta;
        let indices_meta_ref = &*indices_meta;

        if indices_wrapper.dtype != DType::Int64 {
            error::set_last_error("putAlongAxis indices must have Int64 dtype".to_string());
            return ERR_DTYPE;
        }
        let axis_usize = match validate_axis(indices_meta_ref.shape_slice(), axis) {
            Ok(v) => v,
            Err(e) => {
                error::set_last_error(e);
                return ERR_SHAPE;
            }
        };
        let Some(indices_view) = extract_view_i64(indices_wrapper, indices_meta_ref) else {
            error::set_last_error("Failed to extract Int64 indices view".to_string());
            return ERR_GENERIC;
        };

        let result_wrapper = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, meta_ref) else {
                    error::set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                let vals: &[f64] = if values.is_null() || values_len == 0 {
                    &[]
                } else {
                    std::slice::from_raw_parts(values as *const f64, values_len)
                };
                let out = match put_along_axis_impl(
                    view,
                    indices_view,
                    axis_usize,
                    vals,
                    has_scalar.then_some(scalar_value as f64),
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_INDEX;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(out))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, meta_ref) else {
                    error::set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                let vals: &[f32] = if values.is_null() || values_len == 0 {
                    &[]
                } else {
                    std::slice::from_raw_parts(values as *const f32, values_len)
                };
                let out = match put_along_axis_impl(
                    view,
                    indices_view,
                    axis_usize,
                    vals,
                    has_scalar.then_some(scalar_value as f32),
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_INDEX;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(out))),
                    dtype: DType::Float32,
                }
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, meta_ref) else {
                    error::set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                let vals: &[i64] = if values.is_null() || values_len == 0 {
                    &[]
                } else {
                    std::slice::from_raw_parts(values as *const i64, values_len)
                };
                let out = match put_along_axis_impl(
                    view,
                    indices_view,
                    axis_usize,
                    vals,
                    has_scalar.then_some(scalar_value as i64),
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_INDEX;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Int64(Arc::new(RwLock::new(out))),
                    dtype: DType::Int64,
                }
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, meta_ref) else {
                    error::set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                let vals: &[i32] = if values.is_null() || values_len == 0 {
                    &[]
                } else {
                    std::slice::from_raw_parts(values as *const i32, values_len)
                };
                let out = match put_along_axis_impl(
                    view,
                    indices_view,
                    axis_usize,
                    vals,
                    has_scalar.then_some(scalar_value as i32),
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_INDEX;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Int32(Arc::new(RwLock::new(out))),
                    dtype: DType::Int32,
                }
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, meta_ref) else {
                    error::set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                let vals: &[i16] = if values.is_null() || values_len == 0 {
                    &[]
                } else {
                    std::slice::from_raw_parts(values as *const i16, values_len)
                };
                let out = match put_along_axis_impl(
                    view,
                    indices_view,
                    axis_usize,
                    vals,
                    has_scalar.then_some(scalar_value as i16),
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_INDEX;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Int16(Arc::new(RwLock::new(out))),
                    dtype: DType::Int16,
                }
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, meta_ref) else {
                    error::set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                let vals: &[i8] = if values.is_null() || values_len == 0 {
                    &[]
                } else {
                    std::slice::from_raw_parts(values as *const i8, values_len)
                };
                let out = match put_along_axis_impl(
                    view,
                    indices_view,
                    axis_usize,
                    vals,
                    has_scalar.then_some(scalar_value as i8),
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_INDEX;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Int8(Arc::new(RwLock::new(out))),
                    dtype: DType::Int8,
                }
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, meta_ref) else {
                    error::set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                let vals: &[u64] = if values.is_null() || values_len == 0 {
                    &[]
                } else {
                    std::slice::from_raw_parts(values as *const u64, values_len)
                };
                let out = match put_along_axis_impl(
                    view,
                    indices_view,
                    axis_usize,
                    vals,
                    has_scalar.then_some(scalar_value as u64),
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_INDEX;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Uint64(Arc::new(RwLock::new(out))),
                    dtype: DType::Uint64,
                }
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, meta_ref) else {
                    error::set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                let vals: &[u32] = if values.is_null() || values_len == 0 {
                    &[]
                } else {
                    std::slice::from_raw_parts(values as *const u32, values_len)
                };
                let out = match put_along_axis_impl(
                    view,
                    indices_view,
                    axis_usize,
                    vals,
                    has_scalar.then_some(scalar_value as u32),
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_INDEX;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Uint32(Arc::new(RwLock::new(out))),
                    dtype: DType::Uint32,
                }
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, meta_ref) else {
                    error::set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                let vals: &[u16] = if values.is_null() || values_len == 0 {
                    &[]
                } else {
                    std::slice::from_raw_parts(values as *const u16, values_len)
                };
                let out = match put_along_axis_impl(
                    view,
                    indices_view,
                    axis_usize,
                    vals,
                    has_scalar.then_some(scalar_value as u16),
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_INDEX;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Uint16(Arc::new(RwLock::new(out))),
                    dtype: DType::Uint16,
                }
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, meta_ref) else {
                    error::set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                let vals: &[u8] = if values.is_null() || values_len == 0 {
                    &[]
                } else {
                    std::slice::from_raw_parts(values as *const u8, values_len)
                };
                let out = match put_along_axis_impl(
                    view,
                    indices_view,
                    axis_usize,
                    vals,
                    has_scalar.then_some(scalar_value as u8),
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_INDEX;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Uint8(Arc::new(RwLock::new(out))),
                    dtype: DType::Uint8,
                }
            }
            DType::Bool => {
                let Some(view) = extract_view_bool(wrapper, meta_ref) else {
                    error::set_last_error("Failed to extract bool view".to_string());
                    return ERR_GENERIC;
                };
                let vals: &[u8] = if values.is_null() || values_len == 0 {
                    &[]
                } else {
                    std::slice::from_raw_parts(values as *const u8, values_len)
                };
                let scalar = if has_scalar {
                    Some(if scalar_value != 0.0 { 1u8 } else { 0u8 })
                } else {
                    None
                };
                let out = match put_along_axis_impl(view, indices_view, axis_usize, vals, scalar) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_INDEX;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Bool(Arc::new(RwLock::new(out))),
                    dtype: DType::Bool,
                }
            }
        };

        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
