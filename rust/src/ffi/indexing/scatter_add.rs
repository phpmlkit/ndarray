//! scatterAdd operation for duplicate-heavy index updates.

use crate::core::view_helpers::{
    extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32, extract_view_i64,
    extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64, extract_view_u8,
};
use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{self, ERR_GENERIC, ERR_INDEX, ERR_MATH, SUCCESS};
use crate::ffi::NdArrayHandle;
use ndarray::{ArrayD, IxDyn};
use parking_lot::RwLock;
use std::ffi::c_void;
use std::slice;
use std::sync::Arc;

#[inline]
fn normalize_index(idx: i64, len: usize) -> Result<usize, String> {
    let mut i = idx;
    if i < 0 {
        i += len as i64;
    }
    if i < 0 || i >= len as i64 {
        return Err(format!(
            "Index {} is out of bounds for axis with size {}",
            idx, len
        ));
    }
    Ok(i as usize)
}

fn scatter_add_impl<T>(
    view: ndarray::ArrayViewD<'_, T>,
    indices: &[i64],
    updates: &[T],
    scalar: Option<T>,
) -> Result<ArrayD<T>, String>
where
    T: Copy + std::ops::AddAssign,
{
    let shape = view.shape().to_vec();
    let mut flat: Vec<T> = view.iter().copied().collect();
    if updates.is_empty() && scalar.is_none() {
        return Err("scatterAdd requires scalar or non-empty updates".to_string());
    }

    for (k, &idx) in indices.iter().enumerate() {
        let i = normalize_index(idx, flat.len())?;
        let upd = match scalar {
            Some(s) => s,
            None => updates[k % updates.len()],
        };
        flat[i] += upd;
    }

    ArrayD::from_shape_vec(IxDyn(&shape), flat)
        .map_err(|e| format!("Failed to create scatterAdd output: {}", e))
}

/// Add updates into flattened indices and return a mutated copy.
#[no_mangle]
pub unsafe extern "C" fn ndarray_scatter_add_flat(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
    indices: *const i64,
    indices_len: usize,
    updates: *const c_void,
    updates_len: usize,
    scalar_update: f64,
    has_scalar: bool,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || shape.is_null() || strides.is_null() || indices.is_null() || out_handle.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = slice::from_raw_parts(shape, ndim);
        let strides_slice = slice::from_raw_parts(strides, ndim);
        let idx_slice = slice::from_raw_parts(indices, indices_len);

        let result_wrapper = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, offset, shape_slice, strides_slice) else { error::set_last_error("Failed to extract f64 view".to_string()); return ERR_GENERIC; };
                let upd: &[f64] = if updates.is_null() || updates_len == 0 { &[] } else { slice::from_raw_parts(updates as *const f64, updates_len) };
                let out = match scatter_add_impl(view, idx_slice, upd, has_scalar.then_some(scalar_update as f64)) { Ok(v) => v, Err(e) => { error::set_last_error(e); return ERR_INDEX; } };
                NDArrayWrapper { data: ArrayData::Float64(Arc::new(RwLock::new(out))), dtype: DType::Float64 }
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, offset, shape_slice, strides_slice) else { error::set_last_error("Failed to extract f32 view".to_string()); return ERR_GENERIC; };
                let upd: &[f32] = if updates.is_null() || updates_len == 0 { &[] } else { slice::from_raw_parts(updates as *const f32, updates_len) };
                let out = match scatter_add_impl(view, idx_slice, upd, has_scalar.then_some(scalar_update as f32)) { Ok(v) => v, Err(e) => { error::set_last_error(e); return ERR_INDEX; } };
                NDArrayWrapper { data: ArrayData::Float32(Arc::new(RwLock::new(out))), dtype: DType::Float32 }
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, offset, shape_slice, strides_slice) else { error::set_last_error("Failed to extract i64 view".to_string()); return ERR_GENERIC; };
                let upd: &[i64] = if updates.is_null() || updates_len == 0 { &[] } else { slice::from_raw_parts(updates as *const i64, updates_len) };
                let out = match scatter_add_impl(view, idx_slice, upd, has_scalar.then_some(scalar_update as i64)) { Ok(v) => v, Err(e) => { error::set_last_error(e); return ERR_INDEX; } };
                NDArrayWrapper { data: ArrayData::Int64(Arc::new(RwLock::new(out))), dtype: DType::Int64 }
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, offset, shape_slice, strides_slice) else { error::set_last_error("Failed to extract i32 view".to_string()); return ERR_GENERIC; };
                let upd: &[i32] = if updates.is_null() || updates_len == 0 { &[] } else { slice::from_raw_parts(updates as *const i32, updates_len) };
                let out = match scatter_add_impl(view, idx_slice, upd, has_scalar.then_some(scalar_update as i32)) { Ok(v) => v, Err(e) => { error::set_last_error(e); return ERR_INDEX; } };
                NDArrayWrapper { data: ArrayData::Int32(Arc::new(RwLock::new(out))), dtype: DType::Int32 }
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, offset, shape_slice, strides_slice) else { error::set_last_error("Failed to extract i16 view".to_string()); return ERR_GENERIC; };
                let upd: &[i16] = if updates.is_null() || updates_len == 0 { &[] } else { slice::from_raw_parts(updates as *const i16, updates_len) };
                let out = match scatter_add_impl(view, idx_slice, upd, has_scalar.then_some(scalar_update as i16)) { Ok(v) => v, Err(e) => { error::set_last_error(e); return ERR_INDEX; } };
                NDArrayWrapper { data: ArrayData::Int16(Arc::new(RwLock::new(out))), dtype: DType::Int16 }
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, offset, shape_slice, strides_slice) else { error::set_last_error("Failed to extract i8 view".to_string()); return ERR_GENERIC; };
                let upd: &[i8] = if updates.is_null() || updates_len == 0 { &[] } else { slice::from_raw_parts(updates as *const i8, updates_len) };
                let out = match scatter_add_impl(view, idx_slice, upd, has_scalar.then_some(scalar_update as i8)) { Ok(v) => v, Err(e) => { error::set_last_error(e); return ERR_INDEX; } };
                NDArrayWrapper { data: ArrayData::Int8(Arc::new(RwLock::new(out))), dtype: DType::Int8 }
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, offset, shape_slice, strides_slice) else { error::set_last_error("Failed to extract u64 view".to_string()); return ERR_GENERIC; };
                let upd: &[u64] = if updates.is_null() || updates_len == 0 { &[] } else { slice::from_raw_parts(updates as *const u64, updates_len) };
                let out = match scatter_add_impl(view, idx_slice, upd, has_scalar.then_some(scalar_update as u64)) { Ok(v) => v, Err(e) => { error::set_last_error(e); return ERR_INDEX; } };
                NDArrayWrapper { data: ArrayData::Uint64(Arc::new(RwLock::new(out))), dtype: DType::Uint64 }
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, offset, shape_slice, strides_slice) else { error::set_last_error("Failed to extract u32 view".to_string()); return ERR_GENERIC; };
                let upd: &[u32] = if updates.is_null() || updates_len == 0 { &[] } else { slice::from_raw_parts(updates as *const u32, updates_len) };
                let out = match scatter_add_impl(view, idx_slice, upd, has_scalar.then_some(scalar_update as u32)) { Ok(v) => v, Err(e) => { error::set_last_error(e); return ERR_INDEX; } };
                NDArrayWrapper { data: ArrayData::Uint32(Arc::new(RwLock::new(out))), dtype: DType::Uint32 }
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, offset, shape_slice, strides_slice) else { error::set_last_error("Failed to extract u16 view".to_string()); return ERR_GENERIC; };
                let upd: &[u16] = if updates.is_null() || updates_len == 0 { &[] } else { slice::from_raw_parts(updates as *const u16, updates_len) };
                let out = match scatter_add_impl(view, idx_slice, upd, has_scalar.then_some(scalar_update as u16)) { Ok(v) => v, Err(e) => { error::set_last_error(e); return ERR_INDEX; } };
                NDArrayWrapper { data: ArrayData::Uint16(Arc::new(RwLock::new(out))), dtype: DType::Uint16 }
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, offset, shape_slice, strides_slice) else { error::set_last_error("Failed to extract u8 view".to_string()); return ERR_GENERIC; };
                let upd: &[u8] = if updates.is_null() || updates_len == 0 { &[] } else { slice::from_raw_parts(updates as *const u8, updates_len) };
                let out = match scatter_add_impl(view, idx_slice, upd, has_scalar.then_some(scalar_update as u8)) { Ok(v) => v, Err(e) => { error::set_last_error(e); return ERR_INDEX; } };
                NDArrayWrapper { data: ArrayData::Uint8(Arc::new(RwLock::new(out))), dtype: DType::Uint8 }
            }
            DType::Bool => {
                error::set_last_error("scatterAdd is not supported for Bool dtype".to_string());
                return ERR_MATH;
            }
        };

        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
