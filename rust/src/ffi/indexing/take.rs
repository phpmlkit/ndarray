//! Gather values by indices.

use crate::core::view_helpers::{
    extract_view_bool, extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32,
    extract_view_i64, extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64,
    extract_view_u8,
};
use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{self, ERR_DTYPE, ERR_GENERIC, ERR_INDEX, ERR_SHAPE, SUCCESS};
use crate::ffi::indexing::utils::normalize_index;
use crate::ffi::output_meta::write_output_metadata;
use crate::ffi::reductions::helpers::validate_axis;
use crate::ffi::{NdArrayHandle, ViewMetadata};
use ndarray::{ArrayD, Dimension, IxDyn};
use parking_lot::RwLock;
use std::sync::Arc;

fn take_impl<T: Copy>(
    view: ndarray::ArrayViewD<'_, T>,
    indices: &[i64],
    indices_shape: &[usize],
) -> Result<ArrayD<T>, String> {
    let flat: Vec<T> = view.iter().copied().collect();
    let len = flat.len();
    let mut out = Vec::with_capacity(indices.len());
    for &idx in indices {
        let i = normalize_index(idx, len)?;
        out.push(flat[i]);
    }
    ArrayD::from_shape_vec(IxDyn(indices_shape), out)
        .map_err(|e| format!("Failed to create take output: {}", e))
}

/// Gather values by flattened logical indices.
#[no_mangle]
pub unsafe extern "C" fn ndarray_take(
    handle: *const NdArrayHandle,
    meta: *const ViewMetadata,
    indices_handle: *const NdArrayHandle,
    indices_meta: *const ViewMetadata,
    out_handle: *mut *mut NdArrayHandle,
    out_dtype: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if handle.is_null()
        || meta.is_null()
        || indices_handle.is_null()
        || indices_meta.is_null()
        || out_handle.is_null()
        || out_ndim.is_null()
        || out_shape.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let indices_wrapper = NdArrayHandle::as_wrapper(indices_handle as *mut _);
        let meta_ref = &*meta;
        let indices_meta_ref = &*indices_meta;

        if indices_wrapper.dtype != DType::Int64 {
            error::set_last_error("take indices must have Int64 dtype".to_string());
            return ERR_DTYPE;
        }

        let Some(indices_view) = extract_view_i64(indices_wrapper, indices_meta_ref) else {
            error::set_last_error("Failed to extract Int64 indices view".to_string());
            return ERR_GENERIC;
        };

        let idx_slice = indices_view.as_slice().unwrap_or(&[]);
        let idx_shape_slice = indices_meta_ref.shape_slice();

        let result_wrapper = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, meta_ref) else {
                    error::set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                let out = match take_impl(view, idx_slice, idx_shape_slice) {
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
                let out = match take_impl(view, idx_slice, idx_shape_slice) {
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
                let out = match take_impl(view, idx_slice, idx_shape_slice) {
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
                let out = match take_impl(view, idx_slice, idx_shape_slice) {
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
                let out = match take_impl(view, idx_slice, idx_shape_slice) {
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
                let out = match take_impl(view, idx_slice, idx_shape_slice) {
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
                let out = match take_impl(view, idx_slice, idx_shape_slice) {
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
                let out = match take_impl(view, idx_slice, idx_shape_slice) {
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
                let out = match take_impl(view, idx_slice, idx_shape_slice) {
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
                let out = match take_impl(view, idx_slice, idx_shape_slice) {
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
                let out = match take_impl(view, idx_slice, idx_shape_slice) {
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

        if let Err(e) =
            write_output_metadata(&result_wrapper, out_dtype, out_ndim, out_shape, max_ndim)
        {
            error::set_last_error(format!("Failed to write output metadata: {}", e));
            return ERR_GENERIC;
        }

        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}

/// Take slices along an axis.
///
/// This selects entire slices along the specified axis according to indices.
/// Output shape = input_shape[:axis] + indices_shape + input_shape[axis+1:]
fn take_axis_impl<T: Copy>(
    view: ndarray::ArrayViewD<'_, T>,
    indices: &[i64],
    indices_shape: &[usize],
    axis: usize,
) -> Result<ArrayD<T>, String> {
    let in_shape = view.shape();
    let ndim = in_shape.len();

    if axis >= ndim {
        return Err(format!(
            "Axis {} is out of bounds for array with {} dimensions",
            axis, ndim
        ));
    }

    // Validate all indices
    let axis_len = in_shape[axis];
    let mut normalized_indices: Vec<usize> = Vec::with_capacity(indices.len());
    for &idx in indices {
        normalized_indices.push(normalize_index(idx, axis_len)?);
    }

    // Compute output shape: input_shape[:axis] + indices_shape + input_shape[axis+1:]
    let mut out_shape: Vec<usize> = Vec::with_capacity(ndim - 1 + indices_shape.len());
    out_shape.extend_from_slice(&in_shape[..axis]);
    out_shape.extend_from_slice(indices_shape);
    out_shape.extend_from_slice(&in_shape[axis + 1..]);

    let out_size: usize = out_shape.iter().product();
    let mut out = Vec::with_capacity(out_size);

    if ndim == 1 {
        // Simple 1D case: just gather elements at indices
        for &idx in &normalized_indices {
            out.push(view[IxDyn(&[idx])]);
        }
    } else {
        // Multi-dimensional case: iterate through output positions
        // Output shape = input_shape[:axis] + indices_shape + input_shape[axis+1:]
        // For each output position, we need to map it back to an input position
        //
        // The indices dimensions are inserted at position `axis` in the output
        // So out_idx[axis : axis + indices_shape.len()] corresponds to the indices
        //
        // Example: input shape [3, 4], indices shape [2], axis=0
        //   output shape = [2, 4]
        //   out_idx = [i, j] where i is the index into indices array, j is column
        //   input_idx = [indices[i], j]
        //
        // Example: input shape [3, 4], indices shape [2, 2], axis=0
        //   output shape = [2, 2, 4]
        //   out_idx = [i, j, k] where [i, j] maps to flat index in indices, k is column
        //   input_idx = [indices[flat_idx(i,j)], k]

        let indices_ndim = indices_shape.len();

        // Pre-compute strides for the indices shape to convert multi-dim to flat
        let indices_strides: Vec<usize> = indices_shape
            .iter()
            .rev()
            .scan(1, |state, &dim| {
                let stride = *state;
                *state *= dim;
                Some(stride)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        for out_idx in ndarray::indices(IxDyn(&out_shape)) {
            let out_idx_slice = out_idx.slice();

            // Build input index
            let mut in_idx: Vec<usize> = Vec::with_capacity(ndim);

            // Add indices before axis (from out_idx[0..axis])
            in_idx.extend_from_slice(&out_idx_slice[..axis]);

            // Calculate flat position in indices array
            // The indices dimensions are at positions [axis, axis + indices_ndim)
            let mut flat_indices_pos = 0;
            for (i, _) in indices_shape.iter().enumerate() {
                flat_indices_pos += out_idx_slice[axis + i] * indices_strides[i];
            }

            // Add the selected index from the indices array
            if flat_indices_pos < normalized_indices.len() {
                in_idx.push(normalized_indices[flat_indices_pos]);
            }

            // Add indices after axis (skipping the indices_shape dimensions)
            in_idx.extend_from_slice(&out_idx_slice[axis + indices_ndim..]);

            // Get value from input
            if let Some(&val) = view.get(IxDyn(&in_idx)) {
                out.push(val);
            } else {
                return Err(format!(
                    "Failed to get value at index {:?} from input",
                    in_idx
                ));
            }
        }
    }

    ArrayD::from_shape_vec(IxDyn(&out_shape), out)
        .map_err(|e| format!("Failed to create take_axis output: {}", e))
}

/// Take slices along an axis.
///
/// This selects entire slices along the specified axis according to indices.
#[no_mangle]
pub unsafe extern "C" fn ndarray_take_axis(
    handle: *const NdArrayHandle,
    meta: *const ViewMetadata,
    indices_handle: *const NdArrayHandle,
    indices_meta: *const ViewMetadata,
    axis: i32,
    out_handle: *mut *mut NdArrayHandle,
    out_dtype: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if handle.is_null()
        || meta.is_null()
        || indices_handle.is_null()
        || indices_meta.is_null()
        || out_handle.is_null()
        || out_ndim.is_null()
        || out_shape.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let indices_wrapper = NdArrayHandle::as_wrapper(indices_handle as *mut _);
        let meta_ref = &*meta;
        let indices_meta_ref = &*indices_meta;

        if indices_wrapper.dtype != DType::Int64 {
            error::set_last_error("take_axis indices must have Int64 dtype".to_string());
            return ERR_DTYPE;
        }

        let Some(indices_view) = extract_view_i64(indices_wrapper, indices_meta_ref) else {
            error::set_last_error("Failed to extract Int64 indices view".to_string());
            return ERR_GENERIC;
        };

        let idx_slice = indices_view.as_slice().unwrap_or(&[]);
        let idx_shape_slice = indices_meta_ref.shape_slice();

        let axis_usize = match validate_axis(meta_ref.shape_slice(), axis) {
            Ok(v) => v,
            Err(e) => {
                error::set_last_error(e);
                return ERR_SHAPE;
            }
        };

        let result_wrapper = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, meta_ref) else {
                    error::set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                let out = match take_axis_impl(view, idx_slice, idx_shape_slice, axis_usize) {
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
                let out = match take_axis_impl(view, idx_slice, idx_shape_slice, axis_usize) {
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
                let out = match take_axis_impl(view, idx_slice, idx_shape_slice, axis_usize) {
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
                let out = match take_axis_impl(view, idx_slice, idx_shape_slice, axis_usize) {
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
                let out = match take_axis_impl(view, idx_slice, idx_shape_slice, axis_usize) {
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
                let out = match take_axis_impl(view, idx_slice, idx_shape_slice, axis_usize) {
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
                let out = match take_axis_impl(view, idx_slice, idx_shape_slice, axis_usize) {
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
                let out = match take_axis_impl(view, idx_slice, idx_shape_slice, axis_usize) {
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
                let out = match take_axis_impl(view, idx_slice, idx_shape_slice, axis_usize) {
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
                let out = match take_axis_impl(view, idx_slice, idx_shape_slice, axis_usize) {
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
                let out = match take_axis_impl(view, idx_slice, idx_shape_slice, axis_usize) {
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

        if let Err(e) =
            write_output_metadata(&result_wrapper, out_dtype, out_ndim, out_shape, max_ndim)
        {
            error::set_last_error(format!("Failed to write output metadata: {}", e));
            return ERR_GENERIC;
        }

        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
