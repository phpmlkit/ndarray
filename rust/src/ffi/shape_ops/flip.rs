//! Flip operations - reverse elements along axis/axes.

use crate::core::view_helpers::{
    extract_view_bool, extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32,
    extract_view_i64, extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64,
    extract_view_u8,
};
use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{self, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::ffi::write_output_metadata;
use crate::ffi::NdArrayHandle;
use crate::ffi::ViewMetadata;
use ndarray::Axis;
use parking_lot::RwLock;
use std::sync::Arc;

/// Flip (reverse) elements along specified axes.
///
/// # Arguments
/// * `axes` - Array of axis indices to flip. Can be empty (flip all axes).
/// * `num_axes` - Number of axes to flip. If 0, flips all axes.
#[no_mangle]
pub unsafe extern "C" fn ndarray_flip(
    handle: *const NdArrayHandle,
    meta: *const ViewMetadata,
    axes: *const i64,
    num_axes: usize,
    out_handle: *mut *mut NdArrayHandle,
    out_dtype: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if handle.is_null()
        || meta.is_null()
        || out_handle.is_null()
        || out_dtype.is_null()
        || out_shape.is_null()
        || out_ndim.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let meta = &*meta;
        let ndim = meta.ndim;

        // Parse axes
        let axes_to_flip: Vec<usize> = if num_axes == 0 {
            // Flip all axes
            (0..ndim).collect()
        } else {
            let axes_slice = std::slice::from_raw_parts(axes, num_axes);
            axes_slice
                .iter()
                .map(|&ax| {
                    if ax < 0 {
                        (ndim as i64 + ax) as usize
                    } else {
                        ax as usize
                    }
                })
                .collect()
        };

        // Validate all axes
        for &axis in &axes_to_flip {
            if axis >= ndim {
                error::set_last_error(format!(
                    "Axis {} out of bounds for array with {} dimensions",
                    axis, ndim
                ));
                return ERR_SHAPE;
            }
        }

        let result_wrapper = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, meta) else {
                    error::set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                let mut flipped = view.to_owned();
                for &axis in &axes_to_flip {
                    flipped.invert_axis(Axis(axis));
                }
                let data: Vec<f64> = flipped.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(flipped.raw_dim(), data)
                    .expect("Failed to create flipped array");
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, meta) else {
                    error::set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                let mut flipped = view.to_owned();
                for &axis in &axes_to_flip {
                    flipped.invert_axis(Axis(axis));
                }
                let data: Vec<f32> = flipped.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(flipped.raw_dim(), data)
                    .expect("Failed to create flipped array");
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(result))),
                    dtype: DType::Float32,
                }
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, meta) else {
                    error::set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                let mut flipped = view.to_owned();
                for &axis in &axes_to_flip {
                    flipped.invert_axis(Axis(axis));
                }
                let data: Vec<i64> = flipped.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(flipped.raw_dim(), data)
                    .expect("Failed to create flipped array");
                NDArrayWrapper {
                    data: ArrayData::Int64(Arc::new(RwLock::new(result))),
                    dtype: DType::Int64,
                }
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, meta) else {
                    error::set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                let mut flipped = view.to_owned();
                for &axis in &axes_to_flip {
                    flipped.invert_axis(Axis(axis));
                }
                let data: Vec<i32> = flipped.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(flipped.raw_dim(), data)
                    .expect("Failed to create flipped array");
                NDArrayWrapper {
                    data: ArrayData::Int32(Arc::new(RwLock::new(result))),
                    dtype: DType::Int32,
                }
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, meta) else {
                    error::set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                let mut flipped = view.to_owned();
                for &axis in &axes_to_flip {
                    flipped.invert_axis(Axis(axis));
                }
                let data: Vec<i16> = flipped.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(flipped.raw_dim(), data)
                    .expect("Failed to create flipped array");
                NDArrayWrapper {
                    data: ArrayData::Int16(Arc::new(RwLock::new(result))),
                    dtype: DType::Int16,
                }
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, meta) else {
                    error::set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                let mut flipped = view.to_owned();
                for &axis in &axes_to_flip {
                    flipped.invert_axis(Axis(axis));
                }
                let data: Vec<i8> = flipped.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(flipped.raw_dim(), data)
                    .expect("Failed to create flipped array");
                NDArrayWrapper {
                    data: ArrayData::Int8(Arc::new(RwLock::new(result))),
                    dtype: DType::Int8,
                }
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, meta) else {
                    error::set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                let mut flipped = view.to_owned();
                for &axis in &axes_to_flip {
                    flipped.invert_axis(Axis(axis));
                }
                let data: Vec<u64> = flipped.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(flipped.raw_dim(), data)
                    .expect("Failed to create flipped array");
                NDArrayWrapper {
                    data: ArrayData::Uint64(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint64,
                }
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, meta) else {
                    error::set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                let mut flipped = view.to_owned();
                for &axis in &axes_to_flip {
                    flipped.invert_axis(Axis(axis));
                }
                let data: Vec<u32> = flipped.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(flipped.raw_dim(), data)
                    .expect("Failed to create flipped array");
                NDArrayWrapper {
                    data: ArrayData::Uint32(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint32,
                }
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, meta) else {
                    error::set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                let mut flipped = view.to_owned();
                for &axis in &axes_to_flip {
                    flipped.invert_axis(Axis(axis));
                }
                let data: Vec<u16> = flipped.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(flipped.raw_dim(), data)
                    .expect("Failed to create flipped array");
                NDArrayWrapper {
                    data: ArrayData::Uint16(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint16,
                }
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, meta) else {
                    error::set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                let mut flipped = view.to_owned();
                for &axis in &axes_to_flip {
                    flipped.invert_axis(Axis(axis));
                }
                let data: Vec<u8> = flipped.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(flipped.raw_dim(), data)
                    .expect("Failed to create flipped array");
                NDArrayWrapper {
                    data: ArrayData::Uint8(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint8,
                }
            }
            DType::Bool => {
                let Some(view) = extract_view_bool(wrapper, meta) else {
                    error::set_last_error("Failed to extract bool view".to_string());
                    return ERR_GENERIC;
                };
                let mut flipped = view.to_owned();
                for &axis in &axes_to_flip {
                    flipped.invert_axis(Axis(axis));
                }
                let data: Vec<u8> = flipped.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(flipped.raw_dim(), data)
                    .expect("Failed to create flipped array");
                NDArrayWrapper {
                    data: ArrayData::Bool(Arc::new(RwLock::new(result))),
                    dtype: DType::Bool,
                }
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
