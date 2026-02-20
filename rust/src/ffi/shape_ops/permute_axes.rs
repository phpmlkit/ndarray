//! Permute axes operations.

use crate::core::view_helpers::{
    extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32, extract_view_i64,
    extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64, extract_view_u8,
};
use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{self, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::ffi::write_output_metadata;
use crate::ffi::NdArrayHandle;
use crate::ffi::ViewMetadata;
use parking_lot::RwLock;
use std::sync::Arc;

/// Permute axes of the array and return in standard layout.
///
/// The axes array specifies the new order of axes.
/// Panics if axes are invalid (out of bounds, missing, or duplicated).
#[no_mangle]
pub unsafe extern "C" fn ndarray_permute_axes(
    handle: *const NdArrayHandle,
    meta: *const ViewMetadata,
    axes: *const usize,
    num_axes: usize,
    out_handle: *mut *mut NdArrayHandle,
    out_dtype: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if handle.is_null()
        || meta.is_null()
        || axes.is_null()
        || out_handle.is_null()
        || out_dtype.is_null()
        || out_shape.is_null()
        || out_ndim.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let meta_ref = &*meta;
        let shape_slice = meta_ref.shape_slice();
        let ndim = meta_ref.ndim;
        let axes_slice = std::slice::from_raw_parts(axes, num_axes);

        // Validate axes
        if num_axes != ndim {
            error::set_last_error(format!(
                "permute_axes requires {} axes, got {}",
                ndim, num_axes
            ));
            return ERR_SHAPE;
        }

        // Check for valid axis indices and duplicates
        let mut seen = vec![false; ndim];
        for &axis in axes_slice {
            if axis >= ndim {
                error::set_last_error(format!(
                    "Axis {} out of bounds for array with {} dimensions",
                    axis, ndim
                ));
                return ERR_SHAPE;
            }
            if seen[axis] {
                error::set_last_error(format!("Axis {} appears more than once", axis));
                return ERR_SHAPE;
            }
            seen[axis] = true;
        }

        // Compute new shape after permutation
        let new_shape: Vec<usize> = axes_slice.iter().map(|&i| shape_slice[i]).collect();

        // Match on dtype, extract view, permute axes, and create result wrapper
        let result_wrapper = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, meta_ref) else {
                    error::set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                let permuted = view.permuted_axes(axes_slice);
                let data: Vec<f64> = permuted.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&new_shape), data)
                    .expect("Failed to create permuted array");
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, meta_ref) else {
                    error::set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                let permuted = view.permuted_axes(axes_slice);
                let data: Vec<f32> = permuted.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&new_shape), data)
                    .expect("Failed to create permuted array");
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(result))),
                    dtype: DType::Float32,
                }
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, meta_ref) else {
                    error::set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                let permuted = view.permuted_axes(axes_slice);
                let data: Vec<i64> = permuted.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&new_shape), data)
                    .expect("Failed to create permuted array");
                NDArrayWrapper {
                    data: ArrayData::Int64(Arc::new(RwLock::new(result))),
                    dtype: DType::Int64,
                }
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, meta_ref) else {
                    error::set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                let permuted = view.permuted_axes(axes_slice);
                let data: Vec<i32> = permuted.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&new_shape), data)
                    .expect("Failed to create permuted array");
                NDArrayWrapper {
                    data: ArrayData::Int32(Arc::new(RwLock::new(result))),
                    dtype: DType::Int32,
                }
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, meta_ref) else {
                    error::set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                let permuted = view.permuted_axes(axes_slice);
                let data: Vec<i16> = permuted.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&new_shape), data)
                    .expect("Failed to create permuted array");
                NDArrayWrapper {
                    data: ArrayData::Int16(Arc::new(RwLock::new(result))),
                    dtype: DType::Int16,
                }
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, meta_ref) else {
                    error::set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                let permuted = view.permuted_axes(axes_slice);
                let data: Vec<i8> = permuted.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&new_shape), data)
                    .expect("Failed to create permuted array");
                NDArrayWrapper {
                    data: ArrayData::Int8(Arc::new(RwLock::new(result))),
                    dtype: DType::Int8,
                }
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, meta_ref) else {
                    error::set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                let permuted = view.permuted_axes(axes_slice);
                let data: Vec<u64> = permuted.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&new_shape), data)
                    .expect("Failed to create permuted array");
                NDArrayWrapper {
                    data: ArrayData::Uint64(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint64,
                }
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, meta_ref) else {
                    error::set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                let permuted = view.permuted_axes(axes_slice);
                let data: Vec<u32> = permuted.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&new_shape), data)
                    .expect("Failed to create permuted array");
                NDArrayWrapper {
                    data: ArrayData::Uint32(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint32,
                }
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, meta_ref) else {
                    error::set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                let permuted = view.permuted_axes(axes_slice);
                let data: Vec<u16> = permuted.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&new_shape), data)
                    .expect("Failed to create permuted array");
                NDArrayWrapper {
                    data: ArrayData::Uint16(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint16,
                }
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, meta_ref) else {
                    error::set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                let permuted = view.permuted_axes(axes_slice);
                let data: Vec<u8> = permuted.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&new_shape), data)
                    .expect("Failed to create permuted array");
                NDArrayWrapper {
                    data: ArrayData::Uint8(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint8,
                }
            }
            DType::Bool => {
                error::set_last_error("permute_axes() not supported for Bool type".to_string());
                return ERR_GENERIC;
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
