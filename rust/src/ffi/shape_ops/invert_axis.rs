//! Invert axis operations.

use crate::core::view_helpers::{
    extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32, extract_view_i64,
    extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64, extract_view_u8,
};
use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{self, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::ffi::NdArrayHandle;
use ndarray::Axis;
use parking_lot::RwLock;
use std::slice;
use std::sync::Arc;

/// Reverse the stride of axis and return in standard layout.
///
/// Panics if the axis is out of bounds.
#[no_mangle]
pub unsafe extern "C" fn ndarray_invert_axis(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
    axis: usize,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || out_handle.is_null() || shape.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = slice::from_raw_parts(shape, ndim);
        let strides_slice = slice::from_raw_parts(strides, ndim);

        // Validate axis
        if axis >= ndim {
            error::set_last_error(format!(
                "Axis {} out of bounds for array with {} dimensions",
                axis, ndim
            ));
            return ERR_SHAPE;
        }

        // Clone shape for result
        let new_shape = shape_slice.to_vec();

        // Match on dtype, extract view, invert axis, and create result wrapper
        let result_wrapper = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                // Invert axis and collect in standard layout
                let mut inverted = view.to_owned();
                inverted.invert_axis(Axis(axis));
                // Collect in standard layout by iterating
                let data: Vec<f64> = inverted.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&new_shape), data)
                    .expect("Failed to create inverted array");
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                let mut inverted = view.to_owned();
                inverted.invert_axis(Axis(axis));
                let data: Vec<f32> = inverted.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&new_shape), data)
                    .expect("Failed to create inverted array");
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(result))),
                    dtype: DType::Float32,
                }
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                let mut inverted = view.to_owned();
                inverted.invert_axis(Axis(axis));
                let data: Vec<i64> = inverted.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&new_shape), data)
                    .expect("Failed to create inverted array");
                NDArrayWrapper {
                    data: ArrayData::Int64(Arc::new(RwLock::new(result))),
                    dtype: DType::Int64,
                }
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                let mut inverted = view.to_owned();
                inverted.invert_axis(Axis(axis));
                let data: Vec<i32> = inverted.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&new_shape), data)
                    .expect("Failed to create inverted array");
                NDArrayWrapper {
                    data: ArrayData::Int32(Arc::new(RwLock::new(result))),
                    dtype: DType::Int32,
                }
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                let mut inverted = view.to_owned();
                inverted.invert_axis(Axis(axis));
                let data: Vec<i16> = inverted.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&new_shape), data)
                    .expect("Failed to create inverted array");
                NDArrayWrapper {
                    data: ArrayData::Int16(Arc::new(RwLock::new(result))),
                    dtype: DType::Int16,
                }
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                let mut inverted = view.to_owned();
                inverted.invert_axis(Axis(axis));
                let data: Vec<i8> = inverted.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&new_shape), data)
                    .expect("Failed to create inverted array");
                NDArrayWrapper {
                    data: ArrayData::Int8(Arc::new(RwLock::new(result))),
                    dtype: DType::Int8,
                }
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                let mut inverted = view.to_owned();
                inverted.invert_axis(Axis(axis));
                let data: Vec<u64> = inverted.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&new_shape), data)
                    .expect("Failed to create inverted array");
                NDArrayWrapper {
                    data: ArrayData::Uint64(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint64,
                }
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                let mut inverted = view.to_owned();
                inverted.invert_axis(Axis(axis));
                let data: Vec<u32> = inverted.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&new_shape), data)
                    .expect("Failed to create inverted array");
                NDArrayWrapper {
                    data: ArrayData::Uint32(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint32,
                }
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                let mut inverted = view.to_owned();
                inverted.invert_axis(Axis(axis));
                let data: Vec<u16> = inverted.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&new_shape), data)
                    .expect("Failed to create inverted array");
                NDArrayWrapper {
                    data: ArrayData::Uint16(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint16,
                }
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                let mut inverted = view.to_owned();
                inverted.invert_axis(Axis(axis));
                let data: Vec<u8> = inverted.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&new_shape), data)
                    .expect("Failed to create inverted array");
                NDArrayWrapper {
                    data: ArrayData::Uint8(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint8,
                }
            }
            DType::Bool => {
                let Some(view) = extract_view_u8(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                let mut inverted = view.to_owned();
                inverted.invert_axis(Axis(axis));
                let data: Vec<u8> = inverted.iter().cloned().collect();
                let result = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&new_shape), data)
                    .expect("Failed to create inverted array");
                NDArrayWrapper {
                    data: ArrayData::Bool(Arc::new(RwLock::new(result))),
                    dtype: DType::Bool,
                }
            }
        };

        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
