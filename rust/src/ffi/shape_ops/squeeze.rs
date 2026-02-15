//! Squeeze and expand_dims operations.

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

/// Remove axes of length 1 from the array.
///
/// If `axes` is null and `num_axes` is 0, removes all length-1 axes.
/// Otherwise, removes only the specified axes.
#[no_mangle]
pub unsafe extern "C" fn ndarray_squeeze(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
    axes: *const usize,
    num_axes: usize,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || out_handle.is_null() || shape.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = slice::from_raw_parts(shape, ndim);
        let strides_slice = slice::from_raw_parts(strides, ndim);

        // Determine which axes to squeeze
        let axes_to_squeeze: Vec<usize> = if num_axes == 0 {
            // Squeeze all length-1 axes
            shape_slice
                .iter()
                .enumerate()
                .filter(|(_, &dim)| dim == 1)
                .map(|(i, _)| i)
                .collect()
        } else {
            // Squeeze specific axes
            if axes.is_null() {
                return ERR_GENERIC;
            }
            let requested: Vec<usize> = slice::from_raw_parts(axes, num_axes).to_vec();

            // Validate axes
            for &axis in &requested {
                if axis >= ndim {
                    error::set_last_error(format!(
                        "Axis {} out of bounds for array with {} dimensions",
                        axis, ndim
                    ));
                    return ERR_SHAPE;
                }
                if shape_slice[axis] != 1 {
                    error::set_last_error(format!(
                        "Cannot squeeze axis {} with size {} (must be 1)",
                        axis, shape_slice[axis]
                    ));
                    return ERR_SHAPE;
                }
            }
            requested
        };

        // If no axes to squeeze, return a copy of the data
        if axes_to_squeeze.is_empty() {
            let result_wrapper = match wrapper.dtype {
                DType::Float64 => {
                    let Some(view) = extract_view_f64(wrapper, offset, shape_slice, strides_slice)
                    else {
                        error::set_last_error("Failed to extract f64 view".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Float64(Arc::new(RwLock::new(view.to_owned()))),
                        dtype: DType::Float64,
                    }
                }
                DType::Float32 => {
                    let Some(view) = extract_view_f32(wrapper, offset, shape_slice, strides_slice)
                    else {
                        error::set_last_error("Failed to extract f32 view".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Float32(Arc::new(RwLock::new(view.to_owned()))),
                        dtype: DType::Float32,
                    }
                }
                DType::Int64 => {
                    let Some(view) = extract_view_i64(wrapper, offset, shape_slice, strides_slice)
                    else {
                        error::set_last_error("Failed to extract i64 view".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Int64(Arc::new(RwLock::new(view.to_owned()))),
                        dtype: DType::Int64,
                    }
                }
                DType::Int32 => {
                    let Some(view) = extract_view_i32(wrapper, offset, shape_slice, strides_slice)
                    else {
                        error::set_last_error("Failed to extract i32 view".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Int32(Arc::new(RwLock::new(view.to_owned()))),
                        dtype: DType::Int32,
                    }
                }
                DType::Int16 => {
                    let Some(view) = extract_view_i16(wrapper, offset, shape_slice, strides_slice)
                    else {
                        error::set_last_error("Failed to extract i16 view".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Int16(Arc::new(RwLock::new(view.to_owned()))),
                        dtype: DType::Int16,
                    }
                }
                DType::Int8 => {
                    let Some(view) = extract_view_i8(wrapper, offset, shape_slice, strides_slice)
                    else {
                        error::set_last_error("Failed to extract i8 view".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Int8(Arc::new(RwLock::new(view.to_owned()))),
                        dtype: DType::Int8,
                    }
                }
                DType::Uint64 => {
                    let Some(view) = extract_view_u64(wrapper, offset, shape_slice, strides_slice)
                    else {
                        error::set_last_error("Failed to extract u64 view".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Uint64(Arc::new(RwLock::new(view.to_owned()))),
                        dtype: DType::Uint64,
                    }
                }
                DType::Uint32 => {
                    let Some(view) = extract_view_u32(wrapper, offset, shape_slice, strides_slice)
                    else {
                        error::set_last_error("Failed to extract u32 view".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Uint32(Arc::new(RwLock::new(view.to_owned()))),
                        dtype: DType::Uint32,
                    }
                }
                DType::Uint16 => {
                    let Some(view) = extract_view_u16(wrapper, offset, shape_slice, strides_slice)
                    else {
                        error::set_last_error("Failed to extract u16 view".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Uint16(Arc::new(RwLock::new(view.to_owned()))),
                        dtype: DType::Uint16,
                    }
                }
                DType::Uint8 => {
                    let Some(view) = extract_view_u8(wrapper, offset, shape_slice, strides_slice)
                    else {
                        error::set_last_error("Failed to extract u8 view".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Uint8(Arc::new(RwLock::new(view.to_owned()))),
                        dtype: DType::Uint8,
                    }
                }
                DType::Bool => {
                    error::set_last_error("squeeze() not supported for Bool type".to_string());
                    return ERR_GENERIC;
                }
            };
            *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
            return SUCCESS;
        }

        // Match on dtype, extract view, squeeze axes, and create result wrapper
        let result_wrapper = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                let mut result = view.to_owned();
                // Remove axes in reverse order to maintain correct indices
                for &axis in axes_to_squeeze.iter().rev() {
                    result = result.remove_axis(Axis(axis));
                }
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
                let mut result = view.to_owned();
                for &axis in axes_to_squeeze.iter().rev() {
                    result = result.remove_axis(Axis(axis));
                }
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
                let mut result = view.to_owned();
                for &axis in axes_to_squeeze.iter().rev() {
                    result = result.remove_axis(Axis(axis));
                }
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
                let mut result = view.to_owned();
                for &axis in axes_to_squeeze.iter().rev() {
                    result = result.remove_axis(Axis(axis));
                }
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
                let mut result = view.to_owned();
                for &axis in axes_to_squeeze.iter().rev() {
                    result = result.remove_axis(Axis(axis));
                }
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
                let mut result = view.to_owned();
                for &axis in axes_to_squeeze.iter().rev() {
                    result = result.remove_axis(Axis(axis));
                }
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
                let mut result = view.to_owned();
                for &axis in axes_to_squeeze.iter().rev() {
                    result = result.remove_axis(Axis(axis));
                }
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
                let mut result = view.to_owned();
                for &axis in axes_to_squeeze.iter().rev() {
                    result = result.remove_axis(Axis(axis));
                }
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
                let mut result = view.to_owned();
                for &axis in axes_to_squeeze.iter().rev() {
                    result = result.remove_axis(Axis(axis));
                }
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
                let mut result = view.to_owned();
                for &axis in axes_to_squeeze.iter().rev() {
                    result = result.remove_axis(Axis(axis));
                }
                NDArrayWrapper {
                    data: ArrayData::Uint8(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint8,
                }
            }
            DType::Bool => {
                error::set_last_error("squeeze() not supported for Bool type".to_string());
                return ERR_GENERIC;
            }
        };

        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}

/// Insert a new axis at the specified position.
///
/// Equivalent to NumPy's expand_dims.
#[no_mangle]
pub unsafe extern "C" fn ndarray_expand_dims(
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

        // Validate axis - can be from 0 to ndim (inclusive, for appending)
        if axis > ndim {
            error::set_last_error(format!(
                "Axis {} out of bounds for array with {} dimensions",
                axis, ndim
            ));
            return ERR_SHAPE;
        }

        // Match on dtype, extract view, insert axis, and create result wrapper
        let result_wrapper = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.insert_axis(Axis(axis)).to_owned();
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
                let result = view.insert_axis(Axis(axis)).to_owned();
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
                let result = view.insert_axis(Axis(axis)).to_owned();
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
                let result = view.insert_axis(Axis(axis)).to_owned();
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
                let result = view.insert_axis(Axis(axis)).to_owned();
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
                let result = view.insert_axis(Axis(axis)).to_owned();
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
                let result = view.insert_axis(Axis(axis)).to_owned();
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
                let result = view.insert_axis(Axis(axis)).to_owned();
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
                let result = view.insert_axis(Axis(axis)).to_owned();
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
                let result = view.insert_axis(Axis(axis)).to_owned();
                NDArrayWrapper {
                    data: ArrayData::Uint8(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint8,
                }
            }
            DType::Bool => {
                error::set_last_error("expand_dims() not supported for Bool type".to_string());
                return ERR_GENERIC;
            }
        };

        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
