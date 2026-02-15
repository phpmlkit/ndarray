//! Axis sum reduction using ndarray's sum_axis() method.

use crate::core::view_helpers::{
    extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32, extract_view_i64,
    extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64, extract_view_u8,
};
use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::reductions::helpers::validate_axis;
use crate::ffi::NdArrayHandle;
use ndarray::Axis;
use parking_lot::RwLock;
use std::slice;
use std::sync::Arc;

/// Sum along axis.
#[no_mangle]
pub unsafe extern "C" fn ndarray_sum_axis(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
    axis: i32,
    keepdims: bool,
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
        let axis_usize = match validate_axis(shape_slice, axis) {
            Ok(a) => a,
            Err(e) => {
                crate::error::set_last_error(e);
                return ERR_GENERIC;
            }
        };

        // Match on dtype, extract view, compute sum along axis, and create result wrapper
        let result_wrapper = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.sum_axis(Axis(axis_usize));
                let final_arr = if keepdims {
                    result.insert_axis(Axis(axis_usize))
                } else {
                    result
                };
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(final_arr))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.mapv(|x| x as f64).sum_axis(Axis(axis_usize));
                let final_arr = if keepdims {
                    result.insert_axis(Axis(axis_usize))
                } else {
                    result
                };
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(final_arr.mapv(|x| x as f32)))),
                    dtype: DType::Float32,
                }
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.mapv(|x| x as f64).sum_axis(Axis(axis_usize));
                let final_arr = if keepdims {
                    result.insert_axis(Axis(axis_usize))
                } else {
                    result
                };
                NDArrayWrapper {
                    data: ArrayData::Int64(Arc::new(RwLock::new(final_arr.mapv(|x| x as i64)))),
                    dtype: DType::Int64,
                }
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.mapv(|x| x as f64).sum_axis(Axis(axis_usize));
                let final_arr = if keepdims {
                    result.insert_axis(Axis(axis_usize))
                } else {
                    result
                };
                NDArrayWrapper {
                    data: ArrayData::Int32(Arc::new(RwLock::new(final_arr.mapv(|x| x as i32)))),
                    dtype: DType::Int32,
                }
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.mapv(|x| x as f64).sum_axis(Axis(axis_usize));
                let final_arr = if keepdims {
                    result.insert_axis(Axis(axis_usize))
                } else {
                    result
                };
                NDArrayWrapper {
                    data: ArrayData::Int16(Arc::new(RwLock::new(final_arr.mapv(|x| x as i16)))),
                    dtype: DType::Int16,
                }
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.mapv(|x| x as f64).sum_axis(Axis(axis_usize));
                let final_arr = if keepdims {
                    result.insert_axis(Axis(axis_usize))
                } else {
                    result
                };
                NDArrayWrapper {
                    data: ArrayData::Int8(Arc::new(RwLock::new(final_arr.mapv(|x| x as i8)))),
                    dtype: DType::Int8,
                }
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.mapv(|x| x as f64).sum_axis(Axis(axis_usize));
                let final_arr = if keepdims {
                    result.insert_axis(Axis(axis_usize))
                } else {
                    result
                };
                NDArrayWrapper {
                    data: ArrayData::Uint64(Arc::new(RwLock::new(final_arr.mapv(|x| x as u64)))),
                    dtype: DType::Uint64,
                }
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.mapv(|x| x as f64).sum_axis(Axis(axis_usize));
                let final_arr = if keepdims {
                    result.insert_axis(Axis(axis_usize))
                } else {
                    result
                };
                NDArrayWrapper {
                    data: ArrayData::Uint32(Arc::new(RwLock::new(final_arr.mapv(|x| x as u32)))),
                    dtype: DType::Uint32,
                }
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.mapv(|x| x as f64).sum_axis(Axis(axis_usize));
                let final_arr = if keepdims {
                    result.insert_axis(Axis(axis_usize))
                } else {
                    result
                };
                NDArrayWrapper {
                    data: ArrayData::Uint16(Arc::new(RwLock::new(final_arr.mapv(|x| x as u16)))),
                    dtype: DType::Uint16,
                }
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.mapv(|x| x as f64).sum_axis(Axis(axis_usize));
                let final_arr = if keepdims {
                    result.insert_axis(Axis(axis_usize))
                } else {
                    result
                };
                NDArrayWrapper {
                    data: ArrayData::Uint8(Arc::new(RwLock::new(final_arr.mapv(|x| x as u8)))),
                    dtype: DType::Uint8,
                }
            }
            DType::Bool => {
                crate::error::set_last_error("sum_axis() not supported for Bool type".to_string());
                return ERR_GENERIC;
            }
        };

        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
