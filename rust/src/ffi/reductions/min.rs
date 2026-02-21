//! Min reduction.

use std::ffi::c_void;

use crate::core::view_helpers::{
    extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32, extract_view_i64,
    extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64, extract_view_u8,
};
use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::ffi::reductions::helpers::{validate_axis, write_scalar};
use crate::ffi::{write_output_metadata, NdArrayHandle, ViewMetadata};
use ndarray::Axis;
use parking_lot::RwLock;
use std::sync::Arc;

/// Compute the minimum of all elements in the array.
#[no_mangle]
pub unsafe extern "C" fn ndarray_min(
    handle: *const NdArrayHandle,
    meta: *const ViewMetadata,
    out_value: *mut c_void,
    out_dtype: *mut u8,
) -> i32 {
    if handle.is_null() || meta.is_null() || out_value.is_null() || out_dtype.is_null() {
        return ERR_GENERIC;
    }

    let meta = &*meta;

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);

        let min_result = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, meta)
                else {
                    crate::error::set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter().cloned().fold(
                    f64::NAN,
                    |min, x| {
                        if min.is_nan() || x < min {
                            x
                        } else {
                            min
                        }
                    },
                )
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, meta)
                else {
                    crate::error::set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter().cloned().fold(
                    f32::NAN,
                    |min, x| {
                        if min.is_nan() || x < min {
                            x
                        } else {
                            min
                        }
                    },
                ) as f64
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, meta)
                else {
                    crate::error::set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .cloned()
                    .fold(i64::MAX, |min, x| if x < min { x } else { min }) as f64
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, meta)
                else {
                    crate::error::set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .cloned()
                    .fold(i32::MAX, |min, x| if x < min { x } else { min }) as f64
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, meta)
                else {
                    crate::error::set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .cloned()
                    .fold(i16::MAX, |min, x| if x < min { x } else { min }) as f64
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, meta)
                else {
                    crate::error::set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .cloned()
                    .fold(i8::MAX, |min, x| if x < min { x } else { min }) as f64
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, meta)
                else {
                    crate::error::set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .cloned()
                    .fold(u64::MAX, |min, x| if x < min { x } else { min }) as f64
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, meta)
                else {
                    crate::error::set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .cloned()
                    .fold(u32::MAX, |min, x| if x < min { x } else { min }) as f64
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, meta)
                else {
                    crate::error::set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .cloned()
                    .fold(u16::MAX, |min, x| if x < min { x } else { min }) as f64
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, meta)
                else {
                    crate::error::set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .cloned()
                    .fold(u8::MAX, |min, x| if x < min { x } else { min }) as f64
            }
            DType::Bool => {
                crate::error::set_last_error("min() not supported for Bool type".to_string());
                return ERR_GENERIC;
            }
        };

        write_scalar(out_value, out_dtype, min_result, wrapper.dtype);
        SUCCESS
    })
}

/// Compute the minimum along an axis.
#[no_mangle]
pub unsafe extern "C" fn ndarray_min_axis(
    handle: *const NdArrayHandle,
    meta: *const ViewMetadata,
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
        let axis_usize = match validate_axis(shape_slice, axis) {
            Ok(a) => a,
            Err(e) => {
                crate::error::set_last_error(e);
                return ERR_SHAPE;
            }
        };

        let axis_len = shape_slice[axis_usize];

        if axis_len == 0 {
            crate::error::set_last_error("Cannot compute min along empty axis".to_string());
            return ERR_GENERIC;
        }

        let result_wrapper = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, meta)
                else {
                    crate::error::set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.fold_axis(Axis(axis_usize), f64::INFINITY, |acc, &x| acc.min(x));
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
                let Some(view) = extract_view_f32(wrapper, meta)
                else {
                    crate::error::set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.fold_axis(Axis(axis_usize), f32::INFINITY, |acc, &x| acc.min(x));
                let final_arr = if keepdims {
                    result.insert_axis(Axis(axis_usize))
                } else {
                    result
                };
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(final_arr))),
                    dtype: DType::Float32,
                }
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, meta)
                else {
                    crate::error::set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                let result =
                    view.fold_axis(Axis(axis_usize), i64::MAX as i64, |&acc, &x| acc.min(x));
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
                let Some(view) = extract_view_i32(wrapper, meta)
                else {
                    crate::error::set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                let result =
                    view.fold_axis(Axis(axis_usize), i32::MAX as i32, |&acc, &x| acc.min(x));
                let final_arr = if keepdims {
                    result.insert_axis(Axis(axis_usize))
                } else {
                    result
                };
                NDArrayWrapper {
                    data: ArrayData::Int32(Arc::new(RwLock::new(final_arr))),
                    dtype: DType::Int32,
                }
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, meta)
                else {
                    crate::error::set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                let result =
                    view.fold_axis(Axis(axis_usize), i16::MAX as i16, |&acc, &x| acc.min(x));
                let final_arr = if keepdims {
                    result.insert_axis(Axis(axis_usize))
                } else {
                    result
                };
                NDArrayWrapper {
                    data: ArrayData::Int16(Arc::new(RwLock::new(final_arr))),
                    dtype: DType::Int16,
                }
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, meta)
                else {
                    crate::error::set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.fold_axis(Axis(axis_usize), i8::MAX as i8, |&acc, &x| acc.min(x));
                let final_arr = if keepdims {
                    result.insert_axis(Axis(axis_usize))
                } else {
                    result
                };
                NDArrayWrapper {
                    data: ArrayData::Int8(Arc::new(RwLock::new(final_arr))),
                    dtype: DType::Int8,
                }
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, meta)
                else {
                    crate::error::set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                let result =
                    view.fold_axis(Axis(axis_usize), u64::MAX as u64, |&acc, &x| acc.min(x));
                let final_arr = if keepdims {
                    result.insert_axis(Axis(axis_usize))
                } else {
                    result
                };
                NDArrayWrapper {
                    data: ArrayData::Uint64(Arc::new(RwLock::new(final_arr))),
                    dtype: DType::Uint64,
                }
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, meta)
                else {
                    crate::error::set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                let result =
                    view.fold_axis(Axis(axis_usize), u32::MAX as u32, |&acc, &x| acc.min(x));
                let final_arr = if keepdims {
                    result.insert_axis(Axis(axis_usize))
                } else {
                    result
                };
                NDArrayWrapper {
                    data: ArrayData::Uint32(Arc::new(RwLock::new(final_arr))),
                    dtype: DType::Uint32,
                }
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, meta)
                else {
                    crate::error::set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                let result =
                    view.fold_axis(Axis(axis_usize), u16::MAX as u16, |&acc, &x| acc.min(x));
                let final_arr = if keepdims {
                    result.insert_axis(Axis(axis_usize))
                } else {
                    result
                };
                NDArrayWrapper {
                    data: ArrayData::Uint16(Arc::new(RwLock::new(final_arr))),
                    dtype: DType::Uint16,
                }
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, meta)
                else {
                    crate::error::set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.fold_axis(Axis(axis_usize), u8::MAX as u8, |&acc, &x| acc.min(x));
                let final_arr = if keepdims {
                    result.insert_axis(Axis(axis_usize))
                } else {
                    result
                };
                NDArrayWrapper {
                    data: ArrayData::Uint8(Arc::new(RwLock::new(final_arr))),
                    dtype: DType::Uint8,
                }
            }
            DType::Bool => {
                crate::error::set_last_error("min_axis() not supported for Bool type".to_string());
                return ERR_GENERIC;
            }
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
