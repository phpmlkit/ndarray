//! Scalar min/max reductions using manual iteration.

use crate::core::view_helpers::{
    extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32, extract_view_i64,
    extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64, extract_view_u8,
};
use crate::core::NDArrayWrapper;
use crate::dtype::DType;
use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::NdArrayHandle;
use std::slice;

/// Compute the minimum of all elements in the array (scalar reduction).
#[no_mangle]
pub unsafe extern "C" fn ndarray_min(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || out_handle.is_null() || shape.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = slice::from_raw_parts(shape, ndim);
        let strides_slice = slice::from_raw_parts(strides, ndim);

        // Match on dtype, extract view, compute min, and create result wrapper
        let min_result = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, offset, shape_slice, strides_slice)
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
                let Some(view) = extract_view_f32(wrapper, offset, shape_slice, strides_slice)
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
                let Some(view) = extract_view_i64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .cloned()
                    .fold(i64::MAX, |min, x| if x < min { x } else { min }) as f64
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .cloned()
                    .fold(i32::MAX, |min, x| if x < min { x } else { min }) as f64
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .cloned()
                    .fold(i16::MAX, |min, x| if x < min { x } else { min }) as f64
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .cloned()
                    .fold(i8::MAX, |min, x| if x < min { x } else { min }) as f64
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .cloned()
                    .fold(u64::MAX, |min, x| if x < min { x } else { min }) as f64
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .cloned()
                    .fold(u32::MAX, |min, x| if x < min { x } else { min }) as f64
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .cloned()
                    .fold(u16::MAX, |min, x| if x < min { x } else { min }) as f64
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, offset, shape_slice, strides_slice)
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

        let result_wrapper = NDArrayWrapper::create_scalar_wrapper(min_result, wrapper.dtype);
        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));

        SUCCESS
    })
}

/// Compute the maximum of all elements in the array (scalar reduction).
#[no_mangle]
pub unsafe extern "C" fn ndarray_max(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || out_handle.is_null() || shape.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = slice::from_raw_parts(shape, ndim);
        let strides_slice = slice::from_raw_parts(strides, ndim);

        // Match on dtype, extract view, compute max, and create result wrapper
        let max_result = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter().cloned().fold(
                    f64::NAN,
                    |max, x| {
                        if max.is_nan() || x > max {
                            x
                        } else {
                            max
                        }
                    },
                )
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter().cloned().fold(
                    f32::NAN,
                    |max, x| {
                        if max.is_nan() || x > max {
                            x
                        } else {
                            max
                        }
                    },
                ) as f64
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .cloned()
                    .fold(i64::MIN, |max, x| if x > max { x } else { max }) as f64
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .cloned()
                    .fold(i32::MIN, |max, x| if x > max { x } else { max }) as f64
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .cloned()
                    .fold(i16::MIN, |max, x| if x > max { x } else { max }) as f64
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .cloned()
                    .fold(i8::MIN, |max, x| if x > max { x } else { max }) as f64
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .cloned()
                    .fold(u64::MIN, |max, x| if x > max { x } else { max }) as f64
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .cloned()
                    .fold(u32::MIN, |max, x| if x > max { x } else { max }) as f64
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .cloned()
                    .fold(u16::MIN, |max, x| if x > max { x } else { max }) as f64
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                view.iter()
                    .cloned()
                    .fold(u8::MIN, |max, x| if x > max { x } else { max }) as f64
            }
            DType::Bool => {
                crate::error::set_last_error("max() not supported for Bool type".to_string());
                return ERR_GENERIC;
            }
        };

        let result_wrapper = NDArrayWrapper::create_scalar_wrapper(max_result, wrapper.dtype);
        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));

        SUCCESS
    })
}
