//! Flatten and ravel operations.

use crate::core::view_helpers::{
    extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32, extract_view_i64,
    extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64, extract_view_u8,
};
use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{self, ERR_GENERIC, SUCCESS};
use crate::ffi::NdArrayHandle;
use ndarray::Order;
use parking_lot::RwLock;
use std::slice;
use std::sync::Arc;

/// Flatten array to 1D.
///
/// Always returns a copy in C-order (row-major).
#[no_mangle]
pub unsafe extern "C" fn ndarray_flatten(
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

        // Match on dtype, extract view, flatten, and create result wrapper
        let result_wrapper = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                let flat = view.flatten().into_owned().into_dyn();
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(flat))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                let flat = view.flatten().into_owned().into_dyn();
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(flat))),
                    dtype: DType::Float32,
                }
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                let flat = view.flatten().into_owned().into_dyn();
                NDArrayWrapper {
                    data: ArrayData::Int64(Arc::new(RwLock::new(flat))),
                    dtype: DType::Int64,
                }
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                let flat = view.flatten().into_owned().into_dyn();
                NDArrayWrapper {
                    data: ArrayData::Int32(Arc::new(RwLock::new(flat))),
                    dtype: DType::Int32,
                }
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                let flat = view.flatten().into_owned().into_dyn();
                NDArrayWrapper {
                    data: ArrayData::Int16(Arc::new(RwLock::new(flat))),
                    dtype: DType::Int16,
                }
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                let flat = view.flatten().into_owned().into_dyn();
                NDArrayWrapper {
                    data: ArrayData::Int8(Arc::new(RwLock::new(flat))),
                    dtype: DType::Int8,
                }
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                let flat = view.flatten().into_owned().into_dyn();
                NDArrayWrapper {
                    data: ArrayData::Uint64(Arc::new(RwLock::new(flat))),
                    dtype: DType::Uint64,
                }
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                let flat = view.flatten().into_owned().into_dyn();
                NDArrayWrapper {
                    data: ArrayData::Uint32(Arc::new(RwLock::new(flat))),
                    dtype: DType::Uint32,
                }
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                let flat = view.flatten().into_owned().into_dyn();
                NDArrayWrapper {
                    data: ArrayData::Uint16(Arc::new(RwLock::new(flat))),
                    dtype: DType::Uint16,
                }
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                let flat = view.flatten().into_owned().into_dyn();
                NDArrayWrapper {
                    data: ArrayData::Uint8(Arc::new(RwLock::new(flat))),
                    dtype: DType::Uint8,
                }
            }
            DType::Bool => {
                error::set_last_error("flatten() not supported for Bool type".to_string());
                return ERR_GENERIC;
            }
        };

        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}

/// Ravel array to 1D.
///
/// Similar to flatten but may return a view if the array is already contiguous.
/// For FFI simplicity, we always return a copy.
#[no_mangle]
pub unsafe extern "C" fn ndarray_ravel(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
    order: i32,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || out_handle.is_null() || shape.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = slice::from_raw_parts(shape, ndim);
        let strides_slice = slice::from_raw_parts(strides, ndim);

        let order = if order == 1 {
            Order::ColumnMajor
        } else {
            Order::RowMajor
        };

        // Match on dtype, extract view, ravel with order, and create result wrapper
        let result_wrapper = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                let flat = view.flatten_with_order(order).into_owned().into_dyn();
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(flat))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                let flat = view.flatten_with_order(order).into_owned().into_dyn();
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(flat))),
                    dtype: DType::Float32,
                }
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                let flat = view.flatten_with_order(order).into_owned().into_dyn();
                NDArrayWrapper {
                    data: ArrayData::Int64(Arc::new(RwLock::new(flat))),
                    dtype: DType::Int64,
                }
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                let flat = view.flatten_with_order(order).into_owned().into_dyn();
                NDArrayWrapper {
                    data: ArrayData::Int32(Arc::new(RwLock::new(flat))),
                    dtype: DType::Int32,
                }
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                let flat = view.flatten_with_order(order).into_owned().into_dyn();
                NDArrayWrapper {
                    data: ArrayData::Int16(Arc::new(RwLock::new(flat))),
                    dtype: DType::Int16,
                }
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                let flat = view.flatten_with_order(order).into_owned().into_dyn();
                NDArrayWrapper {
                    data: ArrayData::Int8(Arc::new(RwLock::new(flat))),
                    dtype: DType::Int8,
                }
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                let flat = view.flatten_with_order(order).into_owned().into_dyn();
                NDArrayWrapper {
                    data: ArrayData::Uint64(Arc::new(RwLock::new(flat))),
                    dtype: DType::Uint64,
                }
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                let flat = view.flatten_with_order(order).into_owned().into_dyn();
                NDArrayWrapper {
                    data: ArrayData::Uint32(Arc::new(RwLock::new(flat))),
                    dtype: DType::Uint32,
                }
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                let flat = view.flatten_with_order(order).into_owned().into_dyn();
                NDArrayWrapper {
                    data: ArrayData::Uint16(Arc::new(RwLock::new(flat))),
                    dtype: DType::Uint16,
                }
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, offset, shape_slice, strides_slice)
                else {
                    error::set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                let flat = view.flatten_with_order(order).into_owned().into_dyn();
                NDArrayWrapper {
                    data: ArrayData::Uint8(Arc::new(RwLock::new(flat))),
                    dtype: DType::Uint8,
                }
            }
            DType::Bool => {
                error::set_last_error("ravel() not supported for Bool type".to_string());
                return ERR_GENERIC;
            }
        };

        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
