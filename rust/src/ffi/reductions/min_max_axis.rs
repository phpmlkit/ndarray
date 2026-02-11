//! Axis min/max reductions.

use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::reductions::helpers::{compute_axis_output_shape, extract_view, validate_axis};
use crate::ffi::NdArrayHandle;
use ndarray::Axis;
use parking_lot::RwLock;
use std::sync::Arc;

/// Minimum along axis.
#[no_mangle]
pub unsafe extern "C" fn ndarray_min_axis(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    _strides: *const usize,
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
        let shape_slice = std::slice::from_raw_parts(shape, ndim);

        // Validate axis
        let axis_usize = match validate_axis(shape_slice, axis) {
            Ok(a) => a,
            Err(e) => {
                crate::error::set_last_error(e);
                return ERR_GENERIC;
            }
        };

        // Extract view and compute min along axis
        let view = extract_view(wrapper, offset, shape_slice);

        // Use fold_axis to compute minimum
        let axis_len = shape_slice[axis_usize];
        let _out_shape = compute_axis_output_shape(shape_slice, axis_usize, false);

        let result_arr = if axis_len == 0 {
            crate::error::set_last_error("Cannot compute min along empty axis".to_string());
            return ERR_GENERIC;
        } else {
            view.fold_axis(Axis(axis_usize), f64::INFINITY, |acc, &x| acc.min(x))
        };

        // Handle keepdims
        let final_arr = if keepdims {
            result_arr.insert_axis(Axis(axis_usize))
        } else {
            result_arr
        };

        // Create wrapper preserving dtype
        let result_wrapper = match wrapper.dtype {
            DType::Float64 => NDArrayWrapper {
                data: ArrayData::Float64(Arc::new(RwLock::new(final_arr))),
                dtype: DType::Float64,
            },
            DType::Float32 => NDArrayWrapper {
                data: ArrayData::Float32(Arc::new(RwLock::new(final_arr.mapv(|x| x as f32)))),
                dtype: DType::Float32,
            },
            DType::Int64 => NDArrayWrapper {
                data: ArrayData::Int64(Arc::new(RwLock::new(final_arr.mapv(|x| x as i64)))),
                dtype: DType::Int64,
            },
            DType::Int32 => NDArrayWrapper {
                data: ArrayData::Int32(Arc::new(RwLock::new(final_arr.mapv(|x| x as i32)))),
                dtype: DType::Int32,
            },
            DType::Int16 => NDArrayWrapper {
                data: ArrayData::Int16(Arc::new(RwLock::new(final_arr.mapv(|x| x as i16)))),
                dtype: DType::Int16,
            },
            DType::Int8 => NDArrayWrapper {
                data: ArrayData::Int8(Arc::new(RwLock::new(final_arr.mapv(|x| x as i8)))),
                dtype: DType::Int8,
            },
            DType::Uint64 => NDArrayWrapper {
                data: ArrayData::Uint64(Arc::new(RwLock::new(final_arr.mapv(|x| x as u64)))),
                dtype: DType::Uint64,
            },
            DType::Uint32 => NDArrayWrapper {
                data: ArrayData::Uint32(Arc::new(RwLock::new(final_arr.mapv(|x| x as u32)))),
                dtype: DType::Uint32,
            },
            DType::Uint16 => NDArrayWrapper {
                data: ArrayData::Uint16(Arc::new(RwLock::new(final_arr.mapv(|x| x as u16)))),
                dtype: DType::Uint16,
            },
            DType::Uint8 => NDArrayWrapper {
                data: ArrayData::Uint8(Arc::new(RwLock::new(final_arr.mapv(|x| x as u8)))),
                dtype: DType::Uint8,
            },
            DType::Bool => NDArrayWrapper {
                data: ArrayData::Bool(Arc::new(RwLock::new(final_arr.mapv(|x| {
                    if x != 0.0 {
                        1
                    } else {
                        0
                    }
                })))),
                dtype: DType::Bool,
            },
        };

        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}

/// Maximum along axis.
#[no_mangle]
pub unsafe extern "C" fn ndarray_max_axis(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    _strides: *const usize,
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
        let shape_slice = std::slice::from_raw_parts(shape, ndim);

        // Validate axis
        let axis_usize = match validate_axis(shape_slice, axis) {
            Ok(a) => a,
            Err(e) => {
                crate::error::set_last_error(e);
                return ERR_GENERIC;
            }
        };

        // Extract view and compute max along axis
        let view = extract_view(wrapper, offset, shape_slice);

        let axis_len = shape_slice[axis_usize];

        let result_arr = if axis_len == 0 {
            crate::error::set_last_error("Cannot compute max along empty axis".to_string());
            return ERR_GENERIC;
        } else {
            view.fold_axis(Axis(axis_usize), f64::NEG_INFINITY, |acc, &x| acc.max(x))
        };

        // Handle keepdims
        let final_arr = if keepdims {
            result_arr.insert_axis(Axis(axis_usize))
        } else {
            result_arr
        };

        // Create wrapper preserving dtype
        let result_wrapper = match wrapper.dtype {
            DType::Float64 => NDArrayWrapper {
                data: ArrayData::Float64(Arc::new(RwLock::new(final_arr))),
                dtype: DType::Float64,
            },
            DType::Float32 => NDArrayWrapper {
                data: ArrayData::Float32(Arc::new(RwLock::new(final_arr.mapv(|x| x as f32)))),
                dtype: DType::Float32,
            },
            DType::Int64 => NDArrayWrapper {
                data: ArrayData::Int64(Arc::new(RwLock::new(final_arr.mapv(|x| x as i64)))),
                dtype: DType::Int64,
            },
            DType::Int32 => NDArrayWrapper {
                data: ArrayData::Int32(Arc::new(RwLock::new(final_arr.mapv(|x| x as i32)))),
                dtype: DType::Int32,
            },
            DType::Int16 => NDArrayWrapper {
                data: ArrayData::Int16(Arc::new(RwLock::new(final_arr.mapv(|x| x as i16)))),
                dtype: DType::Int16,
            },
            DType::Int8 => NDArrayWrapper {
                data: ArrayData::Int8(Arc::new(RwLock::new(final_arr.mapv(|x| x as i8)))),
                dtype: DType::Int8,
            },
            DType::Uint64 => NDArrayWrapper {
                data: ArrayData::Uint64(Arc::new(RwLock::new(final_arr.mapv(|x| x as u64)))),
                dtype: DType::Uint64,
            },
            DType::Uint32 => NDArrayWrapper {
                data: ArrayData::Uint32(Arc::new(RwLock::new(final_arr.mapv(|x| x as u32)))),
                dtype: DType::Uint32,
            },
            DType::Uint16 => NDArrayWrapper {
                data: ArrayData::Uint16(Arc::new(RwLock::new(final_arr.mapv(|x| x as u16)))),
                dtype: DType::Uint16,
            },
            DType::Uint8 => NDArrayWrapper {
                data: ArrayData::Uint8(Arc::new(RwLock::new(final_arr.mapv(|x| x as u8)))),
                dtype: DType::Uint8,
            },
            DType::Bool => NDArrayWrapper {
                data: ArrayData::Bool(Arc::new(RwLock::new(final_arr.mapv(|x| {
                    if x != 0.0 {
                        1
                    } else {
                        0
                    }
                })))),
                dtype: DType::Bool,
            },
        };

        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
