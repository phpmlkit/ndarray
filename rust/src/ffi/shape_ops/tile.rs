//! Tile an array by repeating it along each axis.
//!
//! Matches NumPy's tile function behavior.

use std::sync::Arc;

use ndarray::{concatenate, Axis};
use parking_lot::RwLock;

use crate::core::view_helpers::{
    extract_view_bool, extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32,
    extract_view_i64, extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64,
    extract_view_u8,
};
use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{set_last_error, ERR_GENERIC, SUCCESS};
use crate::ffi::ViewMetadata;
use crate::ffi::{write_output_metadata, NdArrayHandle};

/// Tile an array by repeating it.
#[no_mangle]
pub unsafe extern "C" fn ndarray_tile(
    handle: *const NdArrayHandle,
    meta: *const ViewMetadata,
    reps: *const usize,
    reps_len: usize,
    out_handle: *mut *mut NdArrayHandle,
    out_dtype: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if handle.is_null()
        || meta.is_null()
        || reps.is_null()
        || out_handle.is_null()
        || out_dtype.is_null()
        || out_ndim.is_null()
        || out_shape.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let meta = &*meta;
        let reps_slice = std::slice::from_raw_parts(reps, reps_len);

        let result_wrapper = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, meta)
                else {
                    set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                let arr = view.to_owned();
                let result = tile_array(arr, reps_slice);
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, meta)
                else {
                    set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                let arr = view.to_owned();
                let result = tile_array(arr, reps_slice);
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(result))),
                    dtype: DType::Float32,
                }
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, meta)
                else {
                    set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                let arr = view.to_owned();
                let result = tile_array(arr, reps_slice);
                NDArrayWrapper {
                    data: ArrayData::Int64(Arc::new(RwLock::new(result))),
                    dtype: DType::Int64,
                }
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, meta)
                else {
                    set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                let arr = view.to_owned();
                let result = tile_array(arr, reps_slice);
                NDArrayWrapper {
                    data: ArrayData::Int32(Arc::new(RwLock::new(result))),
                    dtype: DType::Int32,
                }
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, meta)
                else {
                    set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                let arr = view.to_owned();
                let result = tile_array(arr, reps_slice);
                NDArrayWrapper {
                    data: ArrayData::Int16(Arc::new(RwLock::new(result))),
                    dtype: DType::Int16,
                }
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, meta)
                else {
                    set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                let arr = view.to_owned();
                let result = tile_array(arr, reps_slice);
                NDArrayWrapper {
                    data: ArrayData::Int8(Arc::new(RwLock::new(result))),
                    dtype: DType::Int8,
                }
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, meta)
                else {
                    set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                let arr = view.to_owned();
                let result = tile_array(arr, reps_slice);
                NDArrayWrapper {
                    data: ArrayData::Uint64(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint64,
                }
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, meta)
                else {
                    set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                let arr = view.to_owned();
                let result = tile_array(arr, reps_slice);
                NDArrayWrapper {
                    data: ArrayData::Uint32(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint32,
                }
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, meta)
                else {
                    set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                let arr = view.to_owned();
                let result = tile_array(arr, reps_slice);
                NDArrayWrapper {
                    data: ArrayData::Uint16(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint16,
                }
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, meta)
                else {
                    set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                let arr = view.to_owned();
                let result = tile_array(arr, reps_slice);
                NDArrayWrapper {
                    data: ArrayData::Uint8(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint8,
                }
            }
            DType::Bool => {
                let Some(view) = extract_view_bool(wrapper, meta)
                else {
                    set_last_error("Failed to extract bool view".to_string());
                    return ERR_GENERIC;
                };
                let arr = view.to_owned();
                let result = tile_array(arr, reps_slice);
                NDArrayWrapper {
                    data: ArrayData::Bool(Arc::new(RwLock::new(result))),
                    dtype: DType::Bool,
                }
            }
        };

        if let Err(e) =
            write_output_metadata(&result_wrapper, out_dtype, out_ndim, out_shape, max_ndim)
        {
            set_last_error(e);
            return ERR_GENERIC;
        }
        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}

fn tile_array<T: Copy>(mut arr: ndarray::ArrayD<T>, reps: &[usize]) -> ndarray::ArrayD<T> {
    let d = reps.len().max(arr.ndim());

    // Pad array dimensions if needed
    while arr.ndim() < d {
        arr.insert_axis_inplace(Axis(0));
    }

    // Pad reps if needed (prepend 1s)
    let mut padded_reps = vec![1usize; d];
    for (i, &rep) in reps.iter().rev().enumerate() {
        if i < d {
            padded_reps[d - 1 - i] = rep;
        }
    }

    // Tile along each axis in reverse order (last axis first) to match NumPy behavior
    for (axis, &rep) in padded_reps.iter().enumerate().rev() {
        if rep > 1 {
            let views: Vec<_> = (0..rep).map(|_| arr.view()).collect();
            arr = concatenate(Axis(axis), &views)
                .expect("concatenate should succeed")
                .as_standard_layout()
                .into_owned();
        }
    }

    arr
}
