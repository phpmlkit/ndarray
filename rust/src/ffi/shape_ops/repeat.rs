//! Repeat elements of an array.
//!
//! Matches NumPy's repeat function behavior.

use std::sync::Arc;

use ndarray::{ArrayD, Axis, IxDyn};
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

/// Repeat elements of an array.
#[no_mangle]
pub unsafe extern "C" fn ndarray_repeat(
    handle: *const NdArrayHandle,
    meta: *const ViewMetadata,
    repeats: *const usize,
    repeats_len: usize,
    axis: i32,
    out_handle: *mut *mut NdArrayHandle,
    out_dtype: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if handle.is_null()
        || meta.is_null()
        || repeats.is_null()
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
        let repeats_slice = std::slice::from_raw_parts(repeats, repeats_len);

        let result_wrapper = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, meta) else {
                    set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                let arr = view.to_owned();
                let result = repeat_array(arr, repeats_slice, axis);
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, meta) else {
                    set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                let arr = view.to_owned();
                let result = repeat_array(arr, repeats_slice, axis);
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(result))),
                    dtype: DType::Float32,
                }
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, meta) else {
                    set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                let arr = view.to_owned();
                let result = repeat_array(arr, repeats_slice, axis);
                NDArrayWrapper {
                    data: ArrayData::Int64(Arc::new(RwLock::new(result))),
                    dtype: DType::Int64,
                }
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, meta) else {
                    set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                let arr = view.to_owned();
                let result = repeat_array(arr, repeats_slice, axis);
                NDArrayWrapper {
                    data: ArrayData::Int32(Arc::new(RwLock::new(result))),
                    dtype: DType::Int32,
                }
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, meta) else {
                    set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                let arr = view.to_owned();
                let result = repeat_array(arr, repeats_slice, axis);
                NDArrayWrapper {
                    data: ArrayData::Int16(Arc::new(RwLock::new(result))),
                    dtype: DType::Int16,
                }
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, meta) else {
                    set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                let arr = view.to_owned();
                let result = repeat_array(arr, repeats_slice, axis);
                NDArrayWrapper {
                    data: ArrayData::Int8(Arc::new(RwLock::new(result))),
                    dtype: DType::Int8,
                }
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, meta) else {
                    set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                let arr = view.to_owned();
                let result = repeat_array(arr, repeats_slice, axis);
                NDArrayWrapper {
                    data: ArrayData::Uint64(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint64,
                }
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, meta) else {
                    set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                let arr = view.to_owned();
                let result = repeat_array(arr, repeats_slice, axis);
                NDArrayWrapper {
                    data: ArrayData::Uint32(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint32,
                }
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, meta) else {
                    set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                let arr = view.to_owned();
                let result = repeat_array(arr, repeats_slice, axis);
                NDArrayWrapper {
                    data: ArrayData::Uint16(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint16,
                }
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, meta) else {
                    set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                let arr = view.to_owned();
                let result = repeat_array(arr, repeats_slice, axis);
                NDArrayWrapper {
                    data: ArrayData::Uint8(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint8,
                }
            }
            DType::Bool => {
                let Some(view) = extract_view_bool(wrapper, meta) else {
                    set_last_error("Failed to extract bool view".to_string());
                    return ERR_GENERIC;
                };
                let arr = view.to_owned();
                let result = repeat_array(arr, repeats_slice, axis);
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

fn repeat_array<T: Copy>(arr: ArrayD<T>, repeats: &[usize], axis: i32) -> ArrayD<T> {
    let is_scalar_repeat = repeats.len() == 1;

    if axis < 0 {
        // Flatten and repeat each element
        let flat: Vec<T> = arr.iter().copied().collect();
        let mut result = Vec::new();

        if is_scalar_repeat {
            let rep = repeats[0];
            for val in flat {
                for _ in 0..rep {
                    result.push(val);
                }
            }
        } else {
            assert_eq!(
                repeats.len(),
                flat.len(),
                "repeats length must match number of elements"
            );
            for (i, val) in flat.iter().enumerate() {
                for _ in 0..repeats[i] {
                    result.push(*val);
                }
            }
        }

        ArrayD::from_shape_vec(IxDyn(&[result.len()]), result).unwrap()
    } else {
        let axis_usize = axis as usize;
        assert!(axis_usize < arr.ndim(), "axis out of bounds");

        let axis_len = arr.shape()[axis_usize];

        if is_scalar_repeat {
            // Repeat each element along axis the same number of times
            let rep = repeats[0];
            if rep <= 1 {
                return arr;
            }

            // Collect all owned subviews, re-insert axis, then create views
            let mut expanded_subviews: Vec<_> = Vec::new();
            for subview in arr.axis_iter(Axis(axis_usize)) {
                let owned = subview.to_owned();
                // Insert the axis back so we can concatenate along it
                let with_axis = owned.insert_axis(Axis(axis_usize));
                expanded_subviews.push(with_axis);
            }

            // Now create views from the owned data
            let mut result_slices: Vec<_> = Vec::new();
            for expanded in &expanded_subviews {
                for _ in 0..rep {
                    result_slices.push(expanded.view());
                }
            }

            if result_slices.is_empty() {
                arr
            } else {
                ndarray::concatenate(Axis(axis_usize), &result_slices)
                    .expect("concatenate should succeed")
                    .as_standard_layout()
                    .into_owned()
            }
        } else {
            // Different repeat count for each element along axis
            assert_eq!(
                repeats.len(),
                axis_len,
                "repeats length must match axis length"
            );

            // Collect all owned subviews, re-insert axis, then create views
            let mut expanded_subviews: Vec<_> = Vec::new();
            for subview in arr.axis_iter(Axis(axis_usize)) {
                let owned = subview.to_owned();
                // Insert the axis back so we can concatenate along it
                let with_axis = owned.insert_axis(Axis(axis_usize));
                expanded_subviews.push(with_axis);
            }

            // Now create views from the owned data
            let mut result_slices: Vec<_> = Vec::new();
            for (i, expanded) in expanded_subviews.iter().enumerate() {
                for _ in 0..repeats[i] {
                    result_slices.push(expanded.view());
                }
            }

            if result_slices.is_empty() {
                arr
            } else {
                ndarray::concatenate(Axis(axis_usize), &result_slices)
                    .expect("concatenate should succeed")
                    .as_standard_layout()
                    .into_owned()
            }
        }
    }
}
