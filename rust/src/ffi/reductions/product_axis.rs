//! Axis product reduction.

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
use std::sync::Arc;

/// Compute product along axis using ndarray's native product_axis method.
///
/// Works with all types that implement Mul + One traits.
fn product_axis_view(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: usize,
) -> ndarray::ArrayD<f64> {
    unsafe {
        // Float64 - native support
        if let Some(view) = extract_view_f64(wrapper, offset, shape, strides) {
            return view.product_axis(Axis(axis));
        }
        // Float32 - native support, convert result to f64
        if let Some(view) = extract_view_f32(wrapper, offset, shape, strides) {
            return view.product_axis(Axis(axis)).mapv(|x| x as f64);
        }
        // Int64 - native support via Mul + One
        if let Some(view) = extract_view_i64(wrapper, offset, shape, strides) {
            return view.product_axis(Axis(axis)).mapv(|x| x as f64);
        }
        // Int32
        if let Some(view) = extract_view_i32(wrapper, offset, shape, strides) {
            return view.product_axis(Axis(axis)).mapv(|x| x as f64);
        }
        // Int16
        if let Some(view) = extract_view_i16(wrapper, offset, shape, strides) {
            return view.product_axis(Axis(axis)).mapv(|x| x as f64);
        }
        // Int8
        if let Some(view) = extract_view_i8(wrapper, offset, shape, strides) {
            return view.product_axis(Axis(axis)).mapv(|x| x as f64);
        }
        // Uint64
        if let Some(view) = extract_view_u64(wrapper, offset, shape, strides) {
            return view.product_axis(Axis(axis)).mapv(|x| x as f64);
        }
        // Uint32
        if let Some(view) = extract_view_u32(wrapper, offset, shape, strides) {
            return view.product_axis(Axis(axis)).mapv(|x| x as f64);
        }
        // Uint16
        if let Some(view) = extract_view_u16(wrapper, offset, shape, strides) {
            return view.product_axis(Axis(axis)).mapv(|x| x as f64);
        }
        // Uint8
        if let Some(view) = extract_view_u8(wrapper, offset, shape, strides) {
            return view.product_axis(Axis(axis)).mapv(|x| x as f64);
        }
    }
    // Return empty array if no type matched
    ndarray::ArrayD::zeros(ndarray::IxDyn(&[]))
}

/// Product along axis.
#[no_mangle]
pub unsafe extern "C" fn ndarray_product_axis(
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
        let shape_slice = std::slice::from_raw_parts(shape, ndim);
        let strides_slice = std::slice::from_raw_parts(strides, ndim);

        // Validate axis
        let axis_usize = match validate_axis(shape_slice, axis) {
            Ok(a) => a,
            Err(e) => {
                crate::error::set_last_error(e);
                return ERR_GENERIC;
            }
        };

        // Compute product along axis using proper strided view
        let result_arr = product_axis_view(wrapper, offset, shape_slice, strides_slice, axis_usize);

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
