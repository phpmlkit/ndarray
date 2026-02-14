//! Hypotenuse operation.

use crate::core::math_helpers::{binary_op_f32, binary_op_f64, scalar_op_f32, scalar_op_f64};
use crate::dtype::DType;
use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::NdArrayHandle;
use std::slice;

/// Compute hypot(a, b) = sqrt(a^2 + b^2) element-wise.
#[no_mangle]
pub unsafe extern "C" fn ndarray_hypot(
    a: *const NdArrayHandle,
    a_offset: usize,
    a_shape: *const usize,
    a_strides: *const usize,
    b: *const NdArrayHandle,
    b_offset: usize,
    b_shape: *const usize,
    b_strides: *const usize,
    ndim: usize,
    out: *mut *mut NdArrayHandle,
) -> i32 {
    if a.is_null() || b.is_null() || out.is_null() || a_shape.is_null() || b_shape.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let a_wrapper = NdArrayHandle::as_wrapper(a as *mut _);
        let b_wrapper = NdArrayHandle::as_wrapper(b as *mut _);

        let a_shape_slice = slice::from_raw_parts(a_shape, ndim);
        let b_shape_slice = slice::from_raw_parts(b_shape, ndim);
        let a_strides_slice = slice::from_raw_parts(a_strides, ndim);
        let b_strides_slice = slice::from_raw_parts(b_strides, ndim);

        let out_dtype = DType::promote(a_wrapper.dtype, b_wrapper.dtype);

        // hypot only makes sense for float types
        let result = match out_dtype {
            DType::Float64 => binary_op_f64(
                a_wrapper,
                a_offset,
                a_shape_slice,
                a_strides_slice,
                b_wrapper,
                b_offset,
                b_shape_slice,
                b_strides_slice,
                |a, b| a.hypot(b),
            ),
            DType::Float32 => binary_op_f32(
                a_wrapper,
                a_offset,
                a_shape_slice,
                a_strides_slice,
                b_wrapper,
                b_offset,
                b_shape_slice,
                b_strides_slice,
                |a, b| a.hypot(b),
            ),
            _ => Err("hypot() requires float type (Float64 or Float32)".to_string()),
        };

        match result {
            Ok(wrapper) => {
                *out = NdArrayHandle::from_wrapper(Box::new(wrapper));
                SUCCESS
            }
            Err(_) => ERR_GENERIC,
        }
    })
}

/// Compute hypot(array, scalar) element-wise.
#[no_mangle]
pub unsafe extern "C" fn ndarray_hypot_scalar(
    a: *const NdArrayHandle,
    a_offset: usize,
    a_shape: *const usize,
    a_strides: *const usize,
    ndim: usize,
    scalar: f64,
    out: *mut *mut NdArrayHandle,
) -> i32 {
    if a.is_null() || out.is_null() || a_shape.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let a_wrapper = NdArrayHandle::as_wrapper(a as *mut _);
        let a_shape_slice = slice::from_raw_parts(a_shape, ndim);
        let a_strides_slice = slice::from_raw_parts(a_strides, ndim);

        // hypot only makes sense for float types
        let result = match a_wrapper.dtype {
            DType::Float64 => scalar_op_f64(
                a_wrapper,
                a_offset,
                a_shape_slice,
                a_strides_slice,
                scalar as f64,
                |a, b| a.hypot(b),
            ),
            DType::Float32 => scalar_op_f32(
                a_wrapper,
                a_offset,
                a_shape_slice,
                a_strides_slice,
                scalar as f32,
                |a, b| a.hypot(b),
            ),
            _ => Err("hypot() with scalar requires float type (Float64 or Float32)".to_string()),
        };

        match result {
            Ok(wrapper) => {
                *out = NdArrayHandle::from_wrapper(Box::new(wrapper));
                SUCCESS
            }
            Err(_) => ERR_GENERIC,
        }
    })
}
