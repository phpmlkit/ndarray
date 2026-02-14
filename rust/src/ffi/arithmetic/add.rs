//! Addition operations.

use crate::core::math_helpers::{
    binary_op_f32, binary_op_f64, binary_op_i16, binary_op_i32, binary_op_i64, binary_op_i8,
    binary_op_u16, binary_op_u32, binary_op_u64, binary_op_u8, scalar_op_f32, scalar_op_f64,
    scalar_op_i16, scalar_op_i32, scalar_op_i64, scalar_op_i8, scalar_op_u16, scalar_op_u32,
    scalar_op_u64, scalar_op_u8,
};
use crate::dtype::DType;
use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::NdArrayHandle;
use std::slice;

/// Add two arrays element-wise.
#[no_mangle]
pub unsafe extern "C" fn ndarray_add(
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
                |a, b| a + b,
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
                |a, b| a + b,
            ),
            DType::Int64 => binary_op_i64(
                a_wrapper,
                a_offset,
                a_shape_slice,
                a_strides_slice,
                b_wrapper,
                b_offset,
                b_shape_slice,
                b_strides_slice,
                |a, b| a + b,
            ),
            DType::Int32 => binary_op_i32(
                a_wrapper,
                a_offset,
                a_shape_slice,
                a_strides_slice,
                b_wrapper,
                b_offset,
                b_shape_slice,
                b_strides_slice,
                |a, b| a + b,
            ),
            DType::Int16 => binary_op_i16(
                a_wrapper,
                a_offset,
                a_shape_slice,
                a_strides_slice,
                b_wrapper,
                b_offset,
                b_shape_slice,
                b_strides_slice,
                |a, b| a + b,
            ),
            DType::Int8 => binary_op_i8(
                a_wrapper,
                a_offset,
                a_shape_slice,
                a_strides_slice,
                b_wrapper,
                b_offset,
                b_shape_slice,
                b_strides_slice,
                |a, b| a + b,
            ),
            DType::Uint64 => binary_op_u64(
                a_wrapper,
                a_offset,
                a_shape_slice,
                a_strides_slice,
                b_wrapper,
                b_offset,
                b_shape_slice,
                b_strides_slice,
                |a, b| a + b,
            ),
            DType::Uint32 => binary_op_u32(
                a_wrapper,
                a_offset,
                a_shape_slice,
                a_strides_slice,
                b_wrapper,
                b_offset,
                b_shape_slice,
                b_strides_slice,
                |a, b| a + b,
            ),
            DType::Uint16 => binary_op_u16(
                a_wrapper,
                a_offset,
                a_shape_slice,
                a_strides_slice,
                b_wrapper,
                b_offset,
                b_shape_slice,
                b_strides_slice,
                |a, b| a + b,
            ),
            DType::Uint8 => binary_op_u8(
                a_wrapper,
                a_offset,
                a_shape_slice,
                a_strides_slice,
                b_wrapper,
                b_offset,
                b_shape_slice,
                b_strides_slice,
                |a, b| a + b,
            ),
            DType::Bool => Err("Addition not supported for Bool type".to_string()),
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

/// Add scalar to array.
#[no_mangle]
pub unsafe extern "C" fn ndarray_add_scalar(
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

        let result = match a_wrapper.dtype {
            DType::Float64 => scalar_op_f64(
                a_wrapper,
                a_offset,
                a_shape_slice,
                a_strides_slice,
                scalar as f64,
                |a, b| a + b,
            ),
            DType::Float32 => scalar_op_f32(
                a_wrapper,
                a_offset,
                a_shape_slice,
                a_strides_slice,
                scalar as f32,
                |a, b| a + b,
            ),
            DType::Int64 => scalar_op_i64(
                a_wrapper,
                a_offset,
                a_shape_slice,
                a_strides_slice,
                scalar as i64,
                |a, b| a + b,
            ),
            DType::Int32 => scalar_op_i32(
                a_wrapper,
                a_offset,
                a_shape_slice,
                a_strides_slice,
                scalar as i32,
                |a, b| a + b,
            ),
            DType::Int16 => scalar_op_i16(
                a_wrapper,
                a_offset,
                a_shape_slice,
                a_strides_slice,
                scalar as i16,
                |a, b| a + b,
            ),
            DType::Int8 => scalar_op_i8(
                a_wrapper,
                a_offset,
                a_shape_slice,
                a_strides_slice,
                scalar as i8,
                |a, b| a + b,
            ),
            DType::Uint64 => scalar_op_u64(
                a_wrapper,
                a_offset,
                a_shape_slice,
                a_strides_slice,
                scalar as u64,
                |a, b| a + b,
            ),
            DType::Uint32 => scalar_op_u32(
                a_wrapper,
                a_offset,
                a_shape_slice,
                a_strides_slice,
                scalar as u32,
                |a, b| a + b,
            ),
            DType::Uint16 => scalar_op_u16(
                a_wrapper,
                a_offset,
                a_shape_slice,
                a_strides_slice,
                scalar as u16,
                |a, b| a + b,
            ),
            DType::Uint8 => scalar_op_u8(
                a_wrapper,
                a_offset,
                a_shape_slice,
                a_strides_slice,
                scalar as u8,
                |a, b| a + b,
            ),
            DType::Bool => Err("Addition not supported for Bool type".to_string()),
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
