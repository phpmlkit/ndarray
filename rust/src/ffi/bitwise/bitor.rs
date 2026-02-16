//! Bitwise OR operation using ndarray's native broadcasting.
//!
//! Works with all integer types (signed and unsigned) and Bool.

use crate::binary_op_arm;
use crate::core::view_helpers::{
    extract_view_as_bool, extract_view_as_i16, extract_view_as_i32, extract_view_as_i64,
    extract_view_as_i8, extract_view_as_u16, extract_view_as_u32, extract_view_as_u64,
    extract_view_as_u8,
};
use crate::core::view_helpers::{
    extract_view_bool, extract_view_i16, extract_view_i32, extract_view_i64, extract_view_i8,
    extract_view_u16, extract_view_u32, extract_view_u64, extract_view_u8,
};
use crate::core::ArrayData;
use crate::dtype::DType;
use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::NdArrayHandle;
use crate::scalar_op_arm;

use std::slice;

/// Bitwise OR with proper broadcasting support.
#[no_mangle]
pub unsafe extern "C" fn ndarray_bitor(
    a: *const NdArrayHandle,
    a_offset: usize,
    a_shape: *const usize,
    a_strides: *const usize,
    a_ndim: usize,
    b: *const NdArrayHandle,
    b_offset: usize,
    b_shape: *const usize,
    b_strides: *const usize,
    b_ndim: usize,
    out: *mut *mut NdArrayHandle,
    out_shape: *mut *mut usize,
    out_ndim: *mut usize,
) -> i32 {
    if a.is_null()
        || b.is_null()
        || out.is_null()
        || out_shape.is_null()
        || out_ndim.is_null()
        || a_shape.is_null()
        || b_shape.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let a_wrapper = NdArrayHandle::as_wrapper(a as *mut _);
        let b_wrapper = NdArrayHandle::as_wrapper(b as *mut _);

        let a_shape_slice = slice::from_raw_parts(a_shape, a_ndim);
        let b_shape_slice = slice::from_raw_parts(b_shape, b_ndim);
        let a_strides_slice = slice::from_raw_parts(a_strides, a_ndim);
        let b_strides_slice = slice::from_raw_parts(b_strides, b_ndim);

        let out_dtype = DType::promote(a_wrapper.dtype, b_wrapper.dtype);

        let (result_wrapper, result_shape) = match out_dtype {
            DType::Int64 => binary_op_arm!(
                a_wrapper, a_offset, a_shape_slice, a_strides_slice,
                b_wrapper, b_offset, b_shape_slice, b_strides_slice,
                DType::Int64, extract_view_as_i64, ArrayData::Int64, |
            ),
            DType::Int32 => binary_op_arm!(
                a_wrapper, a_offset, a_shape_slice, a_strides_slice,
                b_wrapper, b_offset, b_shape_slice, b_strides_slice,
                DType::Int32, extract_view_as_i32, ArrayData::Int32, |
            ),
            DType::Int16 => binary_op_arm!(
                a_wrapper, a_offset, a_shape_slice, a_strides_slice,
                b_wrapper, b_offset, b_shape_slice, b_strides_slice,
                DType::Int16, extract_view_as_i16, ArrayData::Int16, |
            ),
            DType::Int8 => binary_op_arm!(
                a_wrapper, a_offset, a_shape_slice, a_strides_slice,
                b_wrapper, b_offset, b_shape_slice, b_strides_slice,
                DType::Int8, extract_view_as_i8, ArrayData::Int8, |
            ),
            DType::Uint64 => binary_op_arm!(
                a_wrapper, a_offset, a_shape_slice, a_strides_slice,
                b_wrapper, b_offset, b_shape_slice, b_strides_slice,
                DType::Uint64, extract_view_as_u64, ArrayData::Uint64, |
            ),
            DType::Uint32 => binary_op_arm!(
                a_wrapper, a_offset, a_shape_slice, a_strides_slice,
                b_wrapper, b_offset, b_shape_slice, b_strides_slice,
                DType::Uint32, extract_view_as_u32, ArrayData::Uint32, |
            ),
            DType::Uint16 => binary_op_arm!(
                a_wrapper, a_offset, a_shape_slice, a_strides_slice,
                b_wrapper, b_offset, b_shape_slice, b_strides_slice,
                DType::Uint16, extract_view_as_u16, ArrayData::Uint16, |
            ),
            DType::Uint8 => binary_op_arm!(
                a_wrapper, a_offset, a_shape_slice, a_strides_slice,
                b_wrapper, b_offset, b_shape_slice, b_strides_slice,
                DType::Uint8, extract_view_as_u8, ArrayData::Uint8, |
            ),
            DType::Bool => binary_op_arm!(
                a_wrapper, a_offset, a_shape_slice, a_strides_slice,
                b_wrapper, b_offset, b_shape_slice, b_strides_slice,
                DType::Bool, extract_view_as_bool, ArrayData::Bool, |
            ),
            DType::Float64 | DType::Float32 => {
                crate::error::set_last_error(
                    "Bitwise OR not supported for float types".to_string(),
                );
                return ERR_GENERIC;
            }
        };

        let ndim = result_shape.len();
        let shape_box: Box<[usize]> = result_shape.into_boxed_slice();
        let shape_ptr = Box::into_raw(shape_box) as *mut usize;

        *out_ndim = ndim;
        *out_shape = shape_ptr;
        *out = NdArrayHandle::from_wrapper(Box::new(result_wrapper));

        SUCCESS
    })
}

/// Bitwise OR with scalar.
#[no_mangle]
pub unsafe extern "C" fn ndarray_bitor_scalar(
    a: *const NdArrayHandle,
    a_offset: usize,
    a_shape: *const usize,
    a_strides: *const usize,
    ndim: usize,
    scalar: i64,
    out: *mut *mut NdArrayHandle,
) -> i32 {
    if a.is_null() || out.is_null() || a_shape.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let a_wrapper = NdArrayHandle::as_wrapper(a as *mut _);
        let a_shape_slice = slice::from_raw_parts(a_shape, ndim);
        let a_strides_slice = slice::from_raw_parts(a_strides, ndim);

        let result_wrapper = match a_wrapper.dtype {
            DType::Int64 => scalar_op_arm!(
                a_wrapper, a_offset, a_shape_slice, a_strides_slice,
                scalar, DType::Int64, extract_view_i64, ArrayData::Int64, |
            ),
            DType::Int32 => scalar_op_arm!(
                a_wrapper, a_offset, a_shape_slice, a_strides_slice,
                scalar as i32, DType::Int32, extract_view_i32, ArrayData::Int32, |
            ),
            DType::Int16 => scalar_op_arm!(
                a_wrapper, a_offset, a_shape_slice, a_strides_slice,
                scalar as i16, DType::Int16, extract_view_i16, ArrayData::Int16, |
            ),
            DType::Int8 => scalar_op_arm!(
                a_wrapper, a_offset, a_shape_slice, a_strides_slice,
                scalar as i8, DType::Int8, extract_view_i8, ArrayData::Int8, |
            ),
            DType::Uint64 => scalar_op_arm!(
                a_wrapper, a_offset, a_shape_slice, a_strides_slice,
                scalar as u64, DType::Uint64, extract_view_u64, ArrayData::Uint64, |
            ),
            DType::Uint32 => scalar_op_arm!(
                a_wrapper, a_offset, a_shape_slice, a_strides_slice,
                scalar as u32, DType::Uint32, extract_view_u32, ArrayData::Uint32, |
            ),
            DType::Uint16 => scalar_op_arm!(
                a_wrapper, a_offset, a_shape_slice, a_strides_slice,
                scalar as u16, DType::Uint16, extract_view_u16, ArrayData::Uint16, |
            ),
            DType::Uint8 => scalar_op_arm!(
                a_wrapper, a_offset, a_shape_slice, a_strides_slice,
                scalar as u8, DType::Uint8, extract_view_u8, ArrayData::Uint8, |
            ),
            DType::Bool => scalar_op_arm!(
                a_wrapper, a_offset, a_shape_slice, a_strides_slice,
                if scalar != 0 { 1u8 } else { 0u8 }, DType::Bool, extract_view_bool, ArrayData::Bool, |
            ),
            DType::Float64 | DType::Float32 => {
                crate::error::set_last_error(
                    "Bitwise OR not supported for float types".to_string(),
                );
                return ERR_GENERIC;
            }
        };

        *out = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
