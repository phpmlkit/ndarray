//! Element-wise greater-or-equal (>=) comparison with broadcasting.

use crate::binary_cmp_op_arm;
use crate::core::view_helpers::{
    extract_view_as_bool, extract_view_as_f32, extract_view_as_f64, extract_view_as_i16,
    extract_view_as_i32, extract_view_as_i64, extract_view_as_i8, extract_view_as_u16,
    extract_view_as_u32, extract_view_as_u64, extract_view_as_u8,
};
use crate::dtype::DType;
use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::{write_output_metadata, NdArrayHandle};
use crate::scalar_cmp_op_arm;

use std::slice;

/// Element-wise greater-or-equal comparison with broadcasting. Returns Bool array.
#[no_mangle]
pub unsafe extern "C" fn ndarray_gte(
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
    out_dtype_ptr: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if a.is_null()
        || b.is_null()
        || out.is_null()
        || out_dtype_ptr.is_null()
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
            DType::Float64 => binary_cmp_op_arm!(a_wrapper, a_offset, a_shape_slice, a_strides_slice, b_wrapper, b_offset, b_shape_slice, b_strides_slice, DType::Float64, extract_view_as_f64, gte),
            DType::Float32 => binary_cmp_op_arm!(a_wrapper, a_offset, a_shape_slice, a_strides_slice, b_wrapper, b_offset, b_shape_slice, b_strides_slice, DType::Float32, extract_view_as_f32, gte),
            DType::Int64 => binary_cmp_op_arm!(a_wrapper, a_offset, a_shape_slice, a_strides_slice, b_wrapper, b_offset, b_shape_slice, b_strides_slice, DType::Int64, extract_view_as_i64, gte),
            DType::Int32 => binary_cmp_op_arm!(a_wrapper, a_offset, a_shape_slice, a_strides_slice, b_wrapper, b_offset, b_shape_slice, b_strides_slice, DType::Int32, extract_view_as_i32, gte),
            DType::Int16 => binary_cmp_op_arm!(a_wrapper, a_offset, a_shape_slice, a_strides_slice, b_wrapper, b_offset, b_shape_slice, b_strides_slice, DType::Int16, extract_view_as_i16, gte),
            DType::Int8 => binary_cmp_op_arm!(a_wrapper, a_offset, a_shape_slice, a_strides_slice, b_wrapper, b_offset, b_shape_slice, b_strides_slice, DType::Int8, extract_view_as_i8, gte),
            DType::Uint64 => binary_cmp_op_arm!(a_wrapper, a_offset, a_shape_slice, a_strides_slice, b_wrapper, b_offset, b_shape_slice, b_strides_slice, DType::Uint64, extract_view_as_u64, gte),
            DType::Uint32 => binary_cmp_op_arm!(a_wrapper, a_offset, a_shape_slice, a_strides_slice, b_wrapper, b_offset, b_shape_slice, b_strides_slice, DType::Uint32, extract_view_as_u32, gte),
            DType::Uint16 => binary_cmp_op_arm!(a_wrapper, a_offset, a_shape_slice, a_strides_slice, b_wrapper, b_offset, b_shape_slice, b_strides_slice, DType::Uint16, extract_view_as_u16, gte),
            DType::Uint8 => binary_cmp_op_arm!(a_wrapper, a_offset, a_shape_slice, a_strides_slice, b_wrapper, b_offset, b_shape_slice, b_strides_slice, DType::Uint8, extract_view_as_u8, gte),
            DType::Bool => binary_cmp_op_arm!(a_wrapper, a_offset, a_shape_slice, a_strides_slice, b_wrapper, b_offset, b_shape_slice, b_strides_slice, DType::Bool, extract_view_as_bool, gte),
        };

        let _ = result_shape;
        if let Err(e) = write_output_metadata(&result_wrapper, out_dtype_ptr, out_ndim, out_shape, max_ndim) {
            crate::error::set_last_error(e);
            return ERR_GENERIC;
        }
        *out = NdArrayHandle::from_wrapper(Box::new(result_wrapper));

        SUCCESS
    })
}

#[no_mangle]
pub unsafe extern "C" fn ndarray_gte_scalar(
    a: *const NdArrayHandle,
    a_offset: usize,
    a_shape: *const usize,
    a_strides: *const usize,
    ndim: usize,
    scalar: f64,
    out: *mut *mut NdArrayHandle,
    out_dtype: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if a.is_null() || out.is_null() || a_shape.is_null() || out_dtype.is_null() || out_ndim.is_null() || out_shape.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let a_wrapper = NdArrayHandle::as_wrapper(a as *mut _);
        let a_shape_slice = slice::from_raw_parts(a_shape, ndim);
        let a_strides_slice = slice::from_raw_parts(a_strides, ndim);

        let result_wrapper = scalar_cmp_op_arm!(a_wrapper, a_offset, a_shape_slice, a_strides_slice, scalar, >=);

        if let Err(e) = write_output_metadata(&result_wrapper, out_dtype, out_ndim, out_shape, max_ndim) {
            crate::error::set_last_error(e);
            return ERR_GENERIC;
        }
        *out = NdArrayHandle::from_wrapper(Box::new(result_wrapper));

        SUCCESS
    })
}
