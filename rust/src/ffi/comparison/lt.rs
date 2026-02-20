//! Element-wise less-than (<) comparison with broadcasting.

use crate::binary_cmp_op_arm;
use crate::core::view_helpers::{
    extract_view_as_bool, extract_view_as_f32, extract_view_as_f64, extract_view_as_i16,
    extract_view_as_i32, extract_view_as_i64, extract_view_as_i8, extract_view_as_u16,
    extract_view_as_u32, extract_view_as_u64, extract_view_as_u8,
};
use crate::dtype::DType;
use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::{write_output_metadata, NdArrayHandle, ViewMetadata};
use crate::scalar_cmp_op_arm;

/// Element-wise less-than comparison with broadcasting. Returns Bool array.
#[no_mangle]
pub unsafe extern "C" fn ndarray_lt(
    a: *const NdArrayHandle,
    a_meta: *const ViewMetadata,
    b: *const NdArrayHandle,
    b_meta: *const ViewMetadata,
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
        || a_meta.is_null()
        || b_meta.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let a_meta = &*a_meta;
        let b_meta = &*b_meta;

        let a_wrapper = NdArrayHandle::as_wrapper(a as *mut _);
        let b_wrapper = NdArrayHandle::as_wrapper(b as *mut _);

        let out_dtype = DType::promote(a_wrapper.dtype, b_wrapper.dtype);

        let (result_wrapper, result_shape) = match out_dtype {
            DType::Float64 => binary_cmp_op_arm!(
                a_wrapper, a_meta,
                b_wrapper, b_meta,
                DType::Float64,
                extract_view_as_f64,
                lt
            ),
            DType::Float32 => binary_cmp_op_arm!(
                a_wrapper, a_meta,
                b_wrapper, b_meta,
                DType::Float32,
                extract_view_as_f32,
                lt
            ),
            DType::Int64 => binary_cmp_op_arm!(
                a_wrapper, a_meta,
                b_wrapper, b_meta,
                DType::Int64,
                extract_view_as_i64,
                lt
            ),
            DType::Int32 => binary_cmp_op_arm!(
                a_wrapper, a_meta,
                b_wrapper, b_meta,
                DType::Int32,
                extract_view_as_i32,
                lt
            ),
            DType::Int16 => binary_cmp_op_arm!(
                a_wrapper, a_meta,
                b_wrapper, b_meta,
                DType::Int16,
                extract_view_as_i16,
                lt
            ),
            DType::Int8 => binary_cmp_op_arm!(
                a_wrapper, a_meta,
                b_wrapper, b_meta,
                DType::Int8,
                extract_view_as_i8,
                lt
            ),
            DType::Uint64 => binary_cmp_op_arm!(
                a_wrapper, a_meta,
                b_wrapper, b_meta,
                DType::Uint64,
                extract_view_as_u64,
                lt
            ),
            DType::Uint32 => binary_cmp_op_arm!(
                a_wrapper, a_meta,
                b_wrapper, b_meta,
                DType::Uint32,
                extract_view_as_u32,
                lt
            ),
            DType::Uint16 => binary_cmp_op_arm!(
                a_wrapper, a_meta,
                b_wrapper, b_meta,
                DType::Uint16,
                extract_view_as_u16,
                lt
            ),
            DType::Uint8 => binary_cmp_op_arm!(
                a_wrapper, a_meta,
                b_wrapper, b_meta,
                DType::Uint8,
                extract_view_as_u8,
                lt
            ),
            DType::Bool => binary_cmp_op_arm!(
                a_wrapper, a_meta,
                b_wrapper, b_meta,
                DType::Bool,
                extract_view_as_bool,
                lt
            ),
        };

        let _ = result_shape;
        if let Err(e) = write_output_metadata(
            &result_wrapper,
            out_dtype_ptr,
            out_ndim,
            out_shape,
            max_ndim,
        ) {
            crate::error::set_last_error(e);
            return ERR_GENERIC;
        }
        *out = NdArrayHandle::from_wrapper(Box::new(result_wrapper));

        SUCCESS
    })
}

#[no_mangle]
pub unsafe extern "C" fn ndarray_lt_scalar(
    a: *const NdArrayHandle,
    a_meta: *const ViewMetadata,
    scalar: f64,
    out: *mut *mut NdArrayHandle,
    out_dtype: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if a.is_null()
        || out.is_null()
        || a_meta.is_null()
        || out_dtype.is_null()
        || out_ndim.is_null()
        || out_shape.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let a_meta = &*a_meta;

        let a_wrapper = NdArrayHandle::as_wrapper(a as *mut _);

        let result_wrapper = scalar_cmp_op_arm!(
            a_wrapper, a_meta, scalar, <
        );

        if let Err(e) =
            write_output_metadata(&result_wrapper, out_dtype, out_ndim, out_shape, max_ndim)
        {
            crate::error::set_last_error(e);
            return ERR_GENERIC;
        }
        *out = NdArrayHandle::from_wrapper(Box::new(result_wrapper));

        SUCCESS
    })
}
