//! Division operation.

use crate::binary_op_arm;
use crate::core::view_helpers::{
    extract_view_as_f32, extract_view_as_f64, extract_view_as_i16, extract_view_as_i32,
    extract_view_as_i64, extract_view_as_i8, extract_view_as_u16, extract_view_as_u32,
    extract_view_as_u64, extract_view_as_u8, extract_view_f32, extract_view_f64, extract_view_i16,
    extract_view_i32, extract_view_i64, extract_view_i8, extract_view_u16, extract_view_u32,
    extract_view_u64, extract_view_u8,
};
use crate::core::ArrayData;
use crate::dtype::DType;
use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::{write_output_metadata, NdArrayHandle, ViewMetadata};
use crate::scalar_op_arm;

/// Divide two arrays.
#[no_mangle]
pub unsafe extern "C" fn ndarray_div(
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
            DType::Float64 => binary_op_arm!(
                a_wrapper, a_meta,
                b_wrapper, b_meta,
                DType::Float64, extract_view_as_f64, ArrayData::Float64, /
            ),
            DType::Float32 => binary_op_arm!(
                a_wrapper, a_meta,
                b_wrapper, b_meta,
                DType::Float32, extract_view_as_f32, ArrayData::Float32, /
            ),
            DType::Int64 => binary_op_arm!(
                a_wrapper, a_meta,
                b_wrapper, b_meta,
                DType::Int64, extract_view_as_i64, ArrayData::Int64, /
            ),
            DType::Int32 => binary_op_arm!(
                a_wrapper, a_meta,
                b_wrapper, b_meta,
                DType::Int32, extract_view_as_i32, ArrayData::Int32, /
            ),
            DType::Int16 => binary_op_arm!(
                a_wrapper, a_meta,
                b_wrapper, b_meta,
                DType::Int16, extract_view_as_i16, ArrayData::Int16, /
            ),
            DType::Int8 => binary_op_arm!(
                a_wrapper, a_meta,
                b_wrapper, b_meta,
                DType::Int8, extract_view_as_i8, ArrayData::Int8, /
            ),
            DType::Uint64 => binary_op_arm!(
                a_wrapper, a_meta,
                b_wrapper, b_meta,
                DType::Uint64, extract_view_as_u64, ArrayData::Uint64, /
            ),
            DType::Uint32 => binary_op_arm!(
                a_wrapper, a_meta,
                b_wrapper, b_meta,
                DType::Uint32, extract_view_as_u32, ArrayData::Uint32, /
            ),
            DType::Uint16 => binary_op_arm!(
                a_wrapper, a_meta,
                b_wrapper, b_meta,
                DType::Uint16, extract_view_as_u16, ArrayData::Uint16, /
            ),
            DType::Uint8 => binary_op_arm!(
                a_wrapper, a_meta,
                b_wrapper, b_meta,
                DType::Uint8, extract_view_as_u8, ArrayData::Uint8, /
            ),
            DType::Bool => {
                crate::error::set_last_error("Division not supported for Bool type".to_string());
                return ERR_GENERIC;
            }
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

/// Divide an array by a scalar.
#[no_mangle]
pub unsafe extern "C" fn ndarray_div_scalar(
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

        let result_wrapper = match a_wrapper.dtype {
            DType::Float64 => scalar_op_arm!(
                a_wrapper, a_meta,
                scalar as f64, DType::Float64, extract_view_f64, ArrayData::Float64, /
            ),
            DType::Float32 => scalar_op_arm!(
                a_wrapper, a_meta,
                scalar as f32, DType::Float32, extract_view_f32, ArrayData::Float32, /
            ),
            DType::Int64 => scalar_op_arm!(
                a_wrapper, a_meta,
                scalar as i64, DType::Int64, extract_view_i64, ArrayData::Int64, /
            ),
            DType::Int32 => scalar_op_arm!(
                a_wrapper, a_meta,
                scalar as i32, DType::Int32, extract_view_i32, ArrayData::Int32, /
            ),
            DType::Int16 => scalar_op_arm!(
                a_wrapper, a_meta,
                scalar as i16, DType::Int16, extract_view_i16, ArrayData::Int16, /
            ),
            DType::Int8 => scalar_op_arm!(
                a_wrapper, a_meta,
                scalar as i8, DType::Int8, extract_view_i8, ArrayData::Int8, /
            ),
            DType::Uint64 => scalar_op_arm!(
                a_wrapper, a_meta,
                scalar as u64, DType::Uint64, extract_view_u64, ArrayData::Uint64, /
            ),
            DType::Uint32 => scalar_op_arm!(
                a_wrapper, a_meta,
                scalar as u32, DType::Uint32, extract_view_u32, ArrayData::Uint32, /
            ),
            DType::Uint16 => scalar_op_arm!(
                a_wrapper, a_meta,
                scalar as u16, DType::Uint16, extract_view_u16, ArrayData::Uint16, /
            ),
            DType::Uint8 => scalar_op_arm!(
                a_wrapper, a_meta,
                scalar as u8, DType::Uint8, extract_view_u8, ArrayData::Uint8, /
            ),
            DType::Bool => {
                crate::error::set_last_error("Division not supported for Bool type".to_string());
                return ERR_GENERIC;
            }
        };

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
