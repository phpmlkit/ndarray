//! Bitwise AND operation.
//!
//! Works with all integer types (signed and unsigned) and Bool.

use crate::binary_op_bitwise;
use crate::core::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::{write_output_metadata, ArrayMetadata, NdArrayHandle};
use crate::scalar_op_bitwise;
use std::ops::BitAnd;

#[inline(always)]
fn bitand<T: Copy + BitAnd<Output = T>>(a: &T, b: &T) -> T {
    *a & *b
}

/// Compute the bitwise AND of two arrays.
#[no_mangle]
pub unsafe extern "C" fn ndarray_bitand(
    a: *const NdArrayHandle,
    a_meta: *const ArrayMetadata,
    b: *const NdArrayHandle,
    b_meta: *const ArrayMetadata,
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

        let result_wrapper = binary_op_bitwise!(a_wrapper, a_meta, b_wrapper, b_meta, bitand);

        if let Err(e) = write_output_metadata(
            &result_wrapper,
            out_dtype_ptr,
            out_ndim,
            out_shape,
            max_ndim,
        ) {
            crate::core::error::set_last_error(e);
            return ERR_GENERIC;
        }
        *out = NdArrayHandle::from_wrapper(Box::new(result_wrapper));

        SUCCESS
    })
}

/// Compute the bitwise AND of an array and a scalar.
#[no_mangle]
pub unsafe extern "C" fn ndarray_bitand_scalar(
    a: *const NdArrayHandle,
    a_meta: *const ArrayMetadata,
    scalar: i64,
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

        let result_wrapper = scalar_op_bitwise!(a_wrapper, a_meta, scalar, &);

        if let Err(e) =
            write_output_metadata(&result_wrapper, out_dtype, out_ndim, out_shape, max_ndim)
        {
            crate::core::error::set_last_error(e);
            return ERR_GENERIC;
        }
        *out = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
