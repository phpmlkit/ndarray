//! Multiplication operation.

use crate::helpers::error::{ERR_GENERIC, SUCCESS};
use crate::helpers::write_output_metadata;
use crate::types::{ArrayMetadata, NdArrayHandle};
use crate::{binary_op_arithmetic, scalar_op_arithmetic};
use std::ops::Mul;

#[inline(always)]
fn mul<T: Copy + Mul<Output = T>>(a: &T, b: &T) -> T {
    *a * *b
}

/// Multiply two arrays.
#[no_mangle]
pub unsafe extern "C" fn ndarray_mul(
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

        let result_wrapper = binary_op_arithmetic!(a_wrapper, a_meta, b_wrapper, b_meta, mul);

        if let Err(e) = write_output_metadata(
            &result_wrapper,
            out_dtype_ptr,
            out_ndim,
            out_shape,
            max_ndim,
        ) {
            crate::helpers::error::set_last_error(e);
            return ERR_GENERIC;
        }
        *out = NdArrayHandle::from_wrapper(Box::new(result_wrapper));

        SUCCESS
    })
}

/// Multiply an array by a scalar.
#[no_mangle]
pub unsafe extern "C" fn ndarray_mul_scalar(
    a: *const NdArrayHandle,
    a_meta: *const ArrayMetadata,
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

        let result_wrapper = scalar_op_arithmetic!(a_wrapper, a_meta, scalar, *);

        if let Err(e) =
            write_output_metadata(&result_wrapper, out_dtype, out_ndim, out_shape, max_ndim)
        {
            crate::helpers::error::set_last_error(e);
            return ERR_GENERIC;
        }
        *out = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
