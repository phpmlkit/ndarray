//! Left shift operation.
//!
//! Works with all integer types (signed and unsigned) only.
//! Bool and float types are not supported.

use crate::binary_op_bitwise;
use crate::helpers::error::{set_last_error, ERR_GENERIC, SUCCESS};
use crate::helpers::write_output_metadata;
use crate::scalar_op_bitwise;
use crate::types::{ArrayMetadata, DType, NdArrayHandle};
use std::ffi::c_void;
use std::ops::Shl;

#[inline(always)]
fn left_shift<T: Copy + Shl<T, Output = T>>(a: &T, b: &T) -> T {
    *a << *b
}

/// Compute the left shift of two arrays.
#[no_mangle]
pub unsafe extern "C" fn ndarray_left_shift(
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
        set_last_error("Invalid input parameters".to_string());
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let a_meta = &*a_meta;
        let b_meta = &*b_meta;
        let a_wrapper = NdArrayHandle::as_wrapper(a as *mut _);
        let b_wrapper = NdArrayHandle::as_wrapper(b as *mut _);

        let result_wrapper = binary_op_bitwise!(a_wrapper, a_meta, b_wrapper, b_meta, left_shift);

        if let Err(e) = write_output_metadata(
            &result_wrapper,
            out_dtype_ptr,
            out_ndim,
            out_shape,
            max_ndim,
        ) {
            set_last_error(e);
            return ERR_GENERIC;
        }
        *out = NdArrayHandle::from_wrapper(Box::new(result_wrapper));

        SUCCESS
    })
}

/// Compute the left shift of an array by a scalar.
#[no_mangle]
pub unsafe extern "C" fn ndarray_left_shift_scalar(
    a: *const NdArrayHandle,
    a_meta: *const ArrayMetadata,
    scalar: *const c_void,
    scalar_dtype: u8,
    out: *mut *mut NdArrayHandle,
    out_dtype: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if a.is_null()
        || scalar.is_null()
        || out.is_null()
        || a_meta.is_null()
        || out_dtype.is_null()
        || out_ndim.is_null()
        || out_shape.is_null()
    {
        set_last_error("Invalid input parameters".to_string());
        return ERR_GENERIC;
    }

    let scalar_dtype = match DType::from_u8(scalar_dtype) {
        Some(d) => d,
        None => {
            set_last_error("Invalid scalar dtype".to_string());
            return ERR_GENERIC;
        }
    };

    crate::ffi_guard!({
        let a_meta = &*a_meta;
        let a_wrapper = NdArrayHandle::as_wrapper(a as *mut _);

        let result_wrapper =
            scalar_op_bitwise!(a_wrapper, a_meta, scalar, scalar_dtype, <<, no_bool);

        if let Err(e) =
            write_output_metadata(&result_wrapper, out_dtype, out_ndim, out_shape, max_ndim)
        {
            set_last_error(e);
            return ERR_GENERIC;
        }
        *out = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
