//! Element-wise equal (==) comparison with broadcasting.

use crate::binary_op_comparison;
use crate::helpers::error::{ERR_GENERIC, SUCCESS};
use crate::helpers::write_output_metadata;
use crate::scalar_op_comparison;
use crate::types::{ArrayMetadata, NdArrayHandle};

/// Equality comparison (==) for use with Zip::map_collect.
#[inline(always)]
fn eq<A: PartialEq>(a: &A, b: &A) -> u8 {
    (a == b) as u8
}

/// Element-wise equal comparison with broadcasting. Returns Bool array.
#[no_mangle]
pub unsafe extern "C" fn ndarray_eq(
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

        let result_wrapper = binary_op_comparison!(a_wrapper, a_meta, b_wrapper, b_meta, eq);

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

/// Element-wise equal comparison with scalar. Returns Bool array.
#[no_mangle]
pub unsafe extern "C" fn ndarray_eq_scalar(
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

        let result_wrapper = scalar_op_comparison!(a_wrapper, a_meta, scalar, ==);

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
