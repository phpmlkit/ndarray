//! Logical NOT operation.
//!
//! Works with all input types (converts to bool first).
//! Always returns a Bool array.

use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::{write_output_metadata, NdArrayHandle, ViewMetadata};
use crate::unary_logical_op_arm;

/// Compute the logical NOT of an array.
/// Array is converted to bool first, result is always Bool.
#[no_mangle]
pub unsafe extern "C" fn ndarray_logical_not(
    a: *const NdArrayHandle,
    a_meta: *const ViewMetadata,
    out: *mut *mut NdArrayHandle,
    out_dtype_ptr: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if a.is_null()
        || out.is_null()
        || out_dtype_ptr.is_null()
        || out_shape.is_null()
        || out_ndim.is_null()
        || a_meta.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let a_meta = &*a_meta;
        let a_wrapper = NdArrayHandle::as_wrapper(a as *mut _);

        let (result_wrapper, _) = unary_logical_op_arm!(a_wrapper, a_meta, not);

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
