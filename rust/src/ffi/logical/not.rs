//! Logical NOT operation.
//!
//! Works with all input types (converts to bool first).
//! Always returns a Bool array.

use crate::helpers::error::{set_last_error, ERR_GENERIC, SUCCESS};
use crate::helpers::extract_view_as_bool;
use crate::helpers::write_output_metadata;
use crate::types::dtype::DType;
use crate::types::{ArrayData, ArrayMetadata, NDArrayWrapper, NdArrayHandle};
use parking_lot::RwLock;
use std::sync::Arc;

/// Logical NOT for use with mapv.
#[inline(always)]
fn not(a: &u8) -> u8 {
    (*a == 0) as u8
}

/// Compute the logical NOT of an array.
/// Array is converted to bool first, result is always Bool.
#[no_mangle]
pub unsafe extern "C" fn ndarray_logical_not(
    a: *const NdArrayHandle,
    a_meta: *const ArrayMetadata,
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

        let Some(view) = extract_view_as_bool(a_wrapper, a_meta) else {
            set_last_error("Failed to extract view as bool".to_string());
            return ERR_GENERIC;
        };
        let result = view.mapv(|x| not(&x));
        let result_wrapper = NDArrayWrapper {
            data: ArrayData::Bool(Arc::new(RwLock::new(result))),
            dtype: DType::Bool,
        };

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
