//! Element-wise minimum operation.
//!
//! Returns the smaller value at each position (like NumPy's np.minimum).

use crate::binary_op_arithmetic;
use crate::helpers::error::{set_last_error, ERR_GENERIC, SUCCESS};
use crate::helpers::write_output_metadata;
use crate::types::{ArrayMetadata, NdArrayHandle};

#[inline(always)]
fn minimum<T: Copy + PartialOrd>(a: &T, b: &T) -> T {
    if a <= b {
        *a
    } else {
        *b
    }
}

/// Element-wise minimum with broadcasting.
#[no_mangle]
pub unsafe extern "C" fn ndarray_minimum(
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

        let result_wrapper = binary_op_arithmetic!(a_wrapper, a_meta, b_wrapper, b_meta, minimum);

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
