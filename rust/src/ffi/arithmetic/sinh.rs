//! Hyperbolic sine operation.

use crate::ffi::arithmetic::helpers::unary_op_helper;
use crate::ffi::NdArrayHandle;

/// Compute hyperbolic sine element-wise.
#[no_mangle]
pub unsafe extern "C" fn ndarray_sinh(
    a: *const NdArrayHandle,
    a_offset: usize,
    a_shape: *const usize,
    _a_strides: *const usize,
    ndim: usize,
    out: *mut *mut NdArrayHandle,
) -> i32 {
    unary_op_helper(a, a_offset, a_shape, ndim, out, |x| x.sinh())
}
