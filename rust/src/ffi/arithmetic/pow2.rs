//! Square operation.

use crate::core::math_helpers::unary_op_generic;
use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::NdArrayHandle;
use std::slice;

/// Compute x^2 element-wise.
#[no_mangle]
pub unsafe extern "C" fn ndarray_pow2(
    a: *const NdArrayHandle,
    a_offset: usize,
    a_shape: *const usize,
    a_strides: *const usize,
    ndim: usize,
    out: *mut *mut NdArrayHandle,
) -> i32 {
    if a.is_null() || out.is_null() || a_shape.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let a_wrapper = NdArrayHandle::as_wrapper(a as *mut _);
        let a_shape_slice = slice::from_raw_parts(a_shape, ndim);
        let a_strides_slice = slice::from_raw_parts(a_strides, ndim);

        let result = unary_op_generic::<f64, f32, _, _, _, _, _, _, _, _, _, _>(
            a_wrapper,
            a_offset,
            a_shape_slice,
            a_strides_slice,
            |x: f64| x.powi(2),
            |x: f32| x.powi(2),
            |x: i64| x * x,
            |x: i32| x * x,
            |x: i16| x * x,
            |x: i8| x * x,
            |x: u64| x * x,
            |x: u32| x * x,
            |x: u16| x * x,
            |x: u8| x * x,
        );

        match result {
            Ok(wrapper) => {
                *out = NdArrayHandle::from_wrapper(Box::new(wrapper));
                SUCCESS
            }
            Err(e) => {
                crate::error::set_last_error(e);
                ERR_GENERIC
            }
        }
    })
}
