//! Hypotenuse operation.

use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::arithmetic::helpers::*;
use crate::ffi::NdArrayHandle;
use std::slice;

/// Compute hypot(a, b) = sqrt(a^2 + b^2) element-wise.
#[no_mangle]
pub unsafe extern "C" fn ndarray_hypot(
    a: *const NdArrayHandle,
    a_offset: usize,
    a_shape: *const usize,
    _a_strides: *const usize,
    b: *const NdArrayHandle,
    b_offset: usize,
    b_shape: *const usize,
    _b_strides: *const usize,
    ndim: usize,
    out: *mut *mut NdArrayHandle,
) -> i32 {
    if a.is_null() || b.is_null() || out.is_null() || a_shape.is_null() || b_shape.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let a_wrapper = NdArrayHandle::as_wrapper(a as *mut _);
        let b_wrapper = NdArrayHandle::as_wrapper(b as *mut _);

        let a_shape_slice = slice::from_raw_parts(a_shape, ndim);
        let b_shape_slice = slice::from_raw_parts(b_shape, ndim);

        let out_dtype = match promote_dtypes(a_wrapper.dtype, b_wrapper.dtype) {
            Some(d) => d,
            None => {
                return ERR_GENERIC;
            }
        };

        let result = match out_dtype {
            crate::dtype::DType::Float64 => binary_op_f64(
                a_wrapper,
                a_offset,
                a_shape_slice,
                b_wrapper,
                b_offset,
                b_shape_slice,
                &|a, b| a.hypot(b),
            ),
            crate::dtype::DType::Float32 => binary_op_f32(
                a_wrapper,
                a_offset,
                a_shape_slice,
                b_wrapper,
                b_offset,
                b_shape_slice,
                &|a, b| a.hypot(b),
            ),
            crate::dtype::DType::Int64 => binary_op_i64(
                a_wrapper,
                a_offset,
                a_shape_slice,
                b_wrapper,
                b_offset,
                b_shape_slice,
                &|a, b| a.hypot(b),
            ),
            crate::dtype::DType::Int32 => binary_op_i32(
                a_wrapper,
                a_offset,
                a_shape_slice,
                b_wrapper,
                b_offset,
                b_shape_slice,
                &|a, b| a.hypot(b),
            ),
            _ => {
                match binary_op_f64(
                    a_wrapper,
                    a_offset,
                    a_shape_slice,
                    b_wrapper,
                    b_offset,
                    b_shape_slice,
                    &|a, b| a.hypot(b),
                ) {
                    Ok(wrapper) => convert_wrapper_dtype(wrapper, out_dtype),
                    Err(e) => Err(e),
                }
            }
        };

        match result {
            Ok(wrapper) => {
                *out = NdArrayHandle::from_wrapper(Box::new(wrapper));
                SUCCESS
            }
            Err(_) => ERR_GENERIC,
        }
    })
}
