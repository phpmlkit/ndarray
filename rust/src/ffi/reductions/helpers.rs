//! Helper functions for reduction operations.

use std::ffi::c_void;

use crate::types::dtype::DType;

/// Write a scalar result to FFI output buffers
///
/// out_value: 8-byte buffer, interpreted per dtype (f64, i64, or u64)
/// out_dtype: 1-byte dtype value
pub unsafe fn write_scalar(out_value: *mut c_void, out_dtype: *mut u8, value: f64, dtype: DType) {
    if !out_value.is_null() {
        match dtype {
            DType::Float64 | DType::Float32 => {
                *(out_value as *mut f64) = value;
            }
            DType::Int64 | DType::Int32 | DType::Int16 | DType::Int8 => {
                *(out_value as *mut i64) = value as i64;
            }
            DType::Uint64 | DType::Uint32 | DType::Uint16 | DType::Uint8 => {
                *(out_value as *mut u64) = value as u64;
            }
            DType::Bool => {
                *(out_value as *mut u64) = value as u64;
            }
        }
    }
    if !out_dtype.is_null() {
        *out_dtype = dtype as u8;
    }
}

/// Compute the output shape for axis reduction with keepdims.
pub fn compute_axis_output_shape(shape: &[usize], axis: usize, keepdims: bool) -> Vec<usize> {
    if keepdims {
        shape
            .iter()
            .enumerate()
            .map(|(i, &dim)| if i == axis { 1 } else { dim })
            .collect()
    } else {
        shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, &dim)| dim)
            .collect()
    }
}
