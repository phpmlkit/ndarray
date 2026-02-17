//! Helper functions for reduction operations.

use std::ffi::c_void;

use crate::dtype::DType;

/// Write a scalar result to FFI output buffers
///
/// out_value: 8-byte buffer, interpreted per dtype (f64, i64, or u64)
/// out_dtype: 1-byte dtype value
pub unsafe fn write_scalar(
    out_value: *mut c_void,
    out_dtype: *mut u8,
    value: f64,
    dtype: DType,
) {
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

/// Validate axis is within bounds.
pub fn validate_axis(shape: &[usize], axis: i32) -> Result<usize, String> {
    let ndim = shape.len();
    if ndim == 0 {
        return Err("Cannot reduce 0-dimensional array along axis".to_string());
    }

    let axis_usize = if axis < 0 {
        (ndim as i32 + axis) as usize
    } else {
        axis as usize
    };

    if axis_usize >= ndim {
        return Err(format!(
            "Axis {} is out of bounds for array with {} dimensions",
            axis, ndim
        ));
    }

    Ok(axis_usize)
}
