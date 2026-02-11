//! Helper functions for reduction operations.

use ndarray::{ArrayD, IxDyn};
use parking_lot::RwLock;
use std::sync::Arc;

use crate::core::ArrayData;
use crate::core::NDArrayWrapper;
use crate::dtype::DType;

/// Create a 0-dimensional (scalar) array wrapper from a single f64 value.
pub fn create_scalar_wrapper(value: f64, dtype: DType) -> NDArrayWrapper {
    let shape = vec![]; // 0-dimensional (scalar)

    match dtype {
        DType::Float64 => {
            let arr = ArrayD::<f64>::from_shape_vec(IxDyn(&shape), vec![value])
                .expect("Failed to create scalar array");
            NDArrayWrapper {
                data: ArrayData::Float64(Arc::new(RwLock::new(arr))),
                dtype,
            }
        }
        DType::Float32 => {
            let arr = ArrayD::<f32>::from_shape_vec(IxDyn(&shape), vec![value as f32])
                .expect("Failed to create scalar array");
            NDArrayWrapper {
                data: ArrayData::Float32(Arc::new(RwLock::new(arr))),
                dtype,
            }
        }
        DType::Int64 => {
            let arr = ArrayD::<i64>::from_shape_vec(IxDyn(&shape), vec![value as i64])
                .expect("Failed to create scalar array");
            NDArrayWrapper {
                data: ArrayData::Int64(Arc::new(RwLock::new(arr))),
                dtype,
            }
        }
        DType::Int32 => {
            let arr = ArrayD::<i32>::from_shape_vec(IxDyn(&shape), vec![value as i32])
                .expect("Failed to create scalar array");
            NDArrayWrapper {
                data: ArrayData::Int32(Arc::new(RwLock::new(arr))),
                dtype,
            }
        }
        DType::Int16 => {
            let arr = ArrayD::<i16>::from_shape_vec(IxDyn(&shape), vec![value as i16])
                .expect("Failed to create scalar array");
            NDArrayWrapper {
                data: ArrayData::Int16(Arc::new(RwLock::new(arr))),
                dtype,
            }
        }
        DType::Int8 => {
            let arr = ArrayD::<i8>::from_shape_vec(IxDyn(&shape), vec![value as i8])
                .expect("Failed to create scalar array");
            NDArrayWrapper {
                data: ArrayData::Int8(Arc::new(RwLock::new(arr))),
                dtype,
            }
        }
        DType::Uint64 => {
            let arr = ArrayD::<u64>::from_shape_vec(IxDyn(&shape), vec![value as u64])
                .expect("Failed to create scalar array");
            NDArrayWrapper {
                data: ArrayData::Uint64(Arc::new(RwLock::new(arr))),
                dtype,
            }
        }
        DType::Uint32 => {
            let arr = ArrayD::<u32>::from_shape_vec(IxDyn(&shape), vec![value as u32])
                .expect("Failed to create scalar array");
            NDArrayWrapper {
                data: ArrayData::Uint32(Arc::new(RwLock::new(arr))),
                dtype,
            }
        }
        DType::Uint16 => {
            let arr = ArrayD::<u16>::from_shape_vec(IxDyn(&shape), vec![value as u16])
                .expect("Failed to create scalar array");
            NDArrayWrapper {
                data: ArrayData::Uint16(Arc::new(RwLock::new(arr))),
                dtype,
            }
        }
        DType::Uint8 => {
            let arr = ArrayD::<u8>::from_shape_vec(IxDyn(&shape), vec![value as u8])
                .expect("Failed to create scalar array");
            NDArrayWrapper {
                data: ArrayData::Uint8(Arc::new(RwLock::new(arr))),
                dtype,
            }
        }
        DType::Bool => {
            let arr =
                ArrayD::<u8>::from_shape_vec(IxDyn(&shape), vec![if value != 0.0 { 1 } else { 0 }])
                    .expect("Failed to create scalar array");
            NDArrayWrapper {
                data: ArrayData::Bool(Arc::new(RwLock::new(arr))),
                dtype,
            }
        }
    }
}

/// Create a 0-dimensional (scalar) array wrapper from an i64 value.
pub fn create_scalar_wrapper_i64(value: i64) -> NDArrayWrapper {
    let shape = vec![]; // 0-dimensional (scalar)
    let arr = ArrayD::<i64>::from_shape_vec(IxDyn(&shape), vec![value])
        .expect("Failed to create scalar array");
    NDArrayWrapper {
        data: ArrayData::Int64(Arc::new(RwLock::new(arr))),
        dtype: DType::Int64,
    }
}

/// Convert a wrapper to a different dtype using f64 as intermediate.
pub fn convert_wrapper_dtype(
    wrapper: NDArrayWrapper,
    target_dtype: DType,
) -> Result<NDArrayWrapper, String> {
    use crate::ffi::arithmetic::helpers::convert_wrapper_dtype as convert_dtype;
    convert_dtype(wrapper, target_dtype)
}

/// Extract a view from a wrapper with the given offset and shape.
pub fn extract_view(wrapper: &NDArrayWrapper, offset: usize, shape: &[usize]) -> ArrayD<f64> {
    let data = wrapper.to_f64_vec();
    let len: usize = shape.iter().product();
    let view_data: Vec<f64> = (0..len).map(|i| data[offset + i]).collect();
    ArrayD::from_shape_vec(IxDyn(shape), view_data).expect("Failed to create view")
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
