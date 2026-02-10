//! Array arithmetic FFI functions (add, sub, mul, div).
//!
//! Provides element-wise arithmetic operations with broadcasting support
//! for both array-array and array-scalar operations.

use ndarray::{ArrayD, IxDyn};
use parking_lot::RwLock;
use std::slice;
use std::sync::Arc;

use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{self, ERR_DTYPE, ERR_GENERIC, SUCCESS};
use crate::ffi::NdArrayHandle;

// ============================================================================
// Type Promotion Helpers
// ============================================================================

/// Get the promoted dtype for binary operations.
fn promote_dtypes(a: DType, b: DType) -> Option<DType> {
    use DType::*;

    // Float64 wins over everything
    if a == Float64 || b == Float64 {
        return Some(Float64);
    }
    // Float32 wins over integers
    if a == Float32 || b == Float32 {
        return Some(Float32);
    }
    // Int64 wins over smaller integers
    if a == Int64 || b == Int64 {
        return Some(Int64);
    }
    // Int32 wins over smaller integers
    if a == Int32 || b == Int32 {
        return Some(Int32);
    }
    // Int16 wins over Int8
    if a == Int16 || b == Int16 {
        return Some(Int16);
    }
    // Uint64 wins over smaller unsigned
    if a == Uint64 || b == Uint64 {
        return Some(Uint64);
    }
    // Uint32 wins over smaller unsigned
    if a == Uint32 || b == Uint32 {
        return Some(Uint32);
    }
    // Uint16 wins over Uint8
    if a == Uint16 || b == Uint16 {
        return Some(Uint16);
    }
    // Both same
    if a == b {
        return Some(a);
    }

    // Mixed signed/unsigned - promote to signed
    match (a, b) {
        (Int8, Uint8) | (Uint8, Int8) => Some(Int16),
        (Int16, Uint8) | (Uint8, Int16) => Some(Int16),
        (Int16, Uint16) | (Uint16, Int16) => Some(Int32),
        (Int32, Uint8) | (Uint8, Int32) => Some(Int32),
        (Int32, Uint16) | (Uint16, Int32) => Some(Int32),
        (Int32, Uint32) | (Uint32, Int32) => Some(Int64),
        (Int64, _) | (_, Int64) => Some(Int64),
        _ => None,
    }
}

// ============================================================================
// Binary Operation Helpers
// ============================================================================

/// Perform element-wise binary operation on two arrays with broadcasting.
///
/// Both arrays can be views (defined by offset, shape, strides).
/// The operation is applied element-wise with ndarray's broadcasting rules.
unsafe fn binary_op_helper<F>(
    a_handle: *const NdArrayHandle,
    a_offset: usize,
    a_shape_ptr: *const usize,
    a_strides_ptr: *const usize,
    b_handle: *const NdArrayHandle,
    b_offset: usize,
    b_shape_ptr: *const usize,
    b_strides_ptr: *const usize,
    ndim: usize,
    out_handle: *mut *mut NdArrayHandle,
    op_name: &str,
    op: F,
) -> i32
where
    F: Fn(f64, f64) -> f64 + std::panic::UnwindSafe,
{
    if a_handle.is_null()
        || b_handle.is_null()
        || out_handle.is_null()
        || a_shape_ptr.is_null()
        || a_strides_ptr.is_null()
        || b_shape_ptr.is_null()
        || b_strides_ptr.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let a_wrapper = NdArrayHandle::as_wrapper(a_handle as *mut _);
        let b_wrapper = NdArrayHandle::as_wrapper(b_handle as *mut _);

        let a_shape = slice::from_raw_parts(a_shape_ptr, ndim);
        let a_strides = slice::from_raw_parts(a_strides_ptr, ndim);
        let b_shape = slice::from_raw_parts(b_shape_ptr, ndim);
        let b_strides = slice::from_raw_parts(b_strides_ptr, ndim);

        // Determine output dtype via type promotion
        let out_dtype = match promote_dtypes(a_wrapper.dtype, b_wrapper.dtype) {
            Some(d) => d,
            None => {
                error::set_last_error(format!(
                    "Cannot promote dtypes {:?} and {:?}",
                    a_wrapper.dtype, b_wrapper.dtype
                ));
                return ERR_DTYPE;
            }
        };

        // For simplicity, we'll convert both to f64, operate, then convert back
        // In production, you'd want type-specific implementations for performance
        let result = match out_dtype {
            DType::Float64 => binary_op_f64(
                a_wrapper, a_offset, a_shape, a_strides, b_wrapper, b_offset, b_shape, b_strides,
                &op,
            ),
            DType::Float32 => binary_op_f32(
                a_wrapper, a_offset, a_shape, a_strides, b_wrapper, b_offset, b_shape, b_strides,
                &op,
            ),
            DType::Int64 => binary_op_i64(
                a_wrapper, a_offset, a_shape, a_strides, b_wrapper, b_offset, b_shape, b_strides,
                &op,
            ),
            DType::Int32 => binary_op_i32(
                a_wrapper, a_offset, a_shape, a_strides, b_wrapper, b_offset, b_shape, b_strides,
                &op,
            ),
            _ => {
                // For other types, convert through f64
                match binary_op_f64(
                    a_wrapper, a_offset, a_shape, a_strides, b_wrapper, b_offset, b_shape,
                    b_strides, &op,
                ) {
                    Ok(wrapper) => convert_wrapper_dtype(wrapper, out_dtype),
                    Err(e) => Err(e),
                }
            }
        };

        match result {
            Ok(wrapper) => {
                *out_handle = NdArrayHandle::from_wrapper(Box::new(wrapper));
                SUCCESS
            }
            Err(e) => {
                error::set_last_error(e);
                ERR_GENERIC
            }
        }
    })
}

/// Convert a wrapper to a different dtype.
fn convert_wrapper_dtype(
    wrapper: NDArrayWrapper,
    target_dtype: DType,
) -> Result<NDArrayWrapper, String> {
    if wrapper.dtype == target_dtype {
        return Ok(wrapper);
    }

    // Convert through f64 vec
    let data = wrapper.to_f64_vec();
    let shape = wrapper.shape();

    match target_dtype {
        DType::Float64 => {
            let arr =
                ArrayD::<f64>::from_shape_vec(IxDyn(&shape), data).map_err(|e| e.to_string())?;
            Ok(NDArrayWrapper {
                data: ArrayData::Float64(Arc::new(RwLock::new(arr))),
                dtype: DType::Float64,
            })
        }
        DType::Float32 => {
            let data: Vec<f32> = data.into_iter().map(|x| x as f32).collect();
            let arr =
                ArrayD::<f32>::from_shape_vec(IxDyn(&shape), data).map_err(|e| e.to_string())?;
            Ok(NDArrayWrapper {
                data: ArrayData::Float32(Arc::new(RwLock::new(arr))),
                dtype: DType::Float32,
            })
        }
        DType::Int64 => {
            let data: Vec<i64> = data.into_iter().map(|x| x as i64).collect();
            let arr =
                ArrayD::<i64>::from_shape_vec(IxDyn(&shape), data).map_err(|e| e.to_string())?;
            Ok(NDArrayWrapper {
                data: ArrayData::Int64(Arc::new(RwLock::new(arr))),
                dtype: DType::Int64,
            })
        }
        DType::Int32 => {
            let data: Vec<i32> = data.into_iter().map(|x| x as i32).collect();
            let arr =
                ArrayD::<i32>::from_shape_vec(IxDyn(&shape), data).map_err(|e| e.to_string())?;
            Ok(NDArrayWrapper {
                data: ArrayData::Int32(Arc::new(RwLock::new(arr))),
                dtype: DType::Int32,
            })
        }
        DType::Int16 => {
            let data: Vec<i16> = data.into_iter().map(|x| x as i16).collect();
            let arr =
                ArrayD::<i16>::from_shape_vec(IxDyn(&shape), data).map_err(|e| e.to_string())?;
            Ok(NDArrayWrapper {
                data: ArrayData::Int16(Arc::new(RwLock::new(arr))),
                dtype: DType::Int16,
            })
        }
        DType::Int8 => {
            let data: Vec<i8> = data.into_iter().map(|x| x as i8).collect();
            let arr =
                ArrayD::<i8>::from_shape_vec(IxDyn(&shape), data).map_err(|e| e.to_string())?;
            Ok(NDArrayWrapper {
                data: ArrayData::Int8(Arc::new(RwLock::new(arr))),
                dtype: DType::Int8,
            })
        }
        DType::Uint64 => {
            let data: Vec<u64> = data.into_iter().map(|x| x as u64).collect();
            let arr =
                ArrayD::<u64>::from_shape_vec(IxDyn(&shape), data).map_err(|e| e.to_string())?;
            Ok(NDArrayWrapper {
                data: ArrayData::Uint64(Arc::new(RwLock::new(arr))),
                dtype: DType::Uint64,
            })
        }
        DType::Uint32 => {
            let data: Vec<u32> = data.into_iter().map(|x| x as u32).collect();
            let arr =
                ArrayD::<u32>::from_shape_vec(IxDyn(&shape), data).map_err(|e| e.to_string())?;
            Ok(NDArrayWrapper {
                data: ArrayData::Uint32(Arc::new(RwLock::new(arr))),
                dtype: DType::Uint32,
            })
        }
        DType::Uint16 => {
            let data: Vec<u16> = data.into_iter().map(|x| x as u16).collect();
            let arr =
                ArrayD::<u16>::from_shape_vec(IxDyn(&shape), data).map_err(|e| e.to_string())?;
            Ok(NDArrayWrapper {
                data: ArrayData::Uint16(Arc::new(RwLock::new(arr))),
                dtype: DType::Uint16,
            })
        }
        DType::Uint8 => {
            let data: Vec<u8> = data.into_iter().map(|x| x as u8).collect();
            let arr =
                ArrayD::<u8>::from_shape_vec(IxDyn(&shape), data).map_err(|e| e.to_string())?;
            Ok(NDArrayWrapper {
                data: ArrayData::Uint8(Arc::new(RwLock::new(arr))),
                dtype: DType::Uint8,
            })
        }
        DType::Bool => {
            let data: Vec<u8> = data
                .into_iter()
                .map(|x| if x != 0.0 { 1 } else { 0 })
                .collect();
            let arr =
                ArrayD::<u8>::from_shape_vec(IxDyn(&shape), data).map_err(|e| e.to_string())?;
            Ok(NDArrayWrapper {
                data: ArrayData::Bool(Arc::new(RwLock::new(arr))),
                dtype: DType::Bool,
            })
        }
    }
}

// Type-specific implementations
unsafe fn binary_op_f64<F>(
    a_wrapper: &NDArrayWrapper,
    a_offset: usize,
    a_shape: &[usize],
    a_strides: &[usize],
    b_wrapper: &NDArrayWrapper,
    b_offset: usize,
    b_shape: &[usize],
    b_strides: &[usize],
    op: &F,
) -> Result<NDArrayWrapper, String>
where
    F: Fn(f64, f64) -> f64,
{
    // Create views from raw pointers
    let a_data = a_wrapper.to_f64_vec();
    let b_data = b_wrapper.to_f64_vec();

    // Get the actual shape sizes
    let a_len: usize = a_shape.iter().product();
    let b_len: usize = b_shape.iter().product();

    // Handle broadcasting for 1D arrays
    let out_shape = if a_shape == b_shape {
        a_shape.to_vec()
    } else if a_len == 1 {
        b_shape.to_vec()
    } else if b_len == 1 {
        a_shape.to_vec()
    } else if a_shape.len() == b_shape.len() {
        // Try broadcasting dimension by dimension
        let mut out = Vec::with_capacity(a_shape.len());
        for (a, b) in a_shape.iter().zip(b_shape.iter()) {
            if a == b {
                out.push(*a);
            } else if *a == 1 {
                out.push(*b);
            } else if *b == 1 {
                out.push(*a);
            } else {
                return Err(format!("Shape mismatch: {:?} vs {:?}", a_shape, b_shape));
            }
        }
        out
    } else {
        return Err(format!(
            "Shape mismatch for broadcasting: {:?} vs {:?}",
            a_shape, b_shape
        ));
    };

    let out_len: usize = out_shape.iter().product();
    let mut result_data = Vec::with_capacity(out_len);

    // Apply operation element-wise
    if a_len == 1 && b_len == 1 {
        // Both scalars (in array form)
        result_data.push(op(a_data[a_offset], b_data[b_offset]));
    } else if a_len == 1 {
        // A is scalar
        let a_val = a_data[a_offset];
        for i in 0..out_len {
            result_data.push(op(a_val, b_data[b_offset + i]));
        }
    } else if b_len == 1 {
        // B is scalar
        let b_val = b_data[b_offset];
        for i in 0..out_len {
            result_data.push(op(a_data[a_offset + i], b_val));
        }
    } else if a_shape == b_shape {
        // Same shape
        for i in 0..out_len {
            result_data.push(op(a_data[a_offset + i], b_data[b_offset + i]));
        }
    } else {
        // Broadcasting needed - simplified implementation
        // For full stride-aware broadcasting, we'd need more complex logic
        return Err("Complex broadcasting not yet fully implemented".to_string());
    }

    let arr =
        ArrayD::<f64>::from_shape_vec(IxDyn(&out_shape), result_data).map_err(|e| e.to_string())?;

    Ok(NDArrayWrapper {
        data: ArrayData::Float64(Arc::new(RwLock::new(arr))),
        dtype: DType::Float64,
    })
}

unsafe fn binary_op_f32<F>(
    a_wrapper: &NDArrayWrapper,
    a_offset: usize,
    a_shape: &[usize],
    _a_strides: &[usize],
    b_wrapper: &NDArrayWrapper,
    b_offset: usize,
    b_shape: &[usize],
    _b_strides: &[usize],
    op: &F,
) -> Result<NDArrayWrapper, String>
where
    F: Fn(f64, f64) -> f64,
{
    let a_data: Vec<f32> = a_wrapper
        .to_f64_vec()
        .into_iter()
        .map(|x| x as f32)
        .collect();
    let b_data: Vec<f32> = b_wrapper
        .to_f64_vec()
        .into_iter()
        .map(|x| x as f32)
        .collect();

    let a_len: usize = a_shape.iter().product();
    let b_len: usize = b_shape.iter().product();

    let out_shape = if a_shape == b_shape {
        a_shape.to_vec()
    } else if a_len == 1 {
        b_shape.to_vec()
    } else if b_len == 1 {
        a_shape.to_vec()
    } else {
        return Err(format!("Shape mismatch: {:?} vs {:?}", a_shape, b_shape));
    };

    let out_len: usize = out_shape.iter().product();
    let mut result_data = Vec::with_capacity(out_len);

    if a_len == 1 && b_len == 1 {
        result_data.push(op(a_data[a_offset] as f64, b_data[b_offset] as f64) as f32);
    } else if a_len == 1 {
        let a_val = a_data[a_offset];
        for i in 0..out_len {
            result_data.push(op(a_val as f64, b_data[b_offset + i] as f64) as f32);
        }
    } else if b_len == 1 {
        let b_val = b_data[b_offset];
        for i in 0..out_len {
            result_data.push(op(a_data[a_offset + i] as f64, b_val as f64) as f32);
        }
    } else {
        for i in 0..out_len {
            result_data.push(op(a_data[a_offset + i] as f64, b_data[b_offset + i] as f64) as f32);
        }
    }

    let arr =
        ArrayD::<f32>::from_shape_vec(IxDyn(&out_shape), result_data).map_err(|e| e.to_string())?;

    Ok(NDArrayWrapper {
        data: ArrayData::Float32(Arc::new(RwLock::new(arr))),
        dtype: DType::Float32,
    })
}

unsafe fn binary_op_i64<F>(
    a_wrapper: &NDArrayWrapper,
    a_offset: usize,
    a_shape: &[usize],
    _a_strides: &[usize],
    b_wrapper: &NDArrayWrapper,
    b_offset: usize,
    b_shape: &[usize],
    _b_strides: &[usize],
    op: &F,
) -> Result<NDArrayWrapper, String>
where
    F: Fn(f64, f64) -> f64,
{
    let a_data: Vec<i64> = a_wrapper
        .to_f64_vec()
        .into_iter()
        .map(|x| x as i64)
        .collect();
    let b_data: Vec<i64> = b_wrapper
        .to_f64_vec()
        .into_iter()
        .map(|x| x as i64)
        .collect();

    let a_len: usize = a_shape.iter().product();
    let b_len: usize = b_shape.iter().product();

    let out_shape = if a_shape == b_shape {
        a_shape.to_vec()
    } else if a_len == 1 {
        b_shape.to_vec()
    } else if b_len == 1 {
        a_shape.to_vec()
    } else {
        return Err(format!("Shape mismatch: {:?} vs {:?}", a_shape, b_shape));
    };

    let out_len: usize = out_shape.iter().product();
    let mut result_data = Vec::with_capacity(out_len);

    if a_len == 1 && b_len == 1 {
        result_data.push(op(a_data[a_offset] as f64, b_data[b_offset] as f64) as i64);
    } else if a_len == 1 {
        let a_val = a_data[a_offset];
        for i in 0..out_len {
            result_data.push(op(a_val as f64, b_data[b_offset + i] as f64) as i64);
        }
    } else if b_len == 1 {
        let b_val = b_data[b_offset];
        for i in 0..out_len {
            result_data.push(op(a_data[a_offset + i] as f64, b_val as f64) as i64);
        }
    } else {
        for i in 0..out_len {
            result_data.push(op(a_data[a_offset + i] as f64, b_data[b_offset + i] as f64) as i64);
        }
    }

    let arr =
        ArrayD::<i64>::from_shape_vec(IxDyn(&out_shape), result_data).map_err(|e| e.to_string())?;

    Ok(NDArrayWrapper {
        data: ArrayData::Int64(Arc::new(RwLock::new(arr))),
        dtype: DType::Int64,
    })
}

unsafe fn binary_op_i32<F>(
    a_wrapper: &NDArrayWrapper,
    a_offset: usize,
    a_shape: &[usize],
    _a_strides: &[usize],
    b_wrapper: &NDArrayWrapper,
    b_offset: usize,
    b_shape: &[usize],
    _b_strides: &[usize],
    op: &F,
) -> Result<NDArrayWrapper, String>
where
    F: Fn(f64, f64) -> f64,
{
    let a_data: Vec<i32> = a_wrapper
        .to_f64_vec()
        .into_iter()
        .map(|x| x as i32)
        .collect();
    let b_data: Vec<i32> = b_wrapper
        .to_f64_vec()
        .into_iter()
        .map(|x| x as i32)
        .collect();

    let a_len: usize = a_shape.iter().product();
    let b_len: usize = b_shape.iter().product();

    let out_shape = if a_shape == b_shape {
        a_shape.to_vec()
    } else if a_len == 1 {
        b_shape.to_vec()
    } else if b_len == 1 {
        a_shape.to_vec()
    } else {
        return Err(format!("Shape mismatch: {:?} vs {:?}", a_shape, b_shape));
    };

    let out_len: usize = out_shape.iter().product();
    let mut result_data = Vec::with_capacity(out_len);

    if a_len == 1 && b_len == 1 {
        result_data.push(op(a_data[a_offset] as f64, b_data[b_offset] as f64) as i32);
    } else if a_len == 1 {
        let a_val = a_data[a_offset];
        for i in 0..out_len {
            result_data.push(op(a_val as f64, b_data[b_offset + i] as f64) as i32);
        }
    } else if b_len == 1 {
        let b_val = b_data[b_offset];
        for i in 0..out_len {
            result_data.push(op(a_data[a_offset + i] as f64, b_val as f64) as i32);
        }
    } else {
        for i in 0..out_len {
            result_data.push(op(a_data[a_offset + i] as f64, b_data[b_offset + i] as f64) as i32);
        }
    }

    let arr =
        ArrayD::<i32>::from_shape_vec(IxDyn(&out_shape), result_data).map_err(|e| e.to_string())?;

    Ok(NDArrayWrapper {
        data: ArrayData::Int32(Arc::new(RwLock::new(arr))),
        dtype: DType::Int32,
    })
}

// ============================================================================
// Scalar Operation Helpers
// ============================================================================

unsafe fn scalar_op_helper<F>(
    a_handle: *const NdArrayHandle,
    a_offset: usize,
    a_shape_ptr: *const usize,
    a_strides_ptr: *const usize,
    ndim: usize,
    scalar: f64,
    out_handle: *mut *mut NdArrayHandle,
    op: F,
) -> i32
where
    F: Fn(f64, f64) -> f64 + std::panic::UnwindSafe,
{
    if a_handle.is_null()
        || out_handle.is_null()
        || a_shape_ptr.is_null()
        || a_strides_ptr.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let a_wrapper = NdArrayHandle::as_wrapper(a_handle as *mut _);
        let a_shape = slice::from_raw_parts(a_shape_ptr, ndim);
        let _a_strides = slice::from_raw_parts(a_strides_ptr, ndim);

        let result = match a_wrapper.dtype {
            DType::Float64 => scalar_op_f64(a_wrapper, a_offset, a_shape, scalar, &op),
            DType::Float32 => scalar_op_f32(a_wrapper, a_offset, a_shape, scalar, &op),
            DType::Int64 => scalar_op_i64(a_wrapper, a_offset, a_shape, scalar, &op),
            DType::Int32 => scalar_op_i32(a_wrapper, a_offset, a_shape, scalar, &op),
            _ => {
                // Convert through f64
                match scalar_op_f64(a_wrapper, a_offset, a_shape, scalar, &op) {
                    Ok(wrapper) => convert_wrapper_dtype(wrapper, a_wrapper.dtype),
                    Err(e) => Err(e),
                }
            }
        };

        match result {
            Ok(wrapper) => {
                *out_handle = NdArrayHandle::from_wrapper(Box::new(wrapper));
                SUCCESS
            }
            Err(e) => {
                error::set_last_error(e);
                ERR_GENERIC
            }
        }
    })
}

unsafe fn scalar_op_f64<F>(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    scalar: f64,
    op: &F,
) -> Result<NDArrayWrapper, String>
where
    F: Fn(f64, f64) -> f64,
{
    let data = wrapper.to_f64_vec();
    let len: usize = shape.iter().product();
    let mut result_data = Vec::with_capacity(len);

    for i in 0..len {
        result_data.push(op(data[offset + i], scalar));
    }

    let arr =
        ArrayD::<f64>::from_shape_vec(IxDyn(shape), result_data).map_err(|e| e.to_string())?;

    Ok(NDArrayWrapper {
        data: ArrayData::Float64(Arc::new(RwLock::new(arr))),
        dtype: DType::Float64,
    })
}

unsafe fn scalar_op_f32<F>(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    scalar: f64,
    op: &F,
) -> Result<NDArrayWrapper, String>
where
    F: Fn(f64, f64) -> f64,
{
    let data: Vec<f32> = wrapper.to_f64_vec().into_iter().map(|x| x as f32).collect();
    let scalar_f32 = scalar as f32;
    let len: usize = shape.iter().product();
    let mut result_data = Vec::with_capacity(len);

    for i in 0..len {
        result_data.push(op(data[offset + i] as f64, scalar) as f32);
    }

    let arr =
        ArrayD::<f32>::from_shape_vec(IxDyn(shape), result_data).map_err(|e| e.to_string())?;

    Ok(NDArrayWrapper {
        data: ArrayData::Float32(Arc::new(RwLock::new(arr))),
        dtype: DType::Float32,
    })
}

unsafe fn scalar_op_i64<F>(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    scalar: f64,
    op: &F,
) -> Result<NDArrayWrapper, String>
where
    F: Fn(f64, f64) -> f64,
{
    let data: Vec<i64> = wrapper.to_f64_vec().into_iter().map(|x| x as i64).collect();
    let scalar_i64 = scalar as i64;
    let len: usize = shape.iter().product();
    let mut result_data = Vec::with_capacity(len);

    for i in 0..len {
        result_data.push(op(data[offset + i] as f64, scalar) as i64);
    }

    let arr =
        ArrayD::<i64>::from_shape_vec(IxDyn(shape), result_data).map_err(|e| e.to_string())?;

    Ok(NDArrayWrapper {
        data: ArrayData::Int64(Arc::new(RwLock::new(arr))),
        dtype: DType::Int64,
    })
}

unsafe fn scalar_op_i32<F>(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    scalar: f64,
    op: &F,
) -> Result<NDArrayWrapper, String>
where
    F: Fn(f64, f64) -> f64,
{
    let data: Vec<i32> = wrapper.to_f64_vec().into_iter().map(|x| x as i32).collect();
    let scalar_i32 = scalar as i32;
    let len: usize = shape.iter().product();
    let mut result_data = Vec::with_capacity(len);

    for i in 0..len {
        result_data.push(op(data[offset + i] as f64, scalar) as i32);
    }

    let arr =
        ArrayD::<i32>::from_shape_vec(IxDyn(shape), result_data).map_err(|e| e.to_string())?;

    Ok(NDArrayWrapper {
        data: ArrayData::Int32(Arc::new(RwLock::new(arr))),
        dtype: DType::Int32,
    })
}

// ============================================================================
// Public FFI Functions - Array-Array Operations
// ============================================================================

/// Add two arrays element-wise.
#[no_mangle]
pub unsafe extern "C" fn ndarray_add(
    a: *const NdArrayHandle,
    a_offset: usize,
    a_shape: *const usize,
    a_strides: *const usize,
    b: *const NdArrayHandle,
    b_offset: usize,
    b_shape: *const usize,
    b_strides: *const usize,
    ndim: usize,
    out: *mut *mut NdArrayHandle,
) -> i32 {
    binary_op_helper(
        a,
        a_offset,
        a_shape,
        a_strides,
        b,
        b_offset,
        b_shape,
        b_strides,
        ndim,
        out,
        "ndarray_add",
        |a, b| a + b,
    )
}

/// Subtract two arrays element-wise.
#[no_mangle]
pub unsafe extern "C" fn ndarray_sub(
    a: *const NdArrayHandle,
    a_offset: usize,
    a_shape: *const usize,
    a_strides: *const usize,
    b: *const NdArrayHandle,
    b_offset: usize,
    b_shape: *const usize,
    b_strides: *const usize,
    ndim: usize,
    out: *mut *mut NdArrayHandle,
) -> i32 {
    binary_op_helper(
        a,
        a_offset,
        a_shape,
        a_strides,
        b,
        b_offset,
        b_shape,
        b_strides,
        ndim,
        out,
        "ndarray_sub",
        |a, b| a - b,
    )
}

/// Multiply two arrays element-wise.
#[no_mangle]
pub unsafe extern "C" fn ndarray_mul(
    a: *const NdArrayHandle,
    a_offset: usize,
    a_shape: *const usize,
    a_strides: *const usize,
    b: *const NdArrayHandle,
    b_offset: usize,
    b_shape: *const usize,
    b_strides: *const usize,
    ndim: usize,
    out: *mut *mut NdArrayHandle,
) -> i32 {
    binary_op_helper(
        a,
        a_offset,
        a_shape,
        a_strides,
        b,
        b_offset,
        b_shape,
        b_strides,
        ndim,
        out,
        "ndarray_mul",
        |a, b| a * b,
    )
}

/// Divide two arrays element-wise.
#[no_mangle]
pub unsafe extern "C" fn ndarray_div(
    a: *const NdArrayHandle,
    a_offset: usize,
    a_shape: *const usize,
    a_strides: *const usize,
    b: *const NdArrayHandle,
    b_offset: usize,
    b_shape: *const usize,
    b_strides: *const usize,
    ndim: usize,
    out: *mut *mut NdArrayHandle,
) -> i32 {
    binary_op_helper(
        a,
        a_offset,
        a_shape,
        a_strides,
        b,
        b_offset,
        b_shape,
        b_strides,
        ndim,
        out,
        "ndarray_div",
        |a, b| a / b,
    )
}

/// Add scalar to array.
#[no_mangle]
pub unsafe extern "C" fn ndarray_add_scalar(
    a: *const NdArrayHandle,
    a_offset: usize,
    a_shape: *const usize,
    a_strides: *const usize,
    ndim: usize,
    scalar: f64,
    out: *mut *mut NdArrayHandle,
) -> i32 {
    scalar_op_helper(
        a,
        a_offset,
        a_shape,
        a_strides,
        ndim,
        scalar,
        out,
        |a, b| a + b,
    )
}

/// Subtract scalar from array.
#[no_mangle]
pub unsafe extern "C" fn ndarray_sub_scalar(
    a: *const NdArrayHandle,
    a_offset: usize,
    a_shape: *const usize,
    a_strides: *const usize,
    ndim: usize,
    scalar: f64,
    out: *mut *mut NdArrayHandle,
) -> i32 {
    scalar_op_helper(
        a,
        a_offset,
        a_shape,
        a_strides,
        ndim,
        scalar,
        out,
        |a, b| a - b,
    )
}

/// Multiply array by scalar.
#[no_mangle]
pub unsafe extern "C" fn ndarray_mul_scalar(
    a: *const NdArrayHandle,
    a_offset: usize,
    a_shape: *const usize,
    a_strides: *const usize,
    ndim: usize,
    scalar: f64,
    out: *mut *mut NdArrayHandle,
) -> i32 {
    scalar_op_helper(
        a,
        a_offset,
        a_shape,
        a_strides,
        ndim,
        scalar,
        out,
        |a, b| a * b,
    )
}

/// Divide array by scalar.
#[no_mangle]
pub unsafe extern "C" fn ndarray_div_scalar(
    a: *const NdArrayHandle,
    a_offset: usize,
    a_shape: *const usize,
    a_strides: *const usize,
    ndim: usize,
    scalar: f64,
    out: *mut *mut NdArrayHandle,
) -> i32 {
    scalar_op_helper(
        a,
        a_offset,
        a_shape,
        a_strides,
        ndim,
        scalar,
        out,
        |a, b| a / b,
    )
}
