//! Math operation helpers for type-specific unary and binary operations.
//!
//! This module provides macros and helper functions for performing mathematical
//! operations while preserving native types and avoiding unnecessary conversions.

use ndarray::{ArrayD, IxDyn};
use parking_lot::RwLock;
use std::sync::Arc;

use crate::core::view_helpers::{
    extract_view_as_f32, extract_view_as_f64, extract_view_as_i16, extract_view_as_i32,
    extract_view_as_i64, extract_view_as_i8, extract_view_as_u16, extract_view_as_u32,
    extract_view_as_u64, extract_view_as_u8, extract_view_f32, extract_view_f64, extract_view_i16,
    extract_view_i32, extract_view_i64, extract_view_i8, extract_view_u16, extract_view_u32,
    extract_view_u64, extract_view_u8,
};
use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;

/// Macro to define unary operation helpers for all types.
///
/// Generates type-specific helper functions that extract views and apply
/// the operation without converting to f64.
#[macro_export]
macro_rules! define_unary_helpers {
    ($($name:ident, $type:ty, $extract:ident, $dtype:expr, $variant:path),*) => {
        $(
            /// Apply a unary operation to an array of the specific type.
            pub unsafe fn $name<F>(
                wrapper: &NDArrayWrapper,
                offset: usize,
                shape: &[usize],
                strides: &[usize],
                op: F,
            ) -> Result<NDArrayWrapper, String>
            where
                F: Fn($type) -> $type,
            {
                if let Some(view) = $extract(wrapper, offset, shape, strides) {
                    let result = view.mapv(op);
                    let arr = ArrayD::<$type>::from_shape_vec(IxDyn(shape), result.iter().cloned().collect())
                        .map_err(|e| e.to_string())?;
                    return Ok(NDArrayWrapper {
                        data: $variant(Arc::new(RwLock::new(arr))),
                        dtype: $dtype,
                    });
                }
                Err(format!("Cannot extract view for type {}", stringify!($type)))
            }
        )*
    };
}

// Generate unary helpers for all types
define_unary_helpers!(
    unary_op_f64,
    f64,
    extract_view_f64,
    DType::Float64,
    ArrayData::Float64,
    unary_op_f32,
    f32,
    extract_view_f32,
    DType::Float32,
    ArrayData::Float32,
    unary_op_i64,
    i64,
    extract_view_i64,
    DType::Int64,
    ArrayData::Int64,
    unary_op_i32,
    i32,
    extract_view_i32,
    DType::Int32,
    ArrayData::Int32,
    unary_op_i16,
    i16,
    extract_view_i16,
    DType::Int16,
    ArrayData::Int16,
    unary_op_i8,
    i8,
    extract_view_i8,
    DType::Int8,
    ArrayData::Int8,
    unary_op_u64,
    u64,
    extract_view_u64,
    DType::Uint64,
    ArrayData::Uint64,
    unary_op_u32,
    u32,
    extract_view_u32,
    DType::Uint32,
    ArrayData::Uint32,
    unary_op_u16,
    u16,
    extract_view_u16,
    DType::Uint16,
    ArrayData::Uint16,
    unary_op_u8,
    u8,
    extract_view_u8,
    DType::Uint8,
    ArrayData::Uint8
);

/// Macro to define scalar operation helpers for all types.
#[macro_export]
macro_rules! define_scalar_helpers {
    ($($name:ident, $type:ty, $extract:ident, $dtype:expr, $variant:path),*) => {
        $(
            /// Apply a binary operation with a scalar to an array of the specific type.
            pub unsafe fn $name<F>(
                wrapper: &NDArrayWrapper,
                offset: usize,
                shape: &[usize],
                strides: &[usize],
                scalar: $type,
                op: F,
            ) -> Result<NDArrayWrapper, String>
            where
                F: Fn($type, $type) -> $type,
            {
                if let Some(view) = $extract(wrapper, offset, shape, strides) {
                    let result = view.mapv(|x| op(x, scalar));
                    let arr = ArrayD::<$type>::from_shape_vec(IxDyn(shape), result.iter().cloned().collect())
                        .map_err(|e| e.to_string())?;
                    return Ok(NDArrayWrapper {
                        data: $variant(Arc::new(RwLock::new(arr))),
                        dtype: $dtype,
                    });
                }
                Err(format!("Cannot extract view for type {}", stringify!($type)))
            }
        )*
    };
}

// Generate scalar helpers for all types
define_scalar_helpers!(
    scalar_op_f64,
    f64,
    extract_view_f64,
    DType::Float64,
    ArrayData::Float64,
    scalar_op_f32,
    f32,
    extract_view_f32,
    DType::Float32,
    ArrayData::Float32,
    scalar_op_i64,
    i64,
    extract_view_i64,
    DType::Int64,
    ArrayData::Int64,
    scalar_op_i32,
    i32,
    extract_view_i32,
    DType::Int32,
    ArrayData::Int32,
    scalar_op_i16,
    i16,
    extract_view_i16,
    DType::Int16,
    ArrayData::Int16,
    scalar_op_i8,
    i8,
    extract_view_i8,
    DType::Int8,
    ArrayData::Int8,
    scalar_op_u64,
    u64,
    extract_view_u64,
    DType::Uint64,
    ArrayData::Uint64,
    scalar_op_u32,
    u32,
    extract_view_u32,
    DType::Uint32,
    ArrayData::Uint32,
    scalar_op_u16,
    u16,
    extract_view_u16,
    DType::Uint16,
    ArrayData::Uint16,
    scalar_op_u8,
    u8,
    extract_view_u8,
    DType::Uint8,
    ArrayData::Uint8
);

/// Helper to dispatch unary operations based on dtype.
///
/// For operations that work on all types (like abs).
pub unsafe fn unary_op_generic<
    F64,
    F32,
    F64Op,
    F32Op,
    I64Op,
    I32Op,
    I16Op,
    I8Op,
    U64Op,
    U32Op,
    U16Op,
    U8Op,
>(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
    f64_op: F64Op,
    f32_op: F32Op,
    i64_op: I64Op,
    i32_op: I32Op,
    i16_op: I16Op,
    i8_op: I8Op,
    u64_op: U64Op,
    u32_op: U32Op,
    u16_op: U16Op,
    u8_op: U8Op,
) -> Result<NDArrayWrapper, String>
where
    F64Op: Fn(f64) -> f64,
    F32Op: Fn(f32) -> f32,
    I64Op: Fn(i64) -> i64,
    I32Op: Fn(i32) -> i32,
    I16Op: Fn(i16) -> i16,
    I8Op: Fn(i8) -> i8,
    U64Op: Fn(u64) -> u64,
    U32Op: Fn(u32) -> u32,
    U16Op: Fn(u16) -> u16,
    U8Op: Fn(u8) -> u8,
{
    match wrapper.dtype {
        DType::Float64 => unary_op_f64(wrapper, offset, shape, strides, f64_op),
        DType::Float32 => unary_op_f32(wrapper, offset, shape, strides, f32_op),
        DType::Int64 => unary_op_i64(wrapper, offset, shape, strides, i64_op),
        DType::Int32 => unary_op_i32(wrapper, offset, shape, strides, i32_op),
        DType::Int16 => unary_op_i16(wrapper, offset, shape, strides, i16_op),
        DType::Int8 => unary_op_i8(wrapper, offset, shape, strides, i8_op),
        DType::Uint64 => unary_op_u64(wrapper, offset, shape, strides, u64_op),
        DType::Uint32 => unary_op_u32(wrapper, offset, shape, strides, u32_op),
        DType::Uint16 => unary_op_u16(wrapper, offset, shape, strides, u16_op),
        DType::Uint8 => unary_op_u8(wrapper, offset, shape, strides, u8_op),
        DType::Bool => Err("Operation not supported for Bool type".to_string()),
    }
}

/// Helper to dispatch float-only unary operations.
///
/// For operations that only make sense on floats (like floor, sqrt, etc.)
pub unsafe fn unary_op_float_only<F64, F32>(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
    f64_op: F64,
    f32_op: F32,
    op_name: &str,
) -> Result<NDArrayWrapper, String>
where
    F64: Fn(f64) -> f64,
    F32: Fn(f32) -> f32,
{
    match wrapper.dtype {
        DType::Float64 => unary_op_f64(wrapper, offset, shape, strides, f64_op),
        DType::Float32 => unary_op_f32(wrapper, offset, shape, strides, f32_op),
        _ => Err(format!(
            "{}() requires float type (Float64 or Float32)",
            op_name
        )),
    }
}

/// Helper to dispatch scalar operations based on dtype.
pub unsafe fn scalar_op_generic<
    F64,
    F32,
    F64Op,
    F32Op,
    I64Op,
    I32Op,
    I16Op,
    I8Op,
    U64Op,
    U32Op,
    U16Op,
    U8Op,
>(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
    scalar: f64,
    f64_op: F64Op,
    f32_op: F32Op,
    i64_op: I64Op,
    i32_op: I32Op,
    i16_op: I16Op,
    i8_op: I8Op,
    u64_op: U64Op,
    u32_op: U32Op,
    u16_op: U16Op,
    u8_op: U8Op,
) -> Result<NDArrayWrapper, String>
where
    F64Op: Fn(f64, f64) -> f64,
    F32Op: Fn(f32, f32) -> f32,
    I64Op: Fn(i64, i64) -> i64,
    I32Op: Fn(i32, i32) -> i32,
    I16Op: Fn(i16, i16) -> i16,
    I8Op: Fn(i8, i8) -> i8,
    U64Op: Fn(u64, u64) -> u64,
    U32Op: Fn(u32, u32) -> u32,
    U16Op: Fn(u16, u16) -> u16,
    U8Op: Fn(u8, u8) -> u8,
{
    match wrapper.dtype {
        DType::Float64 => scalar_op_f64(wrapper, offset, shape, strides, scalar as f64, f64_op),
        DType::Float32 => scalar_op_f32(wrapper, offset, shape, strides, scalar as f32, f32_op),
        DType::Int64 => scalar_op_i64(wrapper, offset, shape, strides, scalar as i64, i64_op),
        DType::Int32 => scalar_op_i32(wrapper, offset, shape, strides, scalar as i32, i32_op),
        DType::Int16 => scalar_op_i16(wrapper, offset, shape, strides, scalar as i16, i16_op),
        DType::Int8 => scalar_op_i8(wrapper, offset, shape, strides, scalar as i8, i8_op),
        DType::Uint64 => scalar_op_u64(wrapper, offset, shape, strides, scalar as u64, u64_op),
        DType::Uint32 => scalar_op_u32(wrapper, offset, shape, strides, scalar as u32, u32_op),
        DType::Uint16 => scalar_op_u16(wrapper, offset, shape, strides, scalar as u16, u16_op),
        DType::Uint8 => scalar_op_u8(wrapper, offset, shape, strides, scalar as u8, u8_op),
        DType::Bool => Err("Operation not supported for Bool type".to_string()),
    }
}

/// Helper to dispatch float-only scalar operations.
pub unsafe fn scalar_op_float_only<F64, F32>(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
    scalar: f64,
    f64_op: F64,
    f32_op: F32,
    op_name: &str,
) -> Result<NDArrayWrapper, String>
where
    F64: Fn(f64, f64) -> f64,
    F32: Fn(f32, f32) -> f32,
{
    match wrapper.dtype {
        DType::Float64 => scalar_op_f64(wrapper, offset, shape, strides, scalar as f64, f64_op),
        DType::Float32 => scalar_op_f32(wrapper, offset, shape, strides, scalar as f32, f32_op),
        _ => Err(format!(
            "{} with scalar requires float type (Float64 or Float32)",
            op_name
        )),
    }
}

// ============================================================================
// Binary Operations
// ============================================================================

/// Check if two shapes are broadcast compatible.
///
/// Broadcasting rules:
/// - Dimensions are compared from right to left
/// - Dimensions must be equal, or one of them must be 1
/// - If shapes have different lengths, the shorter is padded with 1s on the left
pub fn are_broadcast_compatible(a_shape: &[usize], b_shape: &[usize]) -> bool {
    let mut a_iter = a_shape.iter().rev();
    let mut b_iter = b_shape.iter().rev();

    loop {
        match (a_iter.next(), b_iter.next()) {
            (Some(&a_dim), Some(&b_dim)) => {
                if a_dim != b_dim && a_dim != 1 && b_dim != 1 {
                    return false;
                }
            }
            (Some(_), None) | (None, Some(_)) => {
                // Different number of dimensions - this is ok, remaining dims checked above
                continue;
            }
            (None, None) => break,
        }
    }

    true
}

/// Compute the broadcast output shape from two input shapes.
pub fn compute_broadcast_shape(a_shape: &[usize], b_shape: &[usize]) -> Vec<usize> {
    let a_len = a_shape.len();
    let b_len = b_shape.len();
    let out_len = a_len.max(b_len);
    let mut result = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let a_idx = a_len.saturating_sub(out_len - i);
        let b_idx = b_len.saturating_sub(out_len - i);

        let a_dim = if a_idx < a_len { a_shape[a_idx] } else { 1 };
        let b_dim = if b_idx < b_len { b_shape[b_idx] } else { 1 };

        result.push(a_dim.max(b_dim));
    }

    result
}

/// Macro to define binary operation helpers for all types.
///
/// Uses extract_view_as_* functions to handle mixed types by converting
/// inputs to the target type. Silently truncates values that don't fit.
#[macro_export]
macro_rules! define_binary_helpers {
    ($($name:ident, $type:ty, $extract_as:ident, $dtype:expr, $variant:path),*) => {
        $(
            /// Apply a binary operation between two arrays.
            ///
            /// This function converts both inputs to the target type if necessary,
            /// allowing operations between different types (e.g., Int32 + Int16 -> Int32).
            ///
            /// Validates broadcasting compatibility before extraction to prevent
            /// capacity overflow panics on incompatible shapes.
            pub unsafe fn $name<F>(
                a_wrapper: &NDArrayWrapper,
                a_offset: usize,
                a_shape: &[usize],
                a_strides: &[usize],
                b_wrapper: &NDArrayWrapper,
                b_offset: usize,
                b_shape: &[usize],
                b_strides: &[usize],
                op: F,
            ) -> Result<NDArrayWrapper, String>
            where
                F: Fn($type, $type) -> $type,
            {
                // Check broadcasting compatibility first
                if !are_broadcast_compatible(a_shape, b_shape) {
                    return Err(format!(
                        "Shapes {:?} and {:?} are not broadcast compatible",
                        a_shape, b_shape
                    ));
                }

                // Use extract_view_as_* to handle mixed types via conversion
                let a_arr = $extract_as(a_wrapper, a_offset, a_shape, a_strides)
                    .ok_or_else(|| format!("Cannot extract view as {} for operand a", stringify!($type)))?;
                let b_arr = $extract_as(b_wrapper, b_offset, b_shape, b_strides)
                    .ok_or_else(|| format!("Cannot extract view as {} for operand b", stringify!($type)))?;

                // Use ndarray's Zip for element-wise operation
                let result = ndarray::Zip::from(&a_arr).and(&b_arr).map_collect(|&a, &b| op(a, b));

                Ok(NDArrayWrapper {
                    data: $variant(Arc::new(RwLock::new(result))),
                    dtype: $dtype,
                })
            }
        )*
    };
}

// Generate binary helpers for all types using extract_view_as_*
// This allows each helper to accept any input type and convert to the target type
define_binary_helpers!(
    binary_op_f64,
    f64,
    extract_view_as_f64,
    DType::Float64,
    ArrayData::Float64,
    binary_op_f32,
    f32,
    extract_view_as_f32,
    DType::Float32,
    ArrayData::Float32,
    binary_op_i64,
    i64,
    extract_view_as_i64,
    DType::Int64,
    ArrayData::Int64,
    binary_op_i32,
    i32,
    extract_view_as_i32,
    DType::Int32,
    ArrayData::Int32,
    binary_op_i16,
    i16,
    extract_view_as_i16,
    DType::Int16,
    ArrayData::Int16,
    binary_op_i8,
    i8,
    extract_view_as_i8,
    DType::Int8,
    ArrayData::Int8,
    binary_op_u64,
    u64,
    extract_view_as_u64,
    DType::Uint64,
    ArrayData::Uint64,
    binary_op_u32,
    u32,
    extract_view_as_u32,
    DType::Uint32,
    ArrayData::Uint32,
    binary_op_u16,
    u16,
    extract_view_as_u16,
    DType::Uint16,
    ArrayData::Uint16,
    binary_op_u8,
    u8,
    extract_view_as_u8,
    DType::Uint8,
    ArrayData::Uint8
);

/// Get the promoted dtype for binary operations (same as in helpers.rs).
pub fn promote_dtypes(a: DType, b: DType) -> Option<DType> {
    use DType::*;

    if a == Float64 || b == Float64 {
        return Some(Float64);
    }
    if a == Float32 || b == Float32 {
        return Some(Float32);
    }
    if a == Int64 || b == Int64 {
        return Some(Int64);
    }
    if a == Int32 || b == Int32 {
        return Some(Int32);
    }
    if a == Int16 || b == Int16 {
        return Some(Int16);
    }
    if a == Uint64 || b == Uint64 {
        return Some(Uint64);
    }
    if a == Uint32 || b == Uint32 {
        return Some(Uint32);
    }
    if a == Uint16 || b == Uint16 {
        return Some(Uint16);
    }
    if a == b {
        return Some(a);
    }

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
