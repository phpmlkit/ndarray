//! View extraction helpers for proper strided array views.
//!
//! This module provides the CORRECT way to extract views from ArrayData,
//! properly handling offset, shape, and strides for non-contiguous arrays.

use ndarray::{ArrayD, ArrayViewD, IxDyn, ShapeBuilder};
use parking_lot::RwLock;
use std::sync::Arc;

use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;

/// Macro to generate extract_view functions for each type
macro_rules! define_extract_view {
    ($name:ident, $variant:path, $type:ty) => {
        /// Extract a view of the specific type from the wrapper.
        ///
        /// # Safety
        /// The caller must ensure offset/shape/strides are valid.
        pub unsafe fn $name<'a>(
            wrapper: &'a NDArrayWrapper,
            offset: usize,
            shape: &'a [usize],
            strides: &'a [usize],
        ) -> Option<ArrayViewD<'a, $type>> {
            match &wrapper.data {
                $variant(arr) => {
                    let guard = arr.read();
                    let ptr = guard.as_ptr();
                    let view_ptr = ptr.add(offset);
                    let strides_ix = IxDyn(strides);
                    ArrayViewD::<$type>::from_shape_ptr(IxDyn(shape).strides(strides_ix), view_ptr)
                        .into()
                }
                _ => None,
            }
        }
    };
}

// Generate extract_view functions for all types
define_extract_view!(extract_view_f64, ArrayData::Float64, f64);
define_extract_view!(extract_view_f32, ArrayData::Float32, f32);
define_extract_view!(extract_view_i64, ArrayData::Int64, i64);
define_extract_view!(extract_view_i32, ArrayData::Int32, i32);
define_extract_view!(extract_view_i16, ArrayData::Int16, i16);
define_extract_view!(extract_view_i8, ArrayData::Int8, i8);
define_extract_view!(extract_view_u64, ArrayData::Uint64, u64);
define_extract_view!(extract_view_u32, ArrayData::Uint32, u32);
define_extract_view!(extract_view_u16, ArrayData::Uint16, u16);
define_extract_view!(extract_view_u8, ArrayData::Uint8, u8);

/// Extract a view and convert to f64.
///
/// For integer types, this converts each element to f64.
/// For float types, this casts to f64.
pub fn extract_view_as_f64(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> Option<ArrayD<f64>> {
    unsafe {
        // Try to extract native view and convert
        if let Some(view) = extract_view_f64(wrapper, offset, shape, strides) {
            return Some(view.to_owned());
        }
        if let Some(view) = extract_view_f32(wrapper, offset, shape, strides) {
            return Some(view.mapv(|x| x as f64).to_owned());
        }
        if let Some(view) = extract_view_i64(wrapper, offset, shape, strides) {
            return Some(view.mapv(|x| x as f64).to_owned());
        }
        if let Some(view) = extract_view_i32(wrapper, offset, shape, strides) {
            return Some(view.mapv(|x| x as f64).to_owned());
        }
        if let Some(view) = extract_view_i16(wrapper, offset, shape, strides) {
            return Some(view.mapv(|x| x as f64).to_owned());
        }
        if let Some(view) = extract_view_i8(wrapper, offset, shape, strides) {
            return Some(view.mapv(|x| x as f64).to_owned());
        }
        if let Some(view) = extract_view_u64(wrapper, offset, shape, strides) {
            return Some(view.mapv(|x| x as f64).to_owned());
        }
        if let Some(view) = extract_view_u32(wrapper, offset, shape, strides) {
            return Some(view.mapv(|x| x as f64).to_owned());
        }
        if let Some(view) = extract_view_u16(wrapper, offset, shape, strides) {
            return Some(view.mapv(|x| x as f64).to_owned());
        }
        if let Some(view) = extract_view_u8(wrapper, offset, shape, strides) {
            return Some(view.mapv(|x| x as f64).to_owned());
        }
    }
    None
}

/// Extract a view and convert to f32.
///
/// For small integer types (i8, i16, u8, u16), this uses f32.
/// For larger types, may lose precision.
pub fn extract_view_as_f32(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> Option<ArrayD<f32>> {
    unsafe {
        if let Some(view) = extract_view_f32(wrapper, offset, shape, strides) {
            return Some(view.to_owned());
        }
        if let Some(view) = extract_view_f64(wrapper, offset, shape, strides) {
            return Some(view.mapv(|x| x as f32).to_owned());
        }
        if let Some(view) = extract_view_i32(wrapper, offset, shape, strides) {
            return Some(view.mapv(|x| x as f32).to_owned());
        }
        if let Some(view) = extract_view_i16(wrapper, offset, shape, strides) {
            return Some(view.mapv(|x| x as f32).to_owned());
        }
        if let Some(view) = extract_view_i8(wrapper, offset, shape, strides) {
            return Some(view.mapv(|x| x as f32).to_owned());
        }
        if let Some(view) = extract_view_u32(wrapper, offset, shape, strides) {
            return Some(view.mapv(|x| x as f32).to_owned());
        }
        if let Some(view) = extract_view_u16(wrapper, offset, shape, strides) {
            return Some(view.mapv(|x| x as f32).to_owned());
        }
        if let Some(view) = extract_view_u8(wrapper, offset, shape, strides) {
            return Some(view.mapv(|x| x as f32).to_owned());
        }
        // For i64/u64, we still convert but may lose precision
        if let Some(view) = extract_view_i64(wrapper, offset, shape, strides) {
            return Some(view.mapv(|x| x as f32).to_owned());
        }
        if let Some(view) = extract_view_u64(wrapper, offset, shape, strides) {
            return Some(view.mapv(|x| x as f32).to_owned());
        }
    }
    None
}

// ============================================================================
// Scalar Wrapper Creation Helpers
// ============================================================================

/// Create a 0-dimensional (scalar) array wrapper from an f64 value.
pub fn create_scalar_wrapper_f64(value: f64) -> NDArrayWrapper {
    let shape = vec![];
    let arr = ArrayD::<f64>::from_shape_vec(IxDyn(&shape), vec![value])
        .expect("Failed to create scalar array");
    NDArrayWrapper {
        data: ArrayData::Float64(Arc::new(RwLock::new(arr))),
        dtype: DType::Float64,
    }
}

/// Create a 0-dimensional (scalar) array wrapper preserving dtype.
pub fn create_scalar_wrapper_with_dtype(value: f64, dtype: DType) -> NDArrayWrapper {
    let shape = vec![];

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
    let shape = vec![];
    let arr = ArrayD::<i64>::from_shape_vec(IxDyn(&shape), vec![value])
        .expect("Failed to create scalar array");
    NDArrayWrapper {
        data: ArrayData::Int64(Arc::new(RwLock::new(arr))),
        dtype: DType::Int64,
    }
}
