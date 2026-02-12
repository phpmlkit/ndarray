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

/// Extract data from a view using the fastest method available.
///
/// For contiguous views, uses `copy_from_slice` for maximum performance.
/// For non-contiguous views, falls back to element-by-element iteration.
fn extract_view_data<T: Copy>(view: ArrayViewD<T>) -> Vec<T> {
    let size = view.len();
    let mut result = Vec::with_capacity(size);

    // Check if view is contiguous
    if view.is_standard_layout() {
        // Fast path: contiguous memory
        if let Some(slice) = view.as_slice() {
            result.extend_from_slice(slice);
            return result;
        }
    }

    // Fallback: iterate element by element
    for &item in view.iter() {
        result.push(item);
    }
    result
}

/// Extract a view and convert to f64.
///
/// For integer types, this converts each element to f64.
/// For float types, this casts to f64.
/// Uses optimized contiguous copy when possible.
pub fn extract_view_as_f64(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> Option<ArrayD<f64>> {
    unsafe {
        if let Some(view) = extract_view_f64(wrapper, offset, shape, strides) {
            let data = extract_view_data(view);
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_f32(wrapper, offset, shape, strides) {
            let data: Vec<f64> = extract_view_data(view)
                .into_iter()
                .map(|x| x as f64)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i64(wrapper, offset, shape, strides) {
            let data: Vec<f64> = extract_view_data(view)
                .into_iter()
                .map(|x| x as f64)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i32(wrapper, offset, shape, strides) {
            let data: Vec<f64> = extract_view_data(view)
                .into_iter()
                .map(|x| x as f64)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i16(wrapper, offset, shape, strides) {
            let data: Vec<f64> = extract_view_data(view)
                .into_iter()
                .map(|x| x as f64)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i8(wrapper, offset, shape, strides) {
            let data: Vec<f64> = extract_view_data(view)
                .into_iter()
                .map(|x| x as f64)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u64(wrapper, offset, shape, strides) {
            let data: Vec<f64> = extract_view_data(view)
                .into_iter()
                .map(|x| x as f64)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u32(wrapper, offset, shape, strides) {
            let data: Vec<f64> = extract_view_data(view)
                .into_iter()
                .map(|x| x as f64)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u16(wrapper, offset, shape, strides) {
            let data: Vec<f64> = extract_view_data(view)
                .into_iter()
                .map(|x| x as f64)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u8(wrapper, offset, shape, strides) {
            let data: Vec<f64> = extract_view_data(view)
                .into_iter()
                .map(|x| x as f64)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
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
            let data = extract_view_data(view);
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_f64(wrapper, offset, shape, strides) {
            let data: Vec<f32> = extract_view_data(view)
                .into_iter()
                .map(|x| x as f32)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i32(wrapper, offset, shape, strides) {
            let data: Vec<f32> = extract_view_data(view)
                .into_iter()
                .map(|x| x as f32)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i16(wrapper, offset, shape, strides) {
            let data: Vec<f32> = extract_view_data(view)
                .into_iter()
                .map(|x| x as f32)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i8(wrapper, offset, shape, strides) {
            let data: Vec<f32> = extract_view_data(view)
                .into_iter()
                .map(|x| x as f32)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u32(wrapper, offset, shape, strides) {
            let data: Vec<f32> = extract_view_data(view)
                .into_iter()
                .map(|x| x as f32)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u16(wrapper, offset, shape, strides) {
            let data: Vec<f32> = extract_view_data(view)
                .into_iter()
                .map(|x| x as f32)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u8(wrapper, offset, shape, strides) {
            let data: Vec<f32> = extract_view_data(view)
                .into_iter()
                .map(|x| x as f32)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        // For i64/u64, we still convert but may lose precision
        if let Some(view) = extract_view_i64(wrapper, offset, shape, strides) {
            let data: Vec<f32> = extract_view_data(view)
                .into_iter()
                .map(|x| x as f32)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u64(wrapper, offset, shape, strides) {
            let data: Vec<f32> = extract_view_data(view)
                .into_iter()
                .map(|x| x as f32)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
    }
    None
}

/// Extract a view and convert to i64.
///
/// Silently truncates values that don't fit in i64.
pub fn extract_view_as_i64(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> Option<ArrayD<i64>> {
    unsafe {
        if let Some(view) = extract_view_i64(wrapper, offset, shape, strides) {
            let data = extract_view_data(view);
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i32(wrapper, offset, shape, strides) {
            let data: Vec<i64> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i64)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i16(wrapper, offset, shape, strides) {
            let data: Vec<i64> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i64)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i8(wrapper, offset, shape, strides) {
            let data: Vec<i64> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i64)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u64(wrapper, offset, shape, strides) {
            let data: Vec<i64> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i64)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u32(wrapper, offset, shape, strides) {
            let data: Vec<i64> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i64)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u16(wrapper, offset, shape, strides) {
            let data: Vec<i64> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i64)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u8(wrapper, offset, shape, strides) {
            let data: Vec<i64> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i64)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        // Floats truncate toward zero
        if let Some(view) = extract_view_f64(wrapper, offset, shape, strides) {
            let data: Vec<i64> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i64)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_f32(wrapper, offset, shape, strides) {
            let data: Vec<i64> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i64)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
    }
    None
}

/// Extract a view and convert to i32.
///
/// Silently truncates values that don't fit in i32.
pub fn extract_view_as_i32(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> Option<ArrayD<i32>> {
    unsafe {
        if let Some(view) = extract_view_i32(wrapper, offset, shape, strides) {
            let data = extract_view_data(view);
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i64(wrapper, offset, shape, strides) {
            let data: Vec<i32> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i32)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i16(wrapper, offset, shape, strides) {
            let data: Vec<i32> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i32)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i8(wrapper, offset, shape, strides) {
            let data: Vec<i32> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i32)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u32(wrapper, offset, shape, strides) {
            let data: Vec<i32> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i32)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u16(wrapper, offset, shape, strides) {
            let data: Vec<i32> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i32)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u8(wrapper, offset, shape, strides) {
            let data: Vec<i32> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i32)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        // Floats truncate toward zero
        if let Some(view) = extract_view_f64(wrapper, offset, shape, strides) {
            let data: Vec<i32> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i32)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_f32(wrapper, offset, shape, strides) {
            let data: Vec<i32> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i32)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        // u64 last (may lose data)
        if let Some(view) = extract_view_u64(wrapper, offset, shape, strides) {
            let data: Vec<i32> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i32)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
    }
    None
}

/// Extract a view and convert to i16.
///
/// Silently truncates values that don't fit in i16.
pub fn extract_view_as_i16(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> Option<ArrayD<i16>> {
    unsafe {
        if let Some(view) = extract_view_i16(wrapper, offset, shape, strides) {
            let data = extract_view_data(view);
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i8(wrapper, offset, shape, strides) {
            let data: Vec<i16> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i16)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u16(wrapper, offset, shape, strides) {
            let data: Vec<i16> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i16)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u8(wrapper, offset, shape, strides) {
            let data: Vec<i16> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i16)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        // Everything else truncates
        if let Some(view) = extract_view_i32(wrapper, offset, shape, strides) {
            let data: Vec<i16> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i16)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i64(wrapper, offset, shape, strides) {
            let data: Vec<i16> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i16)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u32(wrapper, offset, shape, strides) {
            let data: Vec<i16> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i16)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u64(wrapper, offset, shape, strides) {
            let data: Vec<i16> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i16)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_f64(wrapper, offset, shape, strides) {
            let data: Vec<i16> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i16)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_f32(wrapper, offset, shape, strides) {
            let data: Vec<i16> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i16)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
    }
    None
}

/// Extract a view and convert to i8.
///
/// Silently truncates values that don't fit in i8.
pub fn extract_view_as_i8(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> Option<ArrayD<i8>> {
    unsafe {
        if let Some(view) = extract_view_i8(wrapper, offset, shape, strides) {
            let data = extract_view_data(view);
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u8(wrapper, offset, shape, strides) {
            let data: Vec<i8> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i8)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        // Everything else truncates
        if let Some(view) = extract_view_i16(wrapper, offset, shape, strides) {
            let data: Vec<i8> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i8)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i32(wrapper, offset, shape, strides) {
            let data: Vec<i8> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i8)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i64(wrapper, offset, shape, strides) {
            let data: Vec<i8> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i8)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u16(wrapper, offset, shape, strides) {
            let data: Vec<i8> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i8)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u32(wrapper, offset, shape, strides) {
            let data: Vec<i8> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i8)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u64(wrapper, offset, shape, strides) {
            let data: Vec<i8> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i8)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_f64(wrapper, offset, shape, strides) {
            let data: Vec<i8> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i8)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_f32(wrapper, offset, shape, strides) {
            let data: Vec<i8> = extract_view_data(view)
                .into_iter()
                .map(|x| x as i8)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
    }
    None
}

/// Extract a view and convert to u64.
///
/// Silently truncates negative values.
pub fn extract_view_as_u64(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> Option<ArrayD<u64>> {
    unsafe {
        if let Some(view) = extract_view_u64(wrapper, offset, shape, strides) {
            let data = extract_view_data(view);
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u32(wrapper, offset, shape, strides) {
            let data: Vec<u64> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u64)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u16(wrapper, offset, shape, strides) {
            let data: Vec<u64> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u64)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u8(wrapper, offset, shape, strides) {
            let data: Vec<u64> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u64)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i64(wrapper, offset, shape, strides) {
            let data: Vec<u64> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u64)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i32(wrapper, offset, shape, strides) {
            let data: Vec<u64> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u64)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i16(wrapper, offset, shape, strides) {
            let data: Vec<u64> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u64)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i8(wrapper, offset, shape, strides) {
            let data: Vec<u64> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u64)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_f64(wrapper, offset, shape, strides) {
            let data: Vec<u64> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u64)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_f32(wrapper, offset, shape, strides) {
            let data: Vec<u64> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u64)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
    }
    None
}

/// Extract a view and convert to u32.
///
/// Silently truncates values that don't fit in u32.
pub fn extract_view_as_u32(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> Option<ArrayD<u32>> {
    unsafe {
        if let Some(view) = extract_view_u32(wrapper, offset, shape, strides) {
            let data = extract_view_data(view);
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u16(wrapper, offset, shape, strides) {
            let data: Vec<u32> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u32)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u8(wrapper, offset, shape, strides) {
            let data: Vec<u32> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u32)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i32(wrapper, offset, shape, strides) {
            let data: Vec<u32> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u32)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i16(wrapper, offset, shape, strides) {
            let data: Vec<u32> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u32)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i8(wrapper, offset, shape, strides) {
            let data: Vec<u32> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u32)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_f64(wrapper, offset, shape, strides) {
            let data: Vec<u32> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u32)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_f32(wrapper, offset, shape, strides) {
            let data: Vec<u32> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u32)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u64(wrapper, offset, shape, strides) {
            let data: Vec<u32> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u32)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i64(wrapper, offset, shape, strides) {
            let data: Vec<u32> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u32)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
    }
    None
}

/// Extract a view and convert to u16.
///
/// Silently truncates values that don't fit in u16.
pub fn extract_view_as_u16(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> Option<ArrayD<u16>> {
    unsafe {
        if let Some(view) = extract_view_u16(wrapper, offset, shape, strides) {
            let data = extract_view_data(view);
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u8(wrapper, offset, shape, strides) {
            let data: Vec<u16> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u16)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i16(wrapper, offset, shape, strides) {
            let data: Vec<u16> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u16)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i8(wrapper, offset, shape, strides) {
            let data: Vec<u16> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u16)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u32(wrapper, offset, shape, strides) {
            let data: Vec<u16> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u16)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i32(wrapper, offset, shape, strides) {
            let data: Vec<u16> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u16)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u64(wrapper, offset, shape, strides) {
            let data: Vec<u16> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u16)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i64(wrapper, offset, shape, strides) {
            let data: Vec<u16> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u16)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_f64(wrapper, offset, shape, strides) {
            let data: Vec<u16> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u16)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_f32(wrapper, offset, shape, strides) {
            let data: Vec<u16> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u16)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
    }
    None
}

/// Extract a view and convert to u8.
///
/// Silently truncates values that don't fit in u8.
pub fn extract_view_as_u8(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> Option<ArrayD<u8>> {
    unsafe {
        if let Some(view) = extract_view_u8(wrapper, offset, shape, strides) {
            let data = extract_view_data(view);
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i8(wrapper, offset, shape, strides) {
            let data: Vec<u8> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u8)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u16(wrapper, offset, shape, strides) {
            let data: Vec<u8> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u8)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i16(wrapper, offset, shape, strides) {
            let data: Vec<u8> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u8)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u32(wrapper, offset, shape, strides) {
            let data: Vec<u8> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u8)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i32(wrapper, offset, shape, strides) {
            let data: Vec<u8> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u8)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_u64(wrapper, offset, shape, strides) {
            let data: Vec<u8> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u8)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_i64(wrapper, offset, shape, strides) {
            let data: Vec<u8> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u8)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_f64(wrapper, offset, shape, strides) {
            let data: Vec<u8> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u8)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
        }
        if let Some(view) = extract_view_f32(wrapper, offset, shape, strides) {
            let data: Vec<u8> = extract_view_data(view)
                .into_iter()
                .map(|x| x as u8)
                .collect();
            return ArrayD::from_shape_vec(IxDyn(shape), data).ok();
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

/// Convert a wrapper to a different dtype.
pub fn convert_wrapper_dtype(
    wrapper: NDArrayWrapper,
    target_dtype: DType,
) -> Result<NDArrayWrapper, String> {
    if wrapper.dtype == target_dtype {
        return Ok(wrapper);
    }

    let shape = wrapper.shape();

    // Use view extraction with empty strides (contiguous) for full array
    let strides: Vec<usize> = (0..shape.len())
        .map(|i| shape[i + 1..].iter().product::<usize>())
        .collect();

    match target_dtype {
        DType::Float64 => {
            let arr = extract_view_as_f64(&wrapper, 0, &shape, &strides)
                .ok_or("Failed to extract view as f64")?;
            Ok(NDArrayWrapper {
                data: ArrayData::Float64(Arc::new(RwLock::new(arr))),
                dtype: DType::Float64,
            })
        }
        DType::Float32 => {
            let arr = extract_view_as_f32(&wrapper, 0, &shape, &strides)
                .ok_or("Failed to extract view as f32")?;
            Ok(NDArrayWrapper {
                data: ArrayData::Float32(Arc::new(RwLock::new(arr))),
                dtype: DType::Float32,
            })
        }
        DType::Int64 => {
            let arr = unsafe {
                extract_view_i64(&wrapper, 0, &shape, &strides)
                    .ok_or("Failed to extract view as i64")?
                    .to_owned()
            };
            Ok(NDArrayWrapper {
                data: ArrayData::Int64(Arc::new(RwLock::new(arr))),
                dtype: DType::Int64,
            })
        }
        DType::Int32 => {
            let arr = unsafe {
                extract_view_i32(&wrapper, 0, &shape, &strides)
                    .ok_or("Failed to extract view as i32")?
                    .to_owned()
            };
            Ok(NDArrayWrapper {
                data: ArrayData::Int32(Arc::new(RwLock::new(arr))),
                dtype: DType::Int32,
            })
        }
        DType::Int16 => {
            // Extract as i64 first, then convert
            let view = unsafe {
                extract_view_i64(&wrapper, 0, &shape, &strides).ok_or("Failed to extract view")?
            };
            let arr = view.mapv(|x| x as i16).to_owned();
            Ok(NDArrayWrapper {
                data: ArrayData::Int16(Arc::new(RwLock::new(arr))),
                dtype: DType::Int16,
            })
        }
        DType::Int8 => {
            let view = unsafe {
                extract_view_i64(&wrapper, 0, &shape, &strides).ok_or("Failed to extract view")?
            };
            let arr = view.mapv(|x| x as i8).to_owned();
            Ok(NDArrayWrapper {
                data: ArrayData::Int8(Arc::new(RwLock::new(arr))),
                dtype: DType::Int8,
            })
        }
        DType::Uint64 => {
            let arr = extract_view_as_f64(&wrapper, 0, &shape, &strides)
                .ok_or("Failed to extract view")?
                .mapv(|x| x as u64);
            Ok(NDArrayWrapper {
                data: ArrayData::Uint64(Arc::new(RwLock::new(arr))),
                dtype: DType::Uint64,
            })
        }
        DType::Uint32 => {
            let arr = extract_view_as_f64(&wrapper, 0, &shape, &strides)
                .ok_or("Failed to extract view")?
                .mapv(|x| x as u32);
            Ok(NDArrayWrapper {
                data: ArrayData::Uint32(Arc::new(RwLock::new(arr))),
                dtype: DType::Uint32,
            })
        }
        DType::Uint16 => {
            let arr = extract_view_as_f64(&wrapper, 0, &shape, &strides)
                .ok_or("Failed to extract view")?
                .mapv(|x| x as u16);
            Ok(NDArrayWrapper {
                data: ArrayData::Uint16(Arc::new(RwLock::new(arr))),
                dtype: DType::Uint16,
            })
        }
        DType::Uint8 => {
            let arr = extract_view_as_f64(&wrapper, 0, &shape, &strides)
                .ok_or("Failed to extract view")?
                .mapv(|x| x as u8);
            Ok(NDArrayWrapper {
                data: ArrayData::Uint8(Arc::new(RwLock::new(arr))),
                dtype: DType::Uint8,
            })
        }
        DType::Bool => {
            let arr = extract_view_as_f64(&wrapper, 0, &shape, &strides)
                .ok_or("Failed to extract view")?
                .mapv(|x| if x != 0.0 { 1u8 } else { 0u8 });
            Ok(NDArrayWrapper {
                data: ArrayData::Bool(Arc::new(RwLock::new(arr))),
                dtype: DType::Bool,
            })
        }
    }
}
