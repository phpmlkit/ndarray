//! View extraction helpers for proper strided array views.
//!
//! This module provides methods to extract views from ArrayData,
//! properly handling offset, shape, and strides for non-contiguous arrays.

use ndarray::{ArrayD, ArrayViewD, IxDyn, ShapeBuilder};

use crate::core::{ArrayData, NDArrayWrapper};

// Macro to generate `extract_view` functions for each type
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

// Generate `extract_view` functions for all types
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
define_extract_view!(extract_view_bool, ArrayData::Bool, u8);

// Macro to generate `extract_view_as` functions for type conversion.
macro_rules! define_extract_view_as {
    ($name:ident, $target_type:ty, [$(($source_fn:ident, $conv:expr)),+ $(,)?]) => {
        pub fn $name(
            wrapper: &NDArrayWrapper,
            offset: usize,
            shape: &[usize],
            strides: &[usize],
        ) -> Option<ArrayD<$target_type>> {
            unsafe {
                $(
                    if let Some(view) = $source_fn(wrapper, offset, shape, strides) {
                        return Some(view.mapv($conv).to_owned());
                    }
                )+
            }
            None
        }
    };
}

// Generate optimized `extract_view_as` functions for all types
define_extract_view_as!(
    extract_view_as_f64,
    f64,
    [
        (extract_view_f64, |x: f64| x),
        (extract_view_f32, |x: f32| x as f64),
        (extract_view_i64, |x: i64| x as f64),
        (extract_view_i32, |x: i32| x as f64),
        (extract_view_i16, |x: i16| x as f64),
        (extract_view_i8, |x: i8| x as f64),
        (extract_view_u64, |x: u64| x as f64),
        (extract_view_u32, |x: u32| x as f64),
        (extract_view_u16, |x: u16| x as f64),
        (extract_view_u8, |x: u8| x as f64),
    ]
);

define_extract_view_as!(
    extract_view_as_f32,
    f32,
    [
        (extract_view_f32, |x: f32| x),
        (extract_view_f64, |x: f64| x as f32),
        (extract_view_i32, |x: i32| x as f32),
        (extract_view_i16, |x: i16| x as f32),
        (extract_view_i8, |x: i8| x as f32),
        (extract_view_u32, |x: u32| x as f32),
        (extract_view_u16, |x: u16| x as f32),
        (extract_view_u8, |x: u8| x as f32),
        (extract_view_i64, |x: i64| x as f32),
        (extract_view_u64, |x: u64| x as f32),
    ]
);

define_extract_view_as!(
    extract_view_as_i64,
    i64,
    [
        (extract_view_i64, |x: i64| x),
        (extract_view_i32, |x: i32| x as i64),
        (extract_view_i16, |x: i16| x as i64),
        (extract_view_i8, |x: i8| x as i64),
        (extract_view_u64, |x: u64| x as i64),
        (extract_view_u32, |x: u32| x as i64),
        (extract_view_u16, |x: u16| x as i64),
        (extract_view_u8, |x: u8| x as i64),
        (extract_view_f64, |x: f64| x as i64),
        (extract_view_f32, |x: f32| x as i64),
    ]
);

define_extract_view_as!(
    extract_view_as_i32,
    i32,
    [
        (extract_view_i32, |x: i32| x),
        (extract_view_i64, |x: i64| x as i32),
        (extract_view_i16, |x: i16| x as i32),
        (extract_view_i8, |x: i8| x as i32),
        (extract_view_u32, |x: u32| x as i32),
        (extract_view_u16, |x: u16| x as i32),
        (extract_view_u8, |x: u8| x as i32),
        (extract_view_f64, |x: f64| x as i32),
        (extract_view_f32, |x: f32| x as i32),
        (extract_view_u64, |x: u64| x as i32),
    ]
);

define_extract_view_as!(
    extract_view_as_i16,
    i16,
    [
        (extract_view_i16, |x: i16| x),
        (extract_view_i8, |x: i8| x as i16),
        (extract_view_u16, |x: u16| x as i16),
        (extract_view_u8, |x: u8| x as i16),
        (extract_view_i32, |x: i32| x as i16),
        (extract_view_i64, |x: i64| x as i16),
        (extract_view_u32, |x: u32| x as i16),
        (extract_view_u64, |x: u64| x as i16),
        (extract_view_f64, |x: f64| x as i16),
        (extract_view_f32, |x: f32| x as i16),
    ]
);

define_extract_view_as!(
    extract_view_as_i8,
    i8,
    [
        (extract_view_i8, |x: i8| x),
        (extract_view_u8, |x: u8| x as i8),
        (extract_view_i16, |x: i16| x as i8),
        (extract_view_i32, |x: i32| x as i8),
        (extract_view_i64, |x: i64| x as i8),
        (extract_view_u16, |x: u16| x as i8),
        (extract_view_u32, |x: u32| x as i8),
        (extract_view_u64, |x: u64| x as i8),
        (extract_view_f64, |x: f64| x as i8),
        (extract_view_f32, |x: f32| x as i8),
    ]
);

define_extract_view_as!(
    extract_view_as_u64,
    u64,
    [
        (extract_view_u64, |x: u64| x),
        (extract_view_u32, |x: u32| x as u64),
        (extract_view_u16, |x: u16| x as u64),
        (extract_view_u8, |x: u8| x as u64),
        (extract_view_i64, |x: i64| x as u64),
        (extract_view_i32, |x: i32| x as u64),
        (extract_view_i16, |x: i16| x as u64),
        (extract_view_i8, |x: i8| x as u64),
        (extract_view_f64, |x: f64| x as u64),
        (extract_view_f32, |x: f32| x as u64),
    ]
);

define_extract_view_as!(
    extract_view_as_u32,
    u32,
    [
        (extract_view_u32, |x: u32| x),
        (extract_view_u16, |x: u16| x as u32),
        (extract_view_u8, |x: u8| x as u32),
        (extract_view_i32, |x: i32| x as u32),
        (extract_view_i16, |x: i16| x as u32),
        (extract_view_i8, |x: i8| x as u32),
        (extract_view_f64, |x: f64| x as u32),
        (extract_view_f32, |x: f32| x as u32),
        (extract_view_u64, |x: u64| x as u32),
        (extract_view_i64, |x: i64| x as u32),
    ]
);

define_extract_view_as!(
    extract_view_as_u16,
    u16,
    [
        (extract_view_u16, |x: u16| x),
        (extract_view_u8, |x: u8| x as u16),
        (extract_view_i16, |x: i16| x as u16),
        (extract_view_i8, |x: i8| x as u16),
        (extract_view_u32, |x: u32| x as u16),
        (extract_view_i32, |x: i32| x as u16),
        (extract_view_u64, |x: u64| x as u16),
        (extract_view_i64, |x: i64| x as u16),
        (extract_view_f64, |x: f64| x as u16),
        (extract_view_f32, |x: f32| x as u16),
    ]
);

define_extract_view_as!(
    extract_view_as_u8,
    u8,
    [
        (extract_view_u8, |x: u8| x),
        (extract_view_i8, |x: i8| x as u8),
        (extract_view_u16, |x: u16| x as u8),
        (extract_view_i16, |x: i16| x as u8),
        (extract_view_u32, |x: u32| x as u8),
        (extract_view_i32, |x: i32| x as u8),
        (extract_view_u64, |x: u64| x as u8),
        (extract_view_i64, |x: i64| x as u8),
        (extract_view_f64, |x: f64| x as u8),
        (extract_view_f32, |x: f32| x as u8),
    ]
);
