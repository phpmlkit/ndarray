//! View extraction helpers for proper strided array views.
//!
//! This module provides methods to extract views from ArrayData,
//! properly handling offset, shape, and strides for non-contiguous arrays.

use crate::define_extract_view;
use crate::define_extract_view_as;
use crate::define_extract_view_mut;
use crate::types::ArrayData;
use ndarray::ShapeBuilder;

// Generate `extract_view` functions for all types (immutable)
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
define_extract_view!(
    extract_view_c64,
    ArrayData::Complex64,
    num_complex::Complex32
);
define_extract_view!(
    extract_view_c128,
    ArrayData::Complex128,
    num_complex::Complex64
);

// Generate `extract_view_mut` functions for all types (mutable)
define_extract_view_mut!(extract_view_mut_f64, ArrayData::Float64, f64);
define_extract_view_mut!(extract_view_mut_f32, ArrayData::Float32, f32);
define_extract_view_mut!(extract_view_mut_i64, ArrayData::Int64, i64);
define_extract_view_mut!(extract_view_mut_i32, ArrayData::Int32, i32);
define_extract_view_mut!(extract_view_mut_i16, ArrayData::Int16, i16);
define_extract_view_mut!(extract_view_mut_i8, ArrayData::Int8, i8);
define_extract_view_mut!(extract_view_mut_u64, ArrayData::Uint64, u64);
define_extract_view_mut!(extract_view_mut_u32, ArrayData::Uint32, u32);
define_extract_view_mut!(extract_view_mut_u16, ArrayData::Uint16, u16);
define_extract_view_mut!(extract_view_mut_u8, ArrayData::Uint8, u8);
define_extract_view_mut!(extract_view_mut_bool, ArrayData::Bool, u8);
define_extract_view_mut!(
    extract_view_mut_c64,
    ArrayData::Complex64,
    num_complex::Complex32
);
define_extract_view_mut!(
    extract_view_mut_c128,
    ArrayData::Complex128,
    num_complex::Complex64
);

// Generate `extract_view_as` functions for all types
define_extract_view_as!(
    extract_view_as_f64,
    f64,
    extract_view_f64,
    [
        (extract_view_f32, |x: f32| x as f64),
        (extract_view_i64, |x: i64| x as f64),
        (extract_view_i32, |x: i32| x as f64),
        (extract_view_i16, |x: i16| x as f64),
        (extract_view_i8, |x: i8| x as f64),
        (extract_view_u64, |x: u64| x as f64),
        (extract_view_u32, |x: u32| x as f64),
        (extract_view_u16, |x: u16| x as f64),
        (extract_view_u8, |x: u8| x as f64),
        (extract_view_bool, |x: u8| (x != 0) as i32 as f64),
        (extract_view_c128, |x: num_complex::Complex64| x.re),
        (extract_view_c64, |x: num_complex::Complex32| x.re as f64),
    ]
);

define_extract_view_as!(
    extract_view_as_f32,
    f32,
    extract_view_f32,
    [
        (extract_view_f64, |x: f64| x as f32),
        (extract_view_i32, |x: i32| x as f32),
        (extract_view_i16, |x: i16| x as f32),
        (extract_view_i8, |x: i8| x as f32),
        (extract_view_u32, |x: u32| x as f32),
        (extract_view_u16, |x: u16| x as f32),
        (extract_view_u8, |x: u8| x as f32),
        (extract_view_i64, |x: i64| x as f32),
        (extract_view_u64, |x: u64| x as f32),
        (extract_view_bool, |x: u8| (x != 0) as i32 as f32),
        (extract_view_c128, |x: num_complex::Complex64| x.re as f32),
        (extract_view_c64, |x: num_complex::Complex32| x.re),
    ]
);

define_extract_view_as!(
    extract_view_as_i64,
    i64,
    extract_view_i64,
    [
        (extract_view_i32, |x: i32| x as i64),
        (extract_view_i16, |x: i16| x as i64),
        (extract_view_i8, |x: i8| x as i64),
        (extract_view_u64, |x: u64| x as i64),
        (extract_view_u32, |x: u32| x as i64),
        (extract_view_u16, |x: u16| x as i64),
        (extract_view_u8, |x: u8| x as i64),
        (extract_view_f64, |x: f64| x as i64),
        (extract_view_f32, |x: f32| x as i64),
        (extract_view_bool, |x: u8| (x != 0) as i64),
        (extract_view_c128, |x: num_complex::Complex64| x.re as i64),
        (extract_view_c64, |x: num_complex::Complex32| x.re as i64),
    ]
);

define_extract_view_as!(
    extract_view_as_i32,
    i32,
    extract_view_i32,
    [
        (extract_view_i64, |x: i64| x as i32),
        (extract_view_i16, |x: i16| x as i32),
        (extract_view_i8, |x: i8| x as i32),
        (extract_view_u32, |x: u32| x as i32),
        (extract_view_u16, |x: u16| x as i32),
        (extract_view_u8, |x: u8| x as i32),
        (extract_view_f64, |x: f64| x as i32),
        (extract_view_f32, |x: f32| x as i32),
        (extract_view_u64, |x: u64| x as i32),
        (extract_view_bool, |x: u8| (x != 0) as i32),
        (extract_view_c128, |x: num_complex::Complex64| x.re as i32),
        (extract_view_c64, |x: num_complex::Complex32| x.re as i32),
    ]
);

define_extract_view_as!(
    extract_view_as_i16,
    i16,
    extract_view_i16,
    [
        (extract_view_i8, |x: i8| x as i16),
        (extract_view_u16, |x: u16| x as i16),
        (extract_view_u8, |x: u8| x as i16),
        (extract_view_i32, |x: i32| x as i16),
        (extract_view_i64, |x: i64| x as i16),
        (extract_view_u32, |x: u32| x as i16),
        (extract_view_u64, |x: u64| x as i16),
        (extract_view_f64, |x: f64| x as i16),
        (extract_view_f32, |x: f32| x as i16),
        (extract_view_bool, |x: u8| (x != 0) as i16),
        (extract_view_c128, |x: num_complex::Complex64| x.re as i16),
        (extract_view_c64, |x: num_complex::Complex32| x.re as i16),
    ]
);

define_extract_view_as!(
    extract_view_as_i8,
    i8,
    extract_view_i8,
    [
        (extract_view_u8, |x: u8| x as i8),
        (extract_view_i16, |x: i16| x as i8),
        (extract_view_i32, |x: i32| x as i8),
        (extract_view_i64, |x: i64| x as i8),
        (extract_view_u16, |x: u16| x as i8),
        (extract_view_u32, |x: u32| x as i8),
        (extract_view_u64, |x: u64| x as i8),
        (extract_view_f64, |x: f64| x as i8),
        (extract_view_f32, |x: f32| x as i8),
        (extract_view_bool, |x: u8| (x != 0) as i8),
        (extract_view_c128, |x: num_complex::Complex64| x.re as i8),
        (extract_view_c64, |x: num_complex::Complex32| x.re as i8),
    ]
);

define_extract_view_as!(
    extract_view_as_u64,
    u64,
    extract_view_u64,
    [
        (extract_view_u32, |x: u32| x as u64),
        (extract_view_u16, |x: u16| x as u64),
        (extract_view_u8, |x: u8| x as u64),
        (extract_view_i64, |x: i64| x as u64),
        (extract_view_i32, |x: i32| x as u64),
        (extract_view_i16, |x: i16| x as u64),
        (extract_view_i8, |x: i8| x as u64),
        (extract_view_f64, |x: f64| x as u64),
        (extract_view_f32, |x: f32| x as u64),
        (extract_view_bool, |x: u8| (x != 0) as u64),
        (extract_view_c128, |x: num_complex::Complex64| x.re as u64),
        (extract_view_c64, |x: num_complex::Complex32| x.re as u64),
    ]
);

define_extract_view_as!(
    extract_view_as_u32,
    u32,
    extract_view_u32,
    [
        (extract_view_u16, |x: u16| x as u32),
        (extract_view_u8, |x: u8| x as u32),
        (extract_view_i32, |x: i32| x as u32),
        (extract_view_i16, |x: i16| x as u32),
        (extract_view_i8, |x: i8| x as u32),
        (extract_view_f64, |x: f64| x as u32),
        (extract_view_f32, |x: f32| x as u32),
        (extract_view_u64, |x: u64| x as u32),
        (extract_view_i64, |x: i64| x as u32),
        (extract_view_bool, |x: u8| (x != 0) as u32),
        (extract_view_c128, |x: num_complex::Complex64| x.re as u32),
        (extract_view_c64, |x: num_complex::Complex32| x.re as u32),
    ]
);

define_extract_view_as!(
    extract_view_as_u16,
    u16,
    extract_view_u16,
    [
        (extract_view_u8, |x: u8| x as u16),
        (extract_view_i16, |x: i16| x as u16),
        (extract_view_i8, |x: i8| x as u16),
        (extract_view_u32, |x: u32| x as u16),
        (extract_view_i32, |x: i32| x as u16),
        (extract_view_u64, |x: u64| x as u16),
        (extract_view_i64, |x: i64| x as u16),
        (extract_view_f64, |x: f64| x as u16),
        (extract_view_f32, |x: f32| x as u16),
        (extract_view_bool, |x: u8| (x != 0) as u16),
        (extract_view_c128, |x: num_complex::Complex64| x.re as u16),
        (extract_view_c64, |x: num_complex::Complex32| x.re as u16),
    ]
);

define_extract_view_as!(
    extract_view_as_u8,
    u8,
    extract_view_u8,
    [
        (extract_view_i8, |x: i8| x as u8),
        (extract_view_u16, |x: u16| x as u8),
        (extract_view_i16, |x: i16| x as u8),
        (extract_view_u32, |x: u32| x as u8),
        (extract_view_i32, |x: i32| x as u8),
        (extract_view_u64, |x: u64| x as u8),
        (extract_view_i64, |x: i64| x as u8),
        (extract_view_f64, |x: f64| x as u8),
        (extract_view_f32, |x: f32| x as u8),
        (extract_view_bool, |x: u8| x),
        (extract_view_c128, |x: num_complex::Complex64| x.re as u8),
        (extract_view_c64, |x: num_complex::Complex32| x.re as u8),
    ]
);

define_extract_view_as!(
    extract_view_as_bool,
    u8,
    extract_view_bool,
    [
        (extract_view_u8, |x: u8| if x != 0 { 1 } else { 0 }),
        (extract_view_i8, |x: i8| if x != 0 { 1 } else { 0 }),
        (extract_view_u16, |x: u16| if x != 0 { 1 } else { 0 }),
        (extract_view_i16, |x: i16| if x != 0 { 1 } else { 0 }),
        (extract_view_u32, |x: u32| if x != 0 { 1 } else { 0 }),
        (extract_view_i32, |x: i32| if x != 0 { 1 } else { 0 }),
        (extract_view_u64, |x: u64| if x != 0 { 1 } else { 0 }),
        (extract_view_i64, |x: i64| if x != 0 { 1 } else { 0 }),
        (extract_view_f32, |x: f32| if x != 0.0 { 1 } else { 0 }),
        (extract_view_f64, |x: f64| if x != 0.0 { 1 } else { 0 }),
        (extract_view_c128, |x: num_complex::Complex64| {
            if x.re != 0.0 { 1 } else { 0 }
        }),
        (extract_view_c64, |x: num_complex::Complex32| {
            if x.re != 0.0 { 1 } else { 0 }
        }),
    ]
);

define_extract_view_as!(
    extract_view_as_c64,
    num_complex::Complex32,
    extract_view_c64,
    [
        (extract_view_c128, |x: num_complex::Complex64| {
            num_complex::Complex32::new(x.re as f32, x.im as f32)
        }),
        (extract_view_f32, |x: f32| num_complex::Complex32::new(
            x, 0.0
        )),
        (extract_view_f64, |x: f64| num_complex::Complex32::new(
            x as f32, 0.0
        )),
        (extract_view_i64, |x: i64| num_complex::Complex32::new(
            x as f32, 0.0
        )),
        (extract_view_i32, |x: i32| num_complex::Complex32::new(
            x as f32, 0.0
        )),
        (extract_view_i16, |x: i16| num_complex::Complex32::new(
            x as f32, 0.0
        )),
        (extract_view_i8, |x: i8| num_complex::Complex32::new(
            x as f32, 0.0
        )),
        (extract_view_u64, |x: u64| num_complex::Complex32::new(
            x as f32, 0.0
        )),
        (extract_view_u32, |x: u32| num_complex::Complex32::new(
            x as f32, 0.0
        )),
        (extract_view_u16, |x: u16| num_complex::Complex32::new(
            x as f32, 0.0
        )),
        (extract_view_u8, |x: u8| num_complex::Complex32::new(
            x as f32, 0.0
        )),
        (extract_view_bool, |x: u8| num_complex::Complex32::new(
            (x != 0) as i32 as f32,
            0.0
        )),
    ]
);

define_extract_view_as!(
    extract_view_as_c128,
    num_complex::Complex64,
    extract_view_c128,
    [
        (extract_view_c64, |x: num_complex::Complex32| {
            num_complex::Complex64::new(x.re as f64, x.im as f64)
        }),
        (extract_view_f64, |x: f64| num_complex::Complex64::new(
            x, 0.0
        )),
        (extract_view_f32, |x: f32| num_complex::Complex64::new(
            x as f64, 0.0
        )),
        (extract_view_i64, |x: i64| num_complex::Complex64::new(
            x as f64, 0.0
        )),
        (extract_view_i32, |x: i32| num_complex::Complex64::new(
            x as f64, 0.0
        )),
        (extract_view_i16, |x: i16| num_complex::Complex64::new(
            x as f64, 0.0
        )),
        (extract_view_i8, |x: i8| num_complex::Complex64::new(
            x as f64, 0.0
        )),
        (extract_view_u64, |x: u64| num_complex::Complex64::new(
            x as f64, 0.0
        )),
        (extract_view_u32, |x: u32| num_complex::Complex64::new(
            x as f64, 0.0
        )),
        (extract_view_u16, |x: u16| num_complex::Complex64::new(
            x as f64, 0.0
        )),
        (extract_view_u8, |x: u8| num_complex::Complex64::new(
            x as f64, 0.0
        )),
        (extract_view_bool, |x: u8| num_complex::Complex64::new(
            (x != 0) as i32 as f64,
            0.0
        )),
    ]
);

/// Compute the broadcast shape for two arrays using NumPy-compatible rules.
///
/// Compares shapes from the right; dimensions are compatible if equal or one is 1.
/// Returns `None` if shapes are incompatible.
pub fn broadcast_shape(shape_a: &[usize], shape_b: &[usize]) -> Option<Vec<usize>> {
    let na = shape_a.len();
    let nb = shape_b.len();
    let ndim = na.max(nb);
    let mut out = vec![0; ndim];
    for i in 0..ndim {
        let ia = na as isize - ndim as isize + i as isize;
        let ib = nb as isize - ndim as isize + i as isize;
        let dim_a = if ia >= 0 { shape_a[ia as usize] } else { 1 };
        let dim_b = if ib >= 0 { shape_b[ib as usize] } else { 1 };
        out[i] = if dim_a == dim_b {
            dim_a
        } else if dim_a == 1 {
            dim_b
        } else if dim_b == 1 {
            dim_a
        } else {
            return None;
        };
    }
    Some(out)
}
