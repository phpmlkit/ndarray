//! Element-wise min/max for scalar types used by `binary_op_arithmetic!`.
//!
//! Real values use the same `PartialOrd` rules as before (`>=` / `<=`). Complex values use
//! lexicographic order (compare real parts, then imaginary), matching NumPy `np.minimum` /
//! `np.maximum` on complex arrays.

use num_complex::Complex;
use std::cmp::Ordering;

#[inline(always)]
fn lex_cmp_f32(a: Complex<f32>, b: Complex<f32>) -> Option<Ordering> {
    match a.re.partial_cmp(&b.re)? {
        Ordering::Equal => a.im.partial_cmp(&b.im),
        o => Some(o),
    }
}

#[inline(always)]
fn lex_cmp_f64(a: Complex<f64>, b: Complex<f64>) -> Option<Ordering> {
    match a.re.partial_cmp(&b.re)? {
        Ordering::Equal => a.im.partial_cmp(&b.im),
        o => Some(o),
    }
}

pub trait ElementwiseMaximum: Copy {
    fn elementwise_max(a: Self, b: Self) -> Self;
}

macro_rules! impl_max_real {
    ($t:ty) => {
        impl ElementwiseMaximum for $t {
            #[inline(always)]
            fn elementwise_max(a: Self, b: Self) -> Self {
                if a >= b {
                    a
                } else {
                    b
                }
            }
        }
    };
}

impl_max_real!(f64);
impl_max_real!(f32);
impl_max_real!(i64);
impl_max_real!(i32);
impl_max_real!(i16);
impl_max_real!(i8);
impl_max_real!(u64);
impl_max_real!(u32);
impl_max_real!(u16);
impl_max_real!(u8);

impl ElementwiseMaximum for Complex<f32> {
    #[inline(always)]
    fn elementwise_max(a: Self, b: Self) -> Self {
        match lex_cmp_f32(a, b) {
            Some(Ordering::Less) => b,
            _ => a,
        }
    }
}

impl ElementwiseMaximum for Complex<f64> {
    #[inline(always)]
    fn elementwise_max(a: Self, b: Self) -> Self {
        match lex_cmp_f64(a, b) {
            Some(Ordering::Less) => b,
            _ => a,
        }
    }
}

pub trait ElementwiseMinimum: Copy {
    fn elementwise_min(a: Self, b: Self) -> Self;
}

macro_rules! impl_min_real {
    ($t:ty) => {
        impl ElementwiseMinimum for $t {
            #[inline(always)]
            fn elementwise_min(a: Self, b: Self) -> Self {
                if a <= b {
                    a
                } else {
                    b
                }
            }
        }
    };
}

impl_min_real!(f64);
impl_min_real!(f32);
impl_min_real!(i64);
impl_min_real!(i32);
impl_min_real!(i16);
impl_min_real!(i8);
impl_min_real!(u64);
impl_min_real!(u32);
impl_min_real!(u16);
impl_min_real!(u8);

impl ElementwiseMinimum for Complex<f32> {
    #[inline(always)]
    fn elementwise_min(a: Self, b: Self) -> Self {
        match lex_cmp_f32(a, b) {
            Some(Ordering::Greater) => b,
            _ => a,
        }
    }
}

impl ElementwiseMinimum for Complex<f64> {
    #[inline(always)]
    fn elementwise_min(a: Self, b: Self) -> Self {
        match lex_cmp_f64(a, b) {
            Some(Ordering::Greater) => b,
            _ => a,
        }
    }
}
