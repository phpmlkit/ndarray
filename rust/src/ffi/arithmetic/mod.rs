//! Array arithmetic operations module.
//!
//! Provides element-wise arithmetic and mathematical operations with
//! broadcasting support for both array-array and array-scalar operations.

// Binary operations
pub mod add;
pub mod div;
pub mod mul;
pub mod sub;

// Unary math operations
pub mod abs;
pub mod acos;
pub mod asin;
pub mod atan;
pub mod cbrt;
pub mod ceil;
pub mod clamp;
pub mod cos;
pub mod cosh;
pub mod exp;
pub mod exp2;
pub mod floor;
pub mod hypot;
pub mod ln;
pub mod ln_1p;
pub mod log;
pub mod log10;
pub mod log2;
pub mod pow2;
pub mod powf;
pub mod powi;
pub mod recip;
pub mod round;
pub mod signum;
pub mod sin;
pub mod sinh;
pub mod sqrt;
pub mod tan;
pub mod tanh;
pub mod to_degrees;
pub mod to_radians;

// Re-export all FFI functions
pub use abs::*;
pub use acos::*;
pub use add::*;
pub use asin::*;
pub use atan::*;
pub use cbrt::*;
pub use ceil::*;
pub use clamp::*;
pub use cos::*;
pub use cosh::*;
pub use div::*;
pub use exp::*;
pub use exp2::*;
pub use floor::*;
pub use hypot::*;
pub use ln::*;
pub use ln_1p::*;
pub use log::*;
pub use log10::*;
pub use log2::*;
pub use mul::*;
pub use pow2::*;
pub use powf::*;
pub use powi::*;
pub use recip::*;
pub use round::*;
pub use signum::*;
pub use sin::*;
pub use sinh::*;
pub use sqrt::*;
pub use sub::*;
pub use tan::*;
pub use tanh::*;
pub use to_degrees::*;
pub use to_radians::*;
