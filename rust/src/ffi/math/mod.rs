//! Mathematical operations module.
//!
//! Provides mathematical functions including trigonometry, logarithms,
//! exponentials, powers, and rounding operations.

// Exponential and logarithmic functions
pub mod cbrt;
pub mod exp;
pub mod exp2;
pub mod ln;
pub mod ln_1p;
pub mod log;
pub mod log10;
pub mod log2;
pub mod sqrt;

// Trigonometric functions
pub mod acos;
pub mod asin;
pub mod atan;
pub mod cos;
pub mod cosh;
pub mod sin;
pub mod sinh;
pub mod tan;
pub mod tanh;
pub mod to_degrees;
pub mod to_radians;

// Power and reciprocal functions
pub mod hypot;
pub mod pow2;
pub mod powf;
pub mod powi;
pub mod recip;

// Rounding functions
pub mod ceil;
pub mod floor;
pub mod round;

// Absolute value and sign
pub mod abs;
pub mod neg;
pub mod signum;

// Re-export all FFI functions
pub use abs::*;
pub use acos::*;
pub use asin::*;
pub use atan::*;
pub use cbrt::*;
pub use ceil::*;
pub use cos::*;
pub use cosh::*;
pub use exp::*;
pub use exp2::*;
pub use floor::*;
pub use hypot::*;
pub use ln::*;
pub use ln_1p::*;
pub use log::*;
pub use neg::*;
pub use log10::*;
pub use log2::*;
pub use pow2::*;
pub use powf::*;
pub use powi::*;
pub use recip::*;
pub use round::*;
pub use signum::*;
pub use sin::*;
pub use sinh::*;
pub use sqrt::*;
pub use tan::*;
pub use tanh::*;
pub use to_degrees::*;
pub use to_radians::*;
