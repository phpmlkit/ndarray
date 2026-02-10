//! Array arithmetic operations module.
//!
//! Provides element-wise arithmetic and mathematical operations with
//! broadcasting support for both array-array and array-scalar operations.

pub mod helpers;

// Binary operations
pub mod add;
pub mod sub;
pub mod mul;
pub mod div;

// Unary math operations
pub mod abs;
pub mod sqrt;
pub mod exp;
pub mod log;
pub mod sin;
pub mod cos;
pub mod tan;
pub mod sinh;
pub mod cosh;
pub mod tanh;
pub mod asin;
pub mod acos;
pub mod atan;
pub mod cbrt;
pub mod ceil;
pub mod exp2;
pub mod floor;
pub mod hypot;
pub mod log2;
pub mod log10;
pub mod pow2;
pub mod round;
pub mod signum;
pub mod recip;

// Re-export all FFI functions
pub use add::*;
pub use sub::*;
pub use mul::*;
pub use div::*;
pub use abs::*;
pub use sqrt::*;
pub use exp::*;
pub use log::*;
pub use sin::*;
pub use cos::*;
pub use tan::*;
pub use sinh::*;
pub use cosh::*;
pub use tanh::*;
pub use asin::*;
pub use acos::*;
pub use atan::*;
pub use cbrt::*;
pub use ceil::*;
pub use exp2::*;
pub use floor::*;
pub use hypot::*;
pub use log2::*;
pub use log10::*;
pub use pow2::*;
pub use round::*;
pub use signum::*;
pub use recip::*;
