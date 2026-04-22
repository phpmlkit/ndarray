//! Window functions (signal processing) exposed over FFI.
//!
//! These mirror common NumPy window generators and default to Float64 output.

pub mod bartlett;
pub mod blackman;
pub mod bohman;
pub mod boxcar;
pub mod hamming;
pub mod hanning;
pub mod kaiser;
pub mod lanczos;
pub mod triang;

pub use bartlett::ndarray_bartlett;
pub use blackman::ndarray_blackman;
pub use bohman::ndarray_bohman;
pub use boxcar::ndarray_boxcar;
pub use hamming::ndarray_hamming;
pub use hanning::ndarray_hanning;
pub use kaiser::ndarray_kaiser;
pub use lanczos::ndarray_lanczos;
pub use triang::ndarray_triang;

