//! Array generation FFI functions.

pub mod arange;
pub mod eye;
pub mod full;
pub mod geomspace;
pub mod linspace;
pub mod logspace;
pub mod ones;
pub mod zeros;

pub use arange::ndarray_arange;
pub use eye::ndarray_eye;
pub use full::ndarray_full;
pub use geomspace::ndarray_geomspace;
pub use linspace::ndarray_linspace;
pub use logspace::ndarray_logspace;
pub use ones::ndarray_ones;
pub use zeros::ndarray_zeros;
