//! Array generation FFI functions.

pub mod arange;
pub mod eye;
pub mod full;
pub mod geomspace;
pub mod linspace;
pub mod logspace;
pub mod normal;
pub mod ones;
pub mod randn;
pub mod random;
pub mod random_int;
pub mod uniform;
pub mod zeros;

pub use arange::ndarray_arange;
pub use eye::ndarray_eye;
pub use full::ndarray_full;
pub use geomspace::ndarray_geomspace;
pub use linspace::ndarray_linspace;
pub use logspace::ndarray_logspace;
pub use normal::ndarray_normal;
pub use ones::ndarray_ones;
pub use randn::ndarray_randn;
pub use random::ndarray_random;
pub use random_int::ndarray_random_int;
pub use uniform::ndarray_uniform;
pub use zeros::ndarray_zeros;
