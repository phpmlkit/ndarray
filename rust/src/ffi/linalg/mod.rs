//! Linear algebra operations.

pub mod diagonal;
pub mod dot;
pub mod matmul;
pub mod norm;
pub mod trace;

pub use diagonal::*;
pub use dot::*;
pub use matmul::*;
pub use norm::*;
pub use trace::*;
