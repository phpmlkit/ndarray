//! Linear algebra operations.

pub mod cholesky;
pub mod cond;
pub mod determinant;
pub mod diagonal;
pub mod dot;
pub mod from_diag;
pub mod inverse;
pub mod lstsq;
pub mod matmul;
pub mod norm;
pub mod pinv;
pub mod qr;
pub mod rank;
pub mod solve;
pub mod svd;
pub mod trace;

pub use cholesky::*;
pub use cond::*;
pub use determinant::*;
pub use diagonal::*;
pub use dot::*;
pub use from_diag::*;
pub use inverse::*;
pub use lstsq::*;
pub use matmul::*;
pub use norm::*;
pub use pinv::*;
pub use qr::*;
pub use rank::*;
pub use solve::*;
pub use svd::*;
pub use trace::*;
