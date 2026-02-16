//! Joining and splitting operations.
//!
//! Provides concatenate, stack, and split using ndarray's stacking and split_at.

mod helpers;

pub mod concatenate;
pub mod split;
pub mod stack;

pub use concatenate::*;
pub use split::*;
pub use stack::*;
