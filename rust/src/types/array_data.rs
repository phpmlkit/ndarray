//! ArrayData enum for type-erased array storage.
//!
//! This module defines the core type that holds arrays of different data types
//! with shared ownership and interior mutability.

use ndarray::ArrayD;
use parking_lot::RwLock;
use std::sync::Arc;

/// Enum holding arrays of different data types.
///
/// Each variant wraps an Arc<RwLock<ArrayD<T>>> to enable:
/// - Shared ownership for views (Arc)
/// - Interior mutability with read/write locking (RwLock)
#[derive(Clone)]
pub enum ArrayData {
    Int8(Arc<RwLock<ArrayD<i8>>>),
    Int16(Arc<RwLock<ArrayD<i16>>>),
    Int32(Arc<RwLock<ArrayD<i32>>>),
    Int64(Arc<RwLock<ArrayD<i64>>>),
    Uint8(Arc<RwLock<ArrayD<u8>>>),
    Uint16(Arc<RwLock<ArrayD<u16>>>),
    Uint32(Arc<RwLock<ArrayD<u32>>>),
    Uint64(Arc<RwLock<ArrayD<u64>>>),
    Float32(Arc<RwLock<ArrayD<f32>>>),
    Float64(Arc<RwLock<ArrayD<f64>>>),
    Bool(Arc<RwLock<ArrayD<u8>>>), // Store bool as u8 for FFI compatibility
}
