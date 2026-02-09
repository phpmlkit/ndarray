//! FFI type definitions.
//!
//! This module defines opaque handle types used for FFI communication
//! between PHP and Rust.

use crate::core::NDArrayWrapper;

/// Opaque pointer type for FFI.
///
/// PHP holds this pointer and passes it back to Rust for operations.
/// The actual NDArrayWrapper is stored behind this pointer.
#[repr(C)]
pub struct NdArrayHandle {
    _private: [u8; 0],
}

impl NdArrayHandle {
    /// Convert a boxed NDArrayWrapper into an opaque handle.
    pub fn from_wrapper(wrapper: Box<NDArrayWrapper>) -> *mut Self {
        Box::into_raw(wrapper) as *mut Self
    }

    /// Get a reference to the wrapper from a handle.
    ///
    /// # Safety
    /// The pointer must be valid and not null.
    pub unsafe fn as_wrapper<'a>(ptr: *mut Self) -> &'a NDArrayWrapper {
        &*(ptr as *const NDArrayWrapper)
    }

    /// Get a mutable reference to the wrapper from a handle.
    ///
    /// # Safety
    /// The pointer must be valid and not null.
    #[allow(dead_code)]
    pub unsafe fn as_wrapper_mut<'a>(ptr: *mut Self) -> &'a mut NDArrayWrapper {
        &mut *(ptr as *mut NDArrayWrapper)
    }

    /// Convert a handle back into a boxed wrapper (for destruction).
    ///
    /// # Safety
    /// The pointer must be valid, not null, and not used after this call.
    pub unsafe fn into_wrapper(ptr: *mut Self) -> Box<NDArrayWrapper> {
        Box::from_raw(ptr as *mut NDArrayWrapper)
    }
}
