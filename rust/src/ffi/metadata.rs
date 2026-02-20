//! View metadata structure for FFI operations.

use crate::ffi::NdArrayHandle;

/// Metadata describing a view into an array.
///
/// This struct bundles all the metadata needed to describe a view - offset, shape, strides, and ndim.
#[repr(C)]
pub struct ViewMetadata {
    /// Offset into the underlying data buffer (in elements, not bytes)
    pub offset: usize,
    /// Pointer to shape dimensions array
    pub shape: *const usize,
    /// Pointer to strides array (in elements, not bytes)
    pub strides: *const usize,
    /// Number of dimensions
    pub ndim: usize,
}

impl ViewMetadata {
    /// Create ViewMetadata from individual components.
    pub fn new(offset: usize, shape: *const usize, strides: *const usize, ndim: usize) -> Self {
        Self {
            offset,
            shape,
            strides,
            ndim,
        }
    }

    /// Extract shape as a slice.
    ///
    /// # Safety
    /// Caller must ensure shape pointer is valid for ndim elements.
    pub unsafe fn shape_slice(&self) -> &[usize] {
        std::slice::from_raw_parts(self.shape, self.ndim)
    }

    /// Extract strides as a slice.
    ///
    /// # Safety
    /// Caller must ensure strides pointer is valid for ndim elements.
    pub unsafe fn strides_slice(&self) -> &[usize] {
        std::slice::from_raw_parts(self.strides, self.ndim)
    }
}

/// Trait for types that can provide ViewMetadata.
///
/// This allows uniform access to view metadata from both NdArrayHandle
/// and ViewMetadata itself.
pub trait HasViewMetadata {
    /// Get the view metadata for this object.
    fn view_metadata(&self) -> ViewMetadata;
}

impl ViewMetadata {
    /// Create ViewMetadata from an NdArrayHandle and metadata components.
    ///
    /// # Safety
    /// The shape and strides pointers must be valid for the lifetime of the view operation.
    pub unsafe fn from_handle(
        _handle: *const NdArrayHandle,
        offset: usize,
        shape: *const usize,
        strides: *const usize,
        ndim: usize,
    ) -> Self {
        Self::new(offset, shape, strides, ndim)
    }
}
