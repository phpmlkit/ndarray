//! Array metadata structure for FFI operations.

/// Metadata describing a view into an array.
///
/// This struct bundles all the metadata needed to describe a view - offset, shape, strides, and ndim.
#[repr(C)]
pub struct ArrayMetadata {
    /// Offset into the underlying data buffer (in elements, not bytes)
    pub offset: usize,
    /// Pointer to shape dimensions array
    pub shape: *const usize,
    /// Pointer to strides array (in elements, not bytes)
    pub strides: *const usize,
    /// Number of dimensions
    pub ndim: usize,
}

impl ArrayMetadata {
    /// Create ArrayMetadata from individual components.
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
