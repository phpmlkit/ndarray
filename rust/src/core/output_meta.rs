use crate::core::NDArrayWrapper;

/// Write output metadata into caller-provided buffers.
///
/// This avoids heap-allocating temporary shape buffers across FFI boundaries.
/// out_dtype can be null if dtype is not needed (e.g., for operations where all inputs have same dtype).
pub unsafe fn write_output_metadata(
    wrapper: &NDArrayWrapper,
    out_dtype: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> Result<(), String> {
    if out_ndim.is_null() || out_shape.is_null() {
        return Err("metadata output pointers must not be null".to_string());
    }

    let shape = wrapper.shape();
    let ndim = shape.len();
    if ndim > max_ndim {
        return Err(format!(
            "output shape ndim {} exceeds provided max_ndim {}",
            ndim, max_ndim
        ));
    }

    // Only write dtype if pointer is provided
    if !out_dtype.is_null() {
        *out_dtype = wrapper.dtype as u8;
    }
    *out_ndim = ndim;
    for (i, dim) in shape.iter().enumerate() {
        *out_shape.add(i) = *dim;
    }

    Ok(())
}
