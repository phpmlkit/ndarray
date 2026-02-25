//! Dot product operation with view support - uses views directly without Vec conversion.

use crate::core::view_helpers::{extract_view_as_f64, extract_view_f32, extract_view_f64};
use crate::core::NDArrayWrapper;
use crate::error::{self, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::ffi::{write_output_metadata, NdArrayHandle, ViewMetadata};

/// Compute dot product of two arrays with full view support.
///
/// Works directly on views without converting to Vec first.
#[no_mangle]
pub unsafe extern "C" fn ndarray_dot(
    a: *const NdArrayHandle,
    a_meta: *const ViewMetadata,
    b: *const NdArrayHandle,
    b_meta: *const ViewMetadata,
    out_handle: *mut *mut NdArrayHandle,
    out_dtype_ptr: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if a.is_null()
        || b.is_null()
        || out_handle.is_null()
        || out_dtype_ptr.is_null()
        || out_ndim.is_null()
        || out_shape.is_null()
        || a_meta.is_null()
        || b_meta.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let a_meta_ref = &*a_meta;
        let b_meta_ref = &*b_meta;
        let a_wrapper = NdArrayHandle::as_wrapper(a as *mut _);
        let b_wrapper = NdArrayHandle::as_wrapper(b as *mut _);

        let result = dot_impl(a_wrapper, &a_meta_ref, b_wrapper, &b_meta_ref);

        match result {
            Ok(new_wrapper) => {
                if let Err(e) = write_output_metadata(
                    &new_wrapper,
                    out_dtype_ptr,
                    out_ndim,
                    out_shape,
                    max_ndim,
                ) {
                    error::set_last_error(e);
                    return ERR_GENERIC;
                }
                *out_handle = NdArrayHandle::from_wrapper(Box::new(new_wrapper));
                SUCCESS
            }
            Err(e) => {
                error::set_last_error(format!("Dot product failed: {}", e));
                ERR_SHAPE
            }
        }
    })
}

/// Compute dot product using views directly
fn dot_impl(
    a_wrapper: &NDArrayWrapper,
    a_meta: &ViewMetadata,
    b_wrapper: &NDArrayWrapper,
    b_meta: &ViewMetadata,
) -> Result<NDArrayWrapper, String> {
    match (a_meta.ndim, b_meta.ndim) {
        (1, 1) => dot_1d_1d(a_wrapper, a_meta, b_wrapper, b_meta),
        (2, 2) => dot_2d_2d(a_wrapper, a_meta, b_wrapper, b_meta),
        (2, 1) => dot_2d_1d(a_wrapper, a_meta, b_wrapper, b_meta),
        (1, 2) => dot_1d_2d(a_wrapper, a_meta, b_wrapper, b_meta),
        _ => Err(format!(
            "Dot product not supported for dimensions {}D @ {}D",
            a_meta.ndim, b_meta.ndim
        )),
    }
}

/// 1D @ 1D dot product - iterates directly on views
fn dot_1d_1d(
    a_wrapper: &NDArrayWrapper,
    a_meta: &ViewMetadata,
    b_wrapper: &NDArrayWrapper,
    b_meta: &ViewMetadata,
) -> Result<NDArrayWrapper, String> {
    if unsafe { a_meta.shape_slice()[0] } != unsafe { b_meta.shape_slice()[0] } {
        return Err(format!(
            "Shape mismatch: {} and {}",
            unsafe { a_meta.shape_slice()[0] },
            unsafe { b_meta.shape_slice()[0] }
        ));
    }

    unsafe {
        // Try f64 views - iterate directly using indexed access
        if let Some(a_view) = extract_view_f64(a_wrapper, a_meta) {
            if let Some(b_view) = extract_view_f64(b_wrapper, b_meta) {
                let a_slice = a_view.as_slice().unwrap_or(&[]);
                let b_slice = b_view.as_slice().unwrap_or(&[]);
                if !a_slice.is_empty() && !b_slice.is_empty() {
                    let sum: f64 = a_slice.iter().zip(b_slice.iter()).map(|(a, b)| a * b).sum();
                    return NDArrayWrapper::from_slice_f64(&[sum], &[]);
                }
            }
        }

        // Try f32 views
        if let Some(a_view) = extract_view_f32(a_wrapper, a_meta) {
            if let Some(b_view) = extract_view_f32(b_wrapper, b_meta) {
                let a_slice = a_view.as_slice().unwrap_or(&[]);
                let b_slice = b_view.as_slice().unwrap_or(&[]);
                if !a_slice.is_empty() && !b_slice.is_empty() {
                    let sum: f64 = a_slice
                        .iter()
                        .zip(b_slice.iter())
                        .map(|(a, b)| (*a as f64) * (*b as f64))
                        .sum();
                    return NDArrayWrapper::from_slice_f64(&[sum], &[]);
                }
            }
        }

        // Fall back to generic f64 extraction
        if let Some(a_view) = extract_view_as_f64(a_wrapper, a_meta) {
            if let Some(b_view) = extract_view_as_f64(b_wrapper, b_meta) {
                let sum: f64 = a_view.iter().zip(b_view.iter()).map(|(a, b)| a * b).sum();
                return NDArrayWrapper::from_slice_f64(&[sum], &[]);
            }
        }
    }

    Err("Failed to extract views for dot product".to_string())
}

/// 2D @ 2D matrix multiplication
fn dot_2d_2d(
    a_wrapper: &NDArrayWrapper,
    a_meta: &ViewMetadata,
    b_wrapper: &NDArrayWrapper,
    b_meta: &ViewMetadata,
) -> Result<NDArrayWrapper, String> {
    let a_shape = unsafe { a_meta.shape_slice() };
    let b_shape = unsafe { b_meta.shape_slice() };
    let a_strides = unsafe { a_meta.strides_slice() };
    let b_strides = unsafe { b_meta.strides_slice() };
    if a_shape[1] != b_shape[0] {
        return Err(format!(
            "Shape mismatch: {}x{} and {}x{}",
            a_shape[0], a_shape[1], b_shape[0], b_shape[1]
        ));
    }

    let m = a_shape[0];
    let n = b_shape[1];
    let k = a_shape[1];

    unsafe {
        // Try f64 views - check if contiguous for fast path
        if let Some(a_view) = extract_view_f64(a_wrapper, a_meta) {
            if let Some(b_view) = extract_view_f64(b_wrapper, b_meta) {
                // Check if views are contiguous (standard layout)
                let a_contiguous = a_strides[1] == 1 && a_strides[0] == a_shape[1];
                let b_contiguous = b_strides[1] == 1 && b_strides[0] == b_shape[1];

                if a_contiguous && b_contiguous {
                    // Fast path: contiguous memory
                    let a_slice = a_view.as_slice().unwrap_or(&[]);
                    let b_slice = b_view.as_slice().unwrap_or(&[]);

                    if !a_slice.is_empty() && !b_slice.is_empty() {
                        let mut result = vec![0.0; m * n];
                        for i in 0..m {
                            for j in 0..n {
                                let mut sum = 0.0;
                                for l in 0..k {
                                    sum += a_slice[i * k + l] * b_slice[l * n + j];
                                }
                                result[i * n + j] = sum;
                            }
                        }
                        return NDArrayWrapper::from_slice_f64(&result, &[m, n]);
                    }
                }
            }
        }
    }

    // General path: extract as contiguous and compute
    let a_data = extract_as_contiguous_f64(a_wrapper, a_meta)?;
    let b_data = extract_as_contiguous_f64(b_wrapper, b_meta)?;

    let mut result = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a_data[i * k + l] * b_data[l * n + j];
            }
            result[i * n + j] = sum;
        }
    }

    NDArrayWrapper::from_slice_f64(&result, &[m, n])
}

/// 2D @ 1D matrix-vector multiplication
fn dot_2d_1d(
    a_wrapper: &NDArrayWrapper,
    a_meta: &ViewMetadata,
    b_wrapper: &NDArrayWrapper,
    b_meta: &ViewMetadata,
) -> Result<NDArrayWrapper, String> {
    let a_shape = unsafe { a_meta.shape_slice() };
    let b_shape = unsafe { b_meta.shape_slice() };
    let a_strides = unsafe { a_meta.strides_slice() };
    let b_strides = unsafe { b_meta.strides_slice() };
    if a_shape[1] != b_shape[0] {
        return Err(format!(
            "Shape mismatch: {}x{} and {}",
            a_shape[0], a_shape[1], b_shape[0]
        ));
    }

    let rows = a_shape[0];
    let cols = a_shape[1];

    unsafe {
        // Try f64 views
        if let Some(a_view) = extract_view_f64(a_wrapper, a_meta) {
            if let Some(b_view) = extract_view_f64(b_wrapper, b_meta) {
                let a_contiguous = a_strides[1] == 1 && a_strides[0] == a_shape[1];
                let b_contiguous = b_strides[0] == 1;

                if a_contiguous && b_contiguous {
                    let a_slice = a_view.as_slice().unwrap_or(&[]);
                    let b_slice = b_view.as_slice().unwrap_or(&[]);

                    if !a_slice.is_empty() && !b_slice.is_empty() {
                        let mut result = vec![0.0; rows];
                        for i in 0..rows {
                            let mut sum = 0.0;
                            for j in 0..cols {
                                sum += a_slice[i * cols + j] * b_slice[j];
                            }
                            result[i] = sum;
                        }
                        return NDArrayWrapper::from_slice_f64(&result, &[rows]);
                    }
                }
            }
        }
    }

    // General path
    let a_data = extract_as_contiguous_f64(a_wrapper, a_meta)?;
    let b_data = extract_as_contiguous_f64(b_wrapper, b_meta)?;

    let mut result = vec![0.0; rows];
    for i in 0..rows {
        for j in 0..cols {
            result[i] += a_data[i * cols + j] * b_data[j];
        }
    }

    NDArrayWrapper::from_slice_f64(&result, &[rows])
}

/// 1D @ 2D vector-matrix multiplication
fn dot_1d_2d(
    a_wrapper: &NDArrayWrapper,
    a_meta: &ViewMetadata,
    b_wrapper: &NDArrayWrapper,
    b_meta: &ViewMetadata,
) -> Result<NDArrayWrapper, String> {
    let a_shape = unsafe { a_meta.shape_slice() };
    let b_shape = unsafe { b_meta.shape_slice() };
    let a_strides = unsafe { a_meta.strides_slice() };
    let b_strides = unsafe { b_meta.strides_slice() };
    if a_shape[0] != b_shape[0] {
        return Err(format!(
            "Shape mismatch: {} and {}x{}",
            a_shape[0], b_shape[0], b_shape[1]
        ));
    }

    let cols = b_shape[1];
    let inner = b_shape[0];

    unsafe {
        // Try f64 views
        if let Some(a_view) = extract_view_f64(a_wrapper, a_meta) {
            if let Some(b_view) = extract_view_f64(b_wrapper, b_meta) {
                let a_contiguous = a_strides[0] == 1;
                let b_contiguous = b_strides[1] == 1 && b_strides[0] == b_shape[1];

                if a_contiguous && b_contiguous {
                    let a_slice = a_view.as_slice().unwrap_or(&[]);
                    let b_slice = b_view.as_slice().unwrap_or(&[]);

                    if !a_slice.is_empty() && !b_slice.is_empty() {
                        let mut result = vec![0.0; cols];
                        for j in 0..cols {
                            let mut sum = 0.0;
                            for k in 0..inner {
                                sum += a_slice[k] * b_slice[k * cols + j];
                            }
                            result[j] = sum;
                        }
                        return NDArrayWrapper::from_slice_f64(&result, &[cols]);
                    }
                }
            }
        }
    }

    // General path
    let a_data = extract_as_contiguous_f64(a_wrapper, a_meta)?;
    let b_data = extract_as_contiguous_f64(b_wrapper, b_meta)?;

    let mut result = vec![0.0; cols];
    for j in 0..cols {
        for k in 0..inner {
            result[j] += a_data[k] * b_data[k * cols + j];
        }
    }

    NDArrayWrapper::from_slice_f64(&result, &[cols])
}

/// Extract view data as contiguous f64 Vec
fn extract_as_contiguous_f64(
    wrapper: &NDArrayWrapper,
    meta: &ViewMetadata,
) -> Result<Vec<f64>, String> {
    unsafe {
        // Try native f64 first
        if let Some(view) = extract_view_f64(wrapper, meta) {
            return Ok(view.iter().cloned().collect());
        }

        // Try f32
        if let Some(view) = extract_view_f32(wrapper, meta) {
            return Ok(view.iter().map(|x| *x as f64).collect());
        }

        // Generic fallback
        if let Some(view) = extract_view_as_f64(wrapper, meta) {
            return Ok(view.iter().cloned().collect());
        }
    }

    Err("Failed to extract view".to_string())
}
