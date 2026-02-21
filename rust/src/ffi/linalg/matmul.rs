//! Matrix multiplication operation with view support.

use crate::core::view_helpers::{extract_view_as_f64, extract_view_f32, extract_view_f64};
use crate::core::NDArrayWrapper;
use crate::error::{self, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::ffi::{write_output_metadata, NdArrayHandle, ViewMetadata};

/// Matrix multiplication with full view support.
///
/// Accepts ViewMetadata for both arrays to properly handle views.
#[no_mangle]
pub unsafe extern "C" fn ndarray_matmul(
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
        let a_meta = &*a_meta;
        let b_meta = &*b_meta;
        let a_wrapper = NdArrayHandle::as_wrapper(a as *mut _);
        let b_wrapper = NdArrayHandle::as_wrapper(b as *mut _);

        let a_shape_slice = a_meta.shape_slice();
        let b_shape_slice = b_meta.shape_slice();

        // Validate dimensions
        if a_shape_slice.len() < 2 || b_shape_slice.len() < 2 {
            error::set_last_error("Matmul requires at least 2D arrays".to_string());
            return ERR_SHAPE;
        }

        // let result = matmul_impl(a_wrapper, a_meta, b_wrapper, b_meta);
        let a_shape = unsafe { a_meta.shape_slice() };
        let b_shape = unsafe { b_meta.shape_slice() };

        let a_cols = a_shape[a_shape.len() - 1];
        let b_rows = b_shape[b_shape.len() - 2];

        if a_cols != b_rows {
            error::set_last_error(format!(
                "Shape mismatch for matmul: ...x{} and ...x{}",
                a_cols, b_rows
            ));
            return ERR_SHAPE;
        }

        let (a_arr, _, _) = extract_2d_data(a_wrapper, a_meta).unwrap();
        let (b_arr, _, _) = extract_2d_data(b_wrapper, b_meta).unwrap();

        let result = a_arr.dot(&b_arr);
        let result_shape = vec![result.shape()[0], result.shape()[1]];
        let result_data: Vec<f64> = result.iter().cloned().collect();

        let result_wrapper = NDArrayWrapper::from_slice_f64(&result_data, &result_shape);

        match result_wrapper {
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
                error::set_last_error(format!("Matrix multiplication failed: {}", e));
                ERR_SHAPE
            }
        }
    })
}

/// Extract 2D array data from view
fn extract_2d_data(
    wrapper: &NDArrayWrapper,
    meta: &ViewMetadata,
) -> Result<(ndarray::Array2<f64>, usize, usize), String> {
    let shape = unsafe { meta.shape_slice() };
    unsafe {
        // Try native f64 view first
        if let Some(view) = extract_view_f64(wrapper, meta) {
            let data: Vec<f64> = view.iter().cloned().collect();
            let rows = shape[0];
            let cols = shape[1];
            return ndarray::Array2::from_shape_vec((rows, cols), data)
                .map(|arr| (arr, rows, cols))
                .map_err(|e| format!("Failed to create 2D array: {}", e));
        }

        // Try native f32 view
        if let Some(view) = extract_view_f32(wrapper, meta) {
            let data: Vec<f64> = view.iter().map(|x| *x as f64).collect();
            let rows = shape[0];
            let cols = shape[1];
            return ndarray::Array2::from_shape_vec((rows, cols), data)
                .map(|arr| (arr, rows, cols))
                .map_err(|e| format!("Failed to create 2D array: {}", e));
        }
    }

    // Fall back to generic extraction
    let view = extract_view_as_f64(wrapper, meta).ok_or("Failed to extract view as f64")?;
    let data: Vec<f64> = view.iter().cloned().collect();
    let rows = shape[0];
    let cols = shape[1];

    ndarray::Array2::from_shape_vec((rows, cols), data)
        .map(|arr| (arr, rows, cols))
        .map_err(|e| format!("Failed to create 2D array: {}", e))
}
