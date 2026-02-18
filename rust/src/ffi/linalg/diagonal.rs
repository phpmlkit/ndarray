//! Diagonal extraction with view support.

use ndarray::IxDyn;

use crate::core::NDArrayWrapper;
use crate::dtype::DType;
use crate::error::{self, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::ffi::{write_output_metadata, NdArrayHandle};

/// Extract diagonal elements from a 2D array view.
///
/// Accepts offset, shape, and strides to properly handle views.
#[no_mangle]
pub unsafe extern "C" fn ndarray_diagonal(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
    out_handle: *mut *mut NdArrayHandle,
    out_dtype: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if handle.is_null() || out_handle.is_null() || shape.is_null() || out_dtype.is_null() || out_ndim.is_null() || out_shape.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = std::slice::from_raw_parts(shape, ndim);
        let strides_slice = std::slice::from_raw_parts(strides, ndim);

        if shape_slice.len() < 2 {
            error::set_last_error("Diagonal requires at least 2D array".to_string());
            return ERR_SHAPE;
        }

        let result = diagonal_impl(wrapper, offset, shape_slice, strides_slice);

        match result {
            Ok(new_wrapper) => {
                if let Err(e) = write_output_metadata(&new_wrapper, out_dtype, out_ndim, out_shape, max_ndim) {
                    error::set_last_error(e);
                    return ERR_GENERIC;
                }
                *out_handle = NdArrayHandle::from_wrapper(Box::new(new_wrapper));
                SUCCESS
            }
            Err(e) => {
                error::set_last_error(format!("Diagonal extraction failed: {}", e));
                ERR_SHAPE
            }
        }
    })
}

/// Extract diagonal from a view
fn diagonal_impl(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> Result<NDArrayWrapper, String> {
    use crate::core::view_helpers::extract_view_f64;

    unsafe {
        // Try to extract f64 view first
        if let Some(view) = extract_view_f64(wrapper, offset, shape, strides) {
            // view is ArrayD, use diag() method
            let diag_view = view.diag();
            let diag_data: Vec<f64> = diag_view.iter().cloned().collect();
            let diag_shape = vec![diag_view.len()];

            return NDArrayWrapper::from_slice_f64(&diag_data, &diag_shape);
        }
    }

    // Fall back: extract data as f64 and use ArrayD
    let flat_data = wrapper.to_f64_vec();
    let arr = ndarray::ArrayD::<f64>::from_shape_vec(IxDyn(shape), flat_data)
        .map_err(|e| format!("Failed to create array: {}", e))?;

    let diag_view = arr.diag();
    let diag_data: Vec<f64> = diag_view.iter().cloned().collect();
    let diag_shape = vec![diag_view.len()];

    create_wrapper_from_f64(&diag_data, &diag_shape, wrapper.dtype)
}

/// Helper to create wrapper from f64 data preserving dtype
fn create_wrapper_from_f64(
    data: &[f64],
    shape: &[usize],
    dtype: DType,
) -> Result<NDArrayWrapper, String> {
    match dtype {
        DType::Float64 => NDArrayWrapper::from_slice_f64(data, shape),
        DType::Float32 => NDArrayWrapper::from_slice_f32(
            &data.iter().map(|x| *x as f32).collect::<Vec<_>>(),
            shape,
        ),
        DType::Int64 => NDArrayWrapper::from_slice_i64(
            &data.iter().map(|x| *x as i64).collect::<Vec<_>>(),
            shape,
        ),
        DType::Int32 => NDArrayWrapper::from_slice_i32(
            &data.iter().map(|x| *x as i32).collect::<Vec<_>>(),
            shape,
        ),
        DType::Int16 => NDArrayWrapper::from_slice_i16(
            &data.iter().map(|x| *x as i16).collect::<Vec<_>>(),
            shape,
        ),
        DType::Int8 => {
            NDArrayWrapper::from_slice_i8(&data.iter().map(|x| *x as i8).collect::<Vec<_>>(), shape)
        }
        DType::Uint64 => NDArrayWrapper::from_slice_u64(
            &data.iter().map(|x| *x as u64).collect::<Vec<_>>(),
            shape,
        ),
        DType::Uint32 => NDArrayWrapper::from_slice_u32(
            &data.iter().map(|x| *x as u32).collect::<Vec<_>>(),
            shape,
        ),
        DType::Uint16 => NDArrayWrapper::from_slice_u16(
            &data.iter().map(|x| *x as u16).collect::<Vec<_>>(),
            shape,
        ),
        DType::Uint8 => {
            NDArrayWrapper::from_slice_u8(&data.iter().map(|x| *x as u8).collect::<Vec<_>>(), shape)
        }
        DType::Bool => NDArrayWrapper::from_slice_bool(
            &data
                .iter()
                .map(|x| if *x != 0.0 { 1 } else { 0 })
                .collect::<Vec<_>>(),
            shape,
        ),
    }
}
