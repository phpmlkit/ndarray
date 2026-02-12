//! Reshape operations.

use ndarray::{ArrayD, IxDyn};
use std::slice;

use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{self, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::ffi::NdArrayHandle;

/// Reshape array to new shape.
///
/// Uses ndarray's `to_shape` which returns a view if possible, otherwise copies.
/// Supports both RowMajor (C-style) and ColumnMajor (F-style) ordering.
///
/// # Arguments
/// * `handle` - Array handle
/// * `new_shape` - Pointer to new shape array
/// * `new_ndim` - Number of dimensions in new shape
/// * `order` - Order: 0=RowMajor (C), 1=ColumnMajor (F)
/// * `out_handle` - Output handle pointer
#[no_mangle]
pub unsafe extern "C" fn ndarray_reshape(
    handle: *const NdArrayHandle,
    new_shape: *const usize,
    new_ndim: usize,
    order: i32,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || new_shape.is_null() || out_handle.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = slice::from_raw_parts(new_shape, new_ndim);

        // Validate total elements match
        let new_size: usize = shape_slice.iter().product();
        let old_size: usize = wrapper.shape().iter().product();

        if new_size != old_size {
            error::set_last_error(format!(
                "Cannot reshape array of size {} into shape {:?} (size {})",
                old_size, shape_slice, new_size
            ));
            return ERR_SHAPE;
        }

        // Convert to ndarray Order
        let order_enum = match order {
            0 => ndarray::Order::RowMajor,
            1 => ndarray::Order::ColumnMajor,
            _ => {
                error::set_last_error(format!(
                    "Invalid order: {}. Use 0 for RowMajor, 1 for ColumnMajor",
                    order
                ));
                return ERR_GENERIC;
            }
        };

        // Reshape based on dtype
        let result = reshape_by_dtype(wrapper, shape_slice, order_enum);

        match result {
            Ok(new_wrapper) => {
                *out_handle = NdArrayHandle::from_wrapper(Box::new(new_wrapper));
                SUCCESS
            }
            Err(e) => {
                error::set_last_error(format!("Reshape failed: {}", e));
                ERR_SHAPE
            }
        }
    })
}

/// Reshape array by converting to vec, reshaping, and creating new wrapper
fn reshape_by_dtype(
    wrapper: &NDArrayWrapper,
    new_shape: &[usize],
    order: ndarray::Order,
) -> Result<NDArrayWrapper, String> {
    // Get flat data as f64, then reshape
    let flat_data = wrapper.to_f64_vec();

    // Create ArrayD from flat data with old shape
    let old_shape = wrapper.shape();
    let old_arr = ArrayD::<f64>::from_shape_vec(IxDyn(&old_shape), flat_data)
        .map_err(|e| format!("Failed to create array: {}", e))?;

    // Reshape using to_shape
    let new_ixdyn = IxDyn(new_shape);
    let reshaped = old_arr
        .to_shape((new_ixdyn, order))
        .map_err(|e| format!("Reshape failed: {}", e))?;

    // Extract data and create new wrapper preserving dtype
    let reshaped_data: Vec<f64> = reshaped.iter().cloned().collect();

    match wrapper.dtype {
        DType::Float64 => NDArrayWrapper::from_slice_f64(&reshaped_data, new_shape),
        DType::Float32 => NDArrayWrapper::from_slice_f32(
            &reshaped_data.iter().map(|x| *x as f32).collect::<Vec<_>>(),
            new_shape,
        ),
        DType::Int64 => NDArrayWrapper::from_slice_i64(
            &reshaped_data.iter().map(|x| *x as i64).collect::<Vec<_>>(),
            new_shape,
        ),
        DType::Int32 => NDArrayWrapper::from_slice_i32(
            &reshaped_data.iter().map(|x| *x as i32).collect::<Vec<_>>(),
            new_shape,
        ),
        DType::Int16 => NDArrayWrapper::from_slice_i16(
            &reshaped_data.iter().map(|x| *x as i16).collect::<Vec<_>>(),
            new_shape,
        ),
        DType::Int8 => NDArrayWrapper::from_slice_i8(
            &reshaped_data.iter().map(|x| *x as i8).collect::<Vec<_>>(),
            new_shape,
        ),
        DType::Uint64 => NDArrayWrapper::from_slice_u64(
            &reshaped_data.iter().map(|x| *x as u64).collect::<Vec<_>>(),
            new_shape,
        ),
        DType::Uint32 => NDArrayWrapper::from_slice_u32(
            &reshaped_data.iter().map(|x| *x as u32).collect::<Vec<_>>(),
            new_shape,
        ),
        DType::Uint16 => NDArrayWrapper::from_slice_u16(
            &reshaped_data.iter().map(|x| *x as u16).collect::<Vec<_>>(),
            new_shape,
        ),
        DType::Uint8 => NDArrayWrapper::from_slice_u8(
            &reshaped_data.iter().map(|x| *x as u8).collect::<Vec<_>>(),
            new_shape,
        ),
        DType::Bool => NDArrayWrapper::from_slice_bool(
            &reshaped_data
                .iter()
                .map(|x| if *x != 0.0 { 1 } else { 0 })
                .collect::<Vec<_>>(),
            new_shape,
        ),
    }
}
