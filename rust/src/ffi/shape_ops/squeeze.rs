//! Squeeze and expand_dims operations.

use std::slice;

use crate::core::NDArrayWrapper;
use crate::dtype::DType;
use crate::error::{self, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::ffi::NdArrayHandle;

/// Remove axes of length 1 from the array.
///
/// If `axes` is null and `num_axes` is 0, removes all length-1 axes.
/// Otherwise, removes only the specified axes.
///
/// # Arguments
/// * `handle` - Array handle
/// * `axes` - Pointer to array of axis indices to squeeze (null for all)
/// * `num_axes` - Number of axes to squeeze (0 for all length-1 axes)
/// * `out_handle` - Output handle pointer
#[no_mangle]
pub unsafe extern "C" fn ndarray_squeeze(
    handle: *const NdArrayHandle,
    axes: *const usize,
    num_axes: usize,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || out_handle.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape = wrapper.shape();
        let ndim = shape.len();

        // Determine which axes to squeeze
        let axes_to_squeeze: Vec<usize> = if num_axes == 0 {
            // Squeeze all length-1 axes
            shape
                .iter()
                .enumerate()
                .filter(|(_, &dim)| dim == 1)
                .map(|(i, _)| i)
                .collect()
        } else {
            // Squeeze specific axes
            if axes.is_null() {
                return ERR_GENERIC;
            }
            let requested: Vec<usize> = slice::from_raw_parts(axes, num_axes).to_vec();

            // Validate axes
            for &axis in &requested {
                if axis >= ndim {
                    error::set_last_error(format!(
                        "Axis {} out of bounds for array with {} dimensions",
                        axis, ndim
                    ));
                    return ERR_SHAPE;
                }
                if shape[axis] != 1 {
                    error::set_last_error(format!(
                        "Cannot squeeze axis {} with size {} (must be 1)",
                        axis, shape[axis]
                    ));
                    return ERR_SHAPE;
                }
            }
            requested
        };

        // Compute new shape
        let new_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !axes_to_squeeze.contains(i))
            .map(|(_, &dim)| dim)
            .collect();

        // If no axes were squeezed, just copy
        // Otherwise, reshape to new shape
        if new_shape == shape {
            // Just return a copy
            let flat_data = wrapper.to_f64_vec();
            let result = create_wrapper_from_f64(&flat_data, &shape, wrapper.dtype);
            match result {
                Ok(new_wrapper) => {
                    *out_handle = NdArrayHandle::from_wrapper(Box::new(new_wrapper));
                    SUCCESS
                }
                Err(e) => {
                    error::set_last_error(format!("Squeeze failed: {}", e));
                    ERR_GENERIC
                }
            }
        } else {
            // Reshape to remove length-1 dimensions
            let flat_data = wrapper.to_f64_vec();
            let result = create_wrapper_from_f64(&flat_data, &new_shape, wrapper.dtype);
            match result {
                Ok(new_wrapper) => {
                    *out_handle = NdArrayHandle::from_wrapper(Box::new(new_wrapper));
                    SUCCESS
                }
                Err(e) => {
                    error::set_last_error(format!("Squeeze failed: {}", e));
                    ERR_GENERIC
                }
            }
        }
    })
}

/// Insert a new axis at the specified position.
///
/// Equivalent to NumPy's expand_dims.
///
/// # Arguments
/// * `handle` - Array handle
/// * `axis` - Position where new axis is inserted
/// * `out_handle` - Output handle pointer
#[no_mangle]
pub unsafe extern "C" fn ndarray_expand_dims(
    handle: *const NdArrayHandle,
    axis: usize,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || out_handle.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let mut shape = wrapper.shape();
        let ndim = shape.len();

        // Validate axis - can be from 0 to ndim (inclusive, for appending)
        if axis > ndim {
            error::set_last_error(format!(
                "Axis {} out of bounds for array with {} dimensions",
                axis, ndim
            ));
            return ERR_SHAPE;
        }

        // Insert new axis with size 1
        shape.insert(axis, 1);

        // Get data and create new array with expanded shape
        let flat_data = wrapper.to_f64_vec();
        let result = create_wrapper_from_f64(&flat_data, &shape, wrapper.dtype);

        match result {
            Ok(new_wrapper) => {
                *out_handle = NdArrayHandle::from_wrapper(Box::new(new_wrapper));
                SUCCESS
            }
            Err(e) => {
                error::set_last_error(format!("Expand dims failed: {}", e));
                ERR_GENERIC
            }
        }
    })
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
