//! Transpose operations.

use ndarray::IxDyn;

use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{self, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::ffi::NdArrayHandle;

/// Transpose array (swap all axes).
///
/// For a 2D array, this swaps rows and columns.
/// For nD arrays, reverses the order of all axes.
///
/// # Arguments
/// * `handle` - Array handle
/// * `out_handle` - Output handle pointer
#[no_mangle]
pub unsafe extern "C" fn ndarray_transpose(
    handle: *const NdArrayHandle,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || out_handle.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);

        let result = transpose_by_dtype(wrapper);

        match result {
            Ok(new_wrapper) => {
                *out_handle = NdArrayHandle::from_wrapper(Box::new(new_wrapper));
                SUCCESS
            }
            Err(e) => {
                error::set_last_error(format!("Transpose failed: {}", e));
                ERR_SHAPE
            }
        }
    })
}

/// Swap two axes of the array.
///
/// # Arguments
/// * `handle` - Array handle
/// * `axis1` - First axis to swap
/// * `axis2` - Second axis to swap
/// * `out_handle` - Output handle pointer
#[no_mangle]
pub unsafe extern "C" fn ndarray_swap_axes(
    handle: *const NdArrayHandle,
    axis1: usize,
    axis2: usize,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || out_handle.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let ndim = wrapper.ndim();

        // Validate axes
        if axis1 >= ndim || axis2 >= ndim {
            error::set_last_error(format!(
                "Axis out of bounds: axes are {} and {} but array has {} dimensions",
                axis1, axis2, ndim
            ));
            return ERR_SHAPE;
        }

        let result = swap_axes_by_dtype(wrapper, axis1, axis2);

        match result {
            Ok(new_wrapper) => {
                *out_handle = NdArrayHandle::from_wrapper(Box::new(new_wrapper));
                SUCCESS
            }
            Err(e) => {
                error::set_last_error(format!("Swap axes failed: {}", e));
                ERR_SHAPE
            }
        }
    })
}

/// Move axis from one position to another.
///
/// # Arguments
/// * `handle` - Array handle  
/// * `source` - Source axis position
/// * `destination` - Destination axis position
/// * `out_handle` - Output handle pointer
#[no_mangle]
pub unsafe extern "C" fn ndarray_move_axis(
    handle: *const NdArrayHandle,
    source: usize,
    destination: usize,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || out_handle.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let ndim = wrapper.ndim();

        // Validate axes
        if source >= ndim || destination > ndim {
            error::set_last_error(format!(
                "Axis out of bounds: source={}, destination={} but array has {} dimensions",
                source, destination, ndim
            ));
            return ERR_SHAPE;
        }

        let result = move_axis_by_dtype(wrapper, source, destination);

        match result {
            Ok(new_wrapper) => {
                *out_handle = NdArrayHandle::from_wrapper(Box::new(new_wrapper));
                SUCCESS
            }
            Err(e) => {
                error::set_last_error(format!("Move axis failed: {}", e));
                ERR_SHAPE
            }
        }
    })
}

/// Transpose array by dtype
fn transpose_by_dtype(wrapper: &NDArrayWrapper) -> Result<NDArrayWrapper, String> {
    let flat_data = wrapper.to_f64_vec();
    let old_shape = wrapper.shape();

    // Create ArrayD and transpose
    let arr = ndarray::ArrayD::<f64>::from_shape_vec(IxDyn(&old_shape), flat_data)
        .map_err(|e| format!("Failed to create array: {}", e))?;

    let transposed = arr.t().to_owned();
    let new_shape: Vec<usize> = transposed.shape().to_vec();
    let transposed_data: Vec<f64> = transposed.iter().cloned().collect();

    // Create wrapper with original dtype
    create_wrapper_from_f64(&transposed_data, &new_shape, wrapper.dtype)
}

/// Swap axes by dtype
fn swap_axes_by_dtype(
    wrapper: &NDArrayWrapper,
    axis1: usize,
    axis2: usize,
) -> Result<NDArrayWrapper, String> {
    let flat_data = wrapper.to_f64_vec();
    let mut shape = wrapper.shape();

    // Create ArrayD, clone it, then swap axes
    let mut arr = ndarray::ArrayD::<f64>::from_shape_vec(IxDyn(&shape), flat_data)
        .map_err(|e| format!("Failed to create array: {}", e))?;

    arr.swap_axes(axis1, axis2);
    shape.swap(axis1, axis2);
    let swapped_data: Vec<f64> = arr.iter().cloned().collect();

    create_wrapper_from_f64(&swapped_data, &shape, wrapper.dtype)
}

/// Move axis by dtype
/// Moves axis from source to destination by computing a permutation
fn move_axis_by_dtype(
    wrapper: &NDArrayWrapper,
    source: usize,
    destination: usize,
) -> Result<NDArrayWrapper, String> {
    let flat_data = wrapper.to_f64_vec();
    let shape = wrapper.shape();
    let ndim = shape.len();

    // Create permutation: move source axis to destination
    let mut axes: Vec<usize> = (0..ndim).collect();
    axes.remove(source);
    axes.insert(destination, source);

    // Permute axes using ndarray's permuted_axes
    let arr = ndarray::ArrayD::<f64>::from_shape_vec(IxDyn(&shape), flat_data)
        .map_err(|e| format!("Failed to create array: {}", e))?;

    let moved = arr.permuted_axes(axes).to_owned();
    let new_shape: Vec<usize> = moved.shape().to_vec();
    let moved_data: Vec<f64> = moved.iter().cloned().collect();

    create_wrapper_from_f64(&moved_data, &new_shape, wrapper.dtype)
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
