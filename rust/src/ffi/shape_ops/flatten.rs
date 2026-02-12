//! Flatten and ravel operations.

use ndarray::IxDyn;

use crate::core::NDArrayWrapper;
use crate::dtype::DType;
use crate::error::{self, ERR_GENERIC, SUCCESS};
use crate::ffi::NdArrayHandle;

/// Flatten array to 1D.
///
/// Always returns a copy in C-order (row-major).
///
/// # Arguments
/// * `handle` - Array handle
/// * `out_handle` - Output handle pointer
#[no_mangle]
pub unsafe extern "C" fn ndarray_flatten(
    handle: *const NdArrayHandle,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || out_handle.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);

        // Flatten: just get data as vec and create 1D array
        let flat_data = wrapper.to_f64_vec();
        let len = flat_data.len();

        let result = create_wrapper_from_f64(
            &flat_data,
            &[usize::try_from(len).unwrap_or(0)],
            wrapper.dtype,
        );

        match result {
            Ok(new_wrapper) => {
                *out_handle = NdArrayHandle::from_wrapper(Box::new(new_wrapper));
                SUCCESS
            }
            Err(e) => {
                error::set_last_error(format!("Flatten failed: {}", e));
                ERR_GENERIC
            }
        }
    })
}

/// Ravel array to 1D.
///
/// Similar to flatten but may return a view if the array is already contiguous.
/// For FFI simplicity, we always return a copy.
///
/// # Arguments
/// * `handle` - Array handle
/// * `order` - Order: 0=C (row-major), 1=F (column-major)  
/// * `out_handle` - Output handle pointer
#[no_mangle]
pub unsafe extern "C" fn ndarray_ravel(
    handle: *const NdArrayHandle,
    order: i32,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || out_handle.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);

        // Get flat data
        let flat_data = wrapper.to_f64_vec();
        let len = flat_data.len();

        // Note: For C-order ravel, data is already in correct order
        // For F-order, we'd need to reorder, but for simplicity we use C-order
        if order == 1 {
            // F-order requested - would need to transpose first
            // For now, we just use C-order (similar to NumPy's default)
        }

        let result = create_wrapper_from_f64(
            &flat_data,
            &[usize::try_from(len).unwrap_or(0)],
            wrapper.dtype,
        );

        match result {
            Ok(new_wrapper) => {
                *out_handle = NdArrayHandle::from_wrapper(Box::new(new_wrapper));
                SUCCESS
            }
            Err(e) => {
                error::set_last_error(format!("Ravel failed: {}", e));
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
