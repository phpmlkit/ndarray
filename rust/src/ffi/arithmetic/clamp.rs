//! Clamp operation - limit values to [min, max] range.

use crate::core::NDArrayWrapper;
use crate::dtype::DType;
use crate::error::{self, ERR_GENERIC, SUCCESS};
use crate::ffi::NdArrayHandle;

/// Clamp array values to [min, max] range.
///
/// Similar to NumPy's clip function.
/// Panics if min > max.
#[no_mangle]
pub unsafe extern "C" fn ndarray_clamp(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
    min_val: f64,
    max_val: f64,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || out_handle.is_null() || shape.is_null() {
        return ERR_GENERIC;
    }

    if min_val > max_val {
        error::set_last_error("Clamp failed: min > max".to_string());
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = std::slice::from_raw_parts(shape, ndim);
        let strides_slice = std::slice::from_raw_parts(strides, ndim);

        let result = clamp_by_dtype(
            wrapper,
            offset,
            shape_slice,
            strides_slice,
            min_val,
            max_val,
        );

        match result {
            Ok(new_wrapper) => {
                *out_handle = NdArrayHandle::from_wrapper(Box::new(new_wrapper));
                SUCCESS
            }
            Err(e) => {
                error::set_last_error(format!("Clamp failed: {}", e));
                ERR_GENERIC
            }
        }
    })
}

/// Clamp by dtype - handles each type natively
fn clamp_by_dtype(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
    min_val: f64,
    max_val: f64,
) -> Result<NDArrayWrapper, String> {
    use crate::core::view_helpers::{
        extract_view_f32, extract_view_f64, extract_view_i32, extract_view_i64, extract_view_u32,
        extract_view_u64,
    };
    use ndarray::IxDyn;

    unsafe {
        // Try Float64 first for zero-copy
        if let Some(view) = extract_view_f64(wrapper, offset, shape, strides) {
            let clamped = view.mapv(|x| x.clamp(min_val, max_val));
            let data: Vec<f64> = clamped.iter().cloned().collect();
            return NDArrayWrapper::from_slice_f64(&data, shape);
        }

        // Try Float32 natively
        if let Some(view) = extract_view_f32(wrapper, offset, shape, strides) {
            let min_f32 = min_val as f32;
            let max_f32 = max_val as f32;
            let clamped = view.mapv(|x| {
                if x < min_f32 {
                    min_f32
                } else if x > max_f32 {
                    max_f32
                } else {
                    x
                }
            });
            let data: Vec<f32> = clamped.iter().cloned().collect();
            return NDArrayWrapper::from_slice_f32(&data, shape);
        }

        // Try Int64 natively
        if let Some(view) = extract_view_i64(wrapper, offset, shape, strides) {
            let min_i64 = min_val as i64;
            let max_i64 = max_val as i64;
            let clamped = view.mapv(|x| {
                if x < min_i64 {
                    min_i64
                } else if x > max_i64 {
                    max_i64
                } else {
                    x
                }
            });
            let data: Vec<i64> = clamped.iter().cloned().collect();
            return NDArrayWrapper::from_slice_i64(&data, shape);
        }

        // Try Int32 natively
        if let Some(view) = extract_view_i32(wrapper, offset, shape, strides) {
            let min_i32 = min_val as i32;
            let max_i32 = max_val as i32;
            let clamped = view.mapv(|x| {
                if x < min_i32 {
                    min_i32
                } else if x > max_i32 {
                    max_i32
                } else {
                    x
                }
            });
            let data: Vec<i32> = clamped.iter().cloned().collect();
            return NDArrayWrapper::from_slice_i32(&data, shape);
        }

        // Try UInt64 natively
        if let Some(view) = extract_view_u64(wrapper, offset, shape, strides) {
            let min_u64 = min_val.max(0.0) as u64;
            let max_u64 = max_val.max(0.0) as u64;
            let clamped = view.mapv(|x| {
                if x < min_u64 {
                    min_u64
                } else if x > max_u64 {
                    max_u64
                } else {
                    x
                }
            });
            let data: Vec<u64> = clamped.iter().cloned().collect();
            return NDArrayWrapper::from_slice_u64(&data, shape);
        }

        // Try UInt32 natively
        if let Some(view) = extract_view_u32(wrapper, offset, shape, strides) {
            let min_u32 = min_val.max(0.0) as u32;
            let max_u32 = max_val.max(0.0) as u32;
            let clamped = view.mapv(|x| {
                if x < min_u32 {
                    min_u32
                } else if x > max_u32 {
                    max_u32
                } else {
                    x
                }
            });
            let data: Vec<u32> = clamped.iter().cloned().collect();
            return NDArrayWrapper::from_slice_u32(&data, shape);
        }
    }

    // Fall back to converting through f64 for other types
    let flat_data = wrapper.to_f64_vec();
    let arr = ndarray::ArrayD::<f64>::from_shape_vec(IxDyn(shape), flat_data)
        .map_err(|e| format!("Failed to create array: {}", e))?;

    let clamped = arr.mapv(|x| x.clamp(min_val, max_val));
    let result_data: Vec<f64> = clamped.iter().cloned().collect();

    create_wrapper_from_f64(&result_data, shape, wrapper.dtype)
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
