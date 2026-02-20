//! Trace computation.

use crate::core::view_helpers::{
    extract_view_bool, extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32,
    extract_view_i64, extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64,
    extract_view_u8,
};
use crate::core::NDArrayWrapper;
use crate::dtype::DType;
use crate::error::{self, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::ffi::{write_output_metadata, NdArrayHandle, ViewMetadata};

/// Compute trace (sum of diagonal elements).
#[no_mangle]
pub unsafe extern "C" fn ndarray_trace(
    handle: *const NdArrayHandle,
    meta: *const ViewMetadata,
    out_handle: *mut *mut NdArrayHandle,
    out_dtype: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if handle.is_null()
        || meta.is_null()
        || out_handle.is_null()
        || out_dtype.is_null()
        || out_ndim.is_null()
        || out_shape.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let meta = &*meta;
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = meta.shape_slice();

        if shape_slice.len() < 2 {
            error::set_last_error("Trace requires at least 2D array".to_string());
            return ERR_SHAPE;
        }

        let result = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, meta) else {
                    error::set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                let trace_sum: f64 = view.diag().iter().sum();
                NDArrayWrapper::from_slice_f64(&[trace_sum], &[]).unwrap()
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, meta) else {
                    error::set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                let trace_sum: f32 = view.diag().iter().sum();
                NDArrayWrapper::from_slice_f32(&[trace_sum], &[]).unwrap()
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, meta) else {
                    error::set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                let trace_sum: i64 = view.diag().iter().sum();
                NDArrayWrapper::from_slice_i64(&[trace_sum], &[]).unwrap()
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, meta) else {
                    error::set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                let trace_sum: i32 = view.diag().iter().sum();
                NDArrayWrapper::from_slice_i32(&[trace_sum], &[]).unwrap()
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, meta) else {
                    error::set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                let trace_sum: i16 = view.diag().iter().sum();
                NDArrayWrapper::from_slice_i16(&[trace_sum], &[]).unwrap()
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, meta) else {
                    error::set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                let trace_sum: i8 = view.diag().iter().sum();
                NDArrayWrapper::from_slice_i8(&[trace_sum], &[]).unwrap()
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, meta) else {
                    error::set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                let trace_sum: u64 = view.diag().iter().sum();
                NDArrayWrapper::from_slice_u64(&[trace_sum], &[]).unwrap()
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, meta) else {
                    error::set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                let trace_sum: u32 = view.diag().iter().sum();
                NDArrayWrapper::from_slice_u32(&[trace_sum], &[]).unwrap()
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, meta) else {
                    error::set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                let trace_sum: u16 = view.diag().iter().sum();
                NDArrayWrapper::from_slice_u16(&[trace_sum], &[]).unwrap()
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, meta) else {
                    error::set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                let trace_sum: u8 = view.diag().iter().sum();
                NDArrayWrapper::from_slice_u8(&[trace_sum], &[]).unwrap()
            }
            DType::Bool => {
                let Some(view) = extract_view_bool(wrapper, meta) else {
                    error::set_last_error("Failed to extract bool view".to_string());
                    return ERR_GENERIC;
                };
                let trace_sum: u8 = view.diag().iter().sum();
                NDArrayWrapper::from_slice_bool(&[trace_sum], &[]).unwrap()
            }
        };

        if let Err(e) = write_output_metadata(&result, out_dtype, out_ndim, out_shape, max_ndim) {
            error::set_last_error(e);
            return ERR_GENERIC;
        }
        *out_handle = NdArrayHandle::from_wrapper(Box::new(result));
        SUCCESS
    })
}
