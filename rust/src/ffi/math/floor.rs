//! Floor operation.

use crate::core::view_helpers::{extract_view_f32, extract_view_f64};
use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::{write_output_metadata, NdArrayHandle, ViewMetadata};
use parking_lot::RwLock;
use std::sync::Arc;

/// Compute floor element-wise.
#[no_mangle]
pub unsafe extern "C" fn ndarray_floor(
    a: *const NdArrayHandle,
    meta: *const ViewMetadata,
    out: *mut *mut NdArrayHandle,
    out_dtype: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if a.is_null()
        || meta.is_null()
        || out.is_null()
        || out_dtype.is_null()
        || out_ndim.is_null()
        || out_shape.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let meta = &*meta;
        let a_wrapper = NdArrayHandle::as_wrapper(a as *mut _);

        let result_wrapper = match a_wrapper.dtype {
            DType::Float64 => {
                let Some(view) =
                    extract_view_f64(a_wrapper, meta)
                else {
                    crate::error::set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.floor();
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let Some(view) =
                    extract_view_f32(a_wrapper, meta)
                else {
                    crate::error::set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.floor();
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(result))),
                    dtype: DType::Float32,
                }
            }
            _ => {
                crate::error::set_last_error(
                    "floor() requires float type (Float64 or Float32)".to_string(),
                );
                return ERR_GENERIC;
            }
        };

        if let Err(e) =
            write_output_metadata(&result_wrapper, out_dtype, out_ndim, out_shape, max_ndim)
        {
            crate::error::set_last_error(e);
            return ERR_GENERIC;
        }
        *out = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
