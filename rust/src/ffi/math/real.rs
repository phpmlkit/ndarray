//! Extract real part element-wise.
//!
//! For complex arrays, returns the real component as float.
//! For real arrays, returns a copy with the same dtype.

use crate::helpers::error::{set_last_error, ERR_GENERIC, SUCCESS};
use crate::helpers::write_output_metadata;
use crate::helpers::{
    extract_array_bool, extract_array_c128, extract_array_c64, extract_array_f32,
    extract_array_f64, extract_array_i16, extract_array_i32, extract_array_i64, extract_array_i8,
    extract_array_u16, extract_array_u32, extract_array_u64, extract_array_u8,
};
use crate::types::dtype::DType;
use crate::types::{ArrayData, ArrayMetadata, NDArrayWrapper, NdArrayHandle};
use parking_lot::RwLock;
use std::sync::Arc;

/// Extract real part element-wise.
#[no_mangle]
pub unsafe extern "C" fn ndarray_real(
    a: *const NdArrayHandle,
    meta: *const ArrayMetadata,
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
                let Some(arr) = extract_array_f64(a_wrapper, meta) else {
                    set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = arr.to_owned();
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let Some(arr) = extract_array_f32(a_wrapper, meta) else {
                    set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                let result = arr.to_owned();
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(result))),
                    dtype: DType::Float32,
                }
            }
            DType::Int64 => {
                let Some(arr) = extract_array_i64(a_wrapper, meta) else {
                    set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = arr.to_owned();
                NDArrayWrapper {
                    data: ArrayData::Int64(Arc::new(RwLock::new(result))),
                    dtype: DType::Int64,
                }
            }
            DType::Int32 => {
                let Some(arr) = extract_array_i32(a_wrapper, meta) else {
                    set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                let result = arr.to_owned();
                NDArrayWrapper {
                    data: ArrayData::Int32(Arc::new(RwLock::new(result))),
                    dtype: DType::Int32,
                }
            }
            DType::Int16 => {
                let Some(arr) = extract_array_i16(a_wrapper, meta) else {
                    set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                let result = arr.to_owned();
                NDArrayWrapper {
                    data: ArrayData::Int16(Arc::new(RwLock::new(result))),
                    dtype: DType::Int16,
                }
            }
            DType::Int8 => {
                let Some(arr) = extract_array_i8(a_wrapper, meta) else {
                    set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                let result = arr.to_owned();
                NDArrayWrapper {
                    data: ArrayData::Int8(Arc::new(RwLock::new(result))),
                    dtype: DType::Int8,
                }
            }
            DType::Uint64 => {
                let Some(arr) = extract_array_u64(a_wrapper, meta) else {
                    set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = arr.to_owned();
                NDArrayWrapper {
                    data: ArrayData::Uint64(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint64,
                }
            }
            DType::Uint32 => {
                let Some(arr) = extract_array_u32(a_wrapper, meta) else {
                    set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                let result = arr.to_owned();
                NDArrayWrapper {
                    data: ArrayData::Uint32(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint32,
                }
            }
            DType::Uint16 => {
                let Some(arr) = extract_array_u16(a_wrapper, meta) else {
                    set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                let result = arr.to_owned();
                NDArrayWrapper {
                    data: ArrayData::Uint16(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint16,
                }
            }
            DType::Uint8 => {
                let Some(arr) = extract_array_u8(a_wrapper, meta) else {
                    set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                let result = arr.to_owned();
                NDArrayWrapper {
                    data: ArrayData::Uint8(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint8,
                }
            }
            DType::Bool => {
                let Some(arr) = extract_array_bool(a_wrapper, meta) else {
                    set_last_error("Failed to extract bool view".to_string());
                    return ERR_GENERIC;
                };
                let result = arr.to_owned();
                NDArrayWrapper {
                    data: ArrayData::Bool(Arc::new(RwLock::new(result))),
                    dtype: DType::Bool,
                }
            }
            DType::Complex64 => {
                let Some(arr) = extract_array_c64(a_wrapper, meta) else {
                    set_last_error("Failed to extract Complex64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = arr.mapv(|x| x.re);
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(result))),
                    dtype: DType::Float32,
                }
            }
            DType::Complex128 => {
                let Some(arr) = extract_array_c128(a_wrapper, meta) else {
                    set_last_error("Failed to extract Complex128 view".to_string());
                    return ERR_GENERIC;
                };
                let result = arr.mapv(|x| x.re);
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
                    dtype: DType::Float64,
                }
            }
        };

        if let Err(e) =
            write_output_metadata(&result_wrapper, out_dtype, out_ndim, out_shape, max_ndim)
        {
            set_last_error(e);
            return ERR_GENERIC;
        }
        *out = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
