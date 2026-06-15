//! Diagonal extraction with optional offset.

use ndarray::s;

use crate::helpers::error::{self, ERR_GENERIC, ERR_SHAPE, SUCCESS};
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

fn extract_offset_diag<T: Clone>(arr: &ndarray::ArrayD<T>, offset: isize) -> ndarray::ArrayD<T> {
    let result = if offset >= 0 {
        arr.slice(s![.., offset as usize..]).diag().to_owned()
    } else {
        arr.slice(s![(-offset) as usize.., ..]).diag().to_owned()
    };
    result.into_dyn()
}

/// Extract diagonal elements with an optional offset.
///
/// * offset = 0: main diagonal
/// * offset > 0: upper diagonal
/// * offset < 0: lower diagonal
#[no_mangle]
pub unsafe extern "C" fn ndarray_diagonal(
    handle: *const NdArrayHandle,
    meta: *const ArrayMetadata,
    offset: isize,
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

        if shape_slice.len() != 2 {
            error::set_last_error("Diagonal requires a 2D array".to_string());
            return ERR_SHAPE;
        }

        let result = match wrapper.dtype {
            DType::Float64 => {
                let Some(arr) = extract_array_f64(wrapper, meta) else {
                    error::set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = extract_offset_diag(&arr, offset);
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let Some(arr) = extract_array_f32(wrapper, meta) else {
                    error::set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                let result = extract_offset_diag(&arr, offset);
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(result))),
                    dtype: DType::Float32,
                }
            }
            DType::Int64 => {
                let Some(arr) = extract_array_i64(wrapper, meta) else {
                    error::set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = extract_offset_diag(&arr, offset);
                NDArrayWrapper {
                    data: ArrayData::Int64(Arc::new(RwLock::new(result))),
                    dtype: DType::Int64,
                }
            }
            DType::Int32 => {
                let Some(arr) = extract_array_i32(wrapper, meta) else {
                    error::set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                let result = extract_offset_diag(&arr, offset);
                NDArrayWrapper {
                    data: ArrayData::Int32(Arc::new(RwLock::new(result))),
                    dtype: DType::Int32,
                }
            }
            DType::Int16 => {
                let Some(arr) = extract_array_i16(wrapper, meta) else {
                    error::set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                let result = extract_offset_diag(&arr, offset);
                NDArrayWrapper {
                    data: ArrayData::Int16(Arc::new(RwLock::new(result))),
                    dtype: DType::Int16,
                }
            }
            DType::Int8 => {
                let Some(arr) = extract_array_i8(wrapper, meta) else {
                    error::set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                let result = extract_offset_diag(&arr, offset);
                NDArrayWrapper {
                    data: ArrayData::Int8(Arc::new(RwLock::new(result))),
                    dtype: DType::Int8,
                }
            }
            DType::Uint64 => {
                let Some(arr) = extract_array_u64(wrapper, meta) else {
                    error::set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = extract_offset_diag(&arr, offset);
                NDArrayWrapper {
                    data: ArrayData::Uint64(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint64,
                }
            }
            DType::Uint32 => {
                let Some(arr) = extract_array_u32(wrapper, meta) else {
                    error::set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                let result = extract_offset_diag(&arr, offset);
                NDArrayWrapper {
                    data: ArrayData::Uint32(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint32,
                }
            }
            DType::Uint16 => {
                let Some(arr) = extract_array_u16(wrapper, meta) else {
                    error::set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                let result = extract_offset_diag(&arr, offset);
                NDArrayWrapper {
                    data: ArrayData::Uint16(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint16,
                }
            }
            DType::Uint8 => {
                let Some(arr) = extract_array_u8(wrapper, meta) else {
                    error::set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                let result = extract_offset_diag(&arr, offset);
                NDArrayWrapper {
                    data: ArrayData::Uint8(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint8,
                }
            }
            DType::Complex64 => {
                let Some(arr) = extract_array_c64(wrapper, meta) else {
                    error::set_last_error("Failed to extract c64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = extract_offset_diag(&arr, offset);
                NDArrayWrapper {
                    data: ArrayData::Complex64(Arc::new(RwLock::new(result))),
                    dtype: DType::Complex64,
                }
            }
            DType::Complex128 => {
                let Some(arr) = extract_array_c128(wrapper, meta) else {
                    error::set_last_error("Failed to extract c128 view".to_string());
                    return ERR_GENERIC;
                };
                let result = extract_offset_diag(&arr, offset);
                NDArrayWrapper {
                    data: ArrayData::Complex128(Arc::new(RwLock::new(result))),
                    dtype: DType::Complex128,
                }
            }
            DType::Bool => {
                let Some(arr) = extract_array_bool(wrapper, meta) else {
                    error::set_last_error("Failed to extract bool view".to_string());
                    return ERR_GENERIC;
                };
                let result = extract_offset_diag(&arr, offset);
                NDArrayWrapper {
                    data: ArrayData::Bool(Arc::new(RwLock::new(result))),
                    dtype: DType::Bool,
                }
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
