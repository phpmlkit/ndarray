//! Element-wise minimum operation.
//!
//! Returns the smaller value at each position (like NumPy's np.minimum).

use crate::binary_op_arithmetic;
use crate::helpers::elementwise_minmax::ElementwiseMinimum;
use crate::helpers::error::{set_last_error, ERR_GENERIC, SUCCESS};
use crate::helpers::write_output_metadata;
use crate::helpers::{
    extract_array_f32, extract_array_f64, extract_array_i16, extract_array_i32, extract_array_i64,
    extract_array_i8, extract_array_u16, extract_array_u32, extract_array_u64, extract_array_u8,
};
use crate::types::dtype::DType;
use crate::types::{ArrayData, ArrayMetadata, NDArrayWrapper, NdArrayHandle};
use parking_lot::RwLock;
use std::sync::Arc;

#[inline(always)]
fn minimum<T: ElementwiseMinimum>(a: &T, b: &T) -> T {
    T::elementwise_min(*a, *b)
}

/// Element-wise minimum with broadcasting.
#[no_mangle]
pub unsafe extern "C" fn ndarray_minimum(
    a: *const NdArrayHandle,
    a_meta: *const ArrayMetadata,
    b: *const NdArrayHandle,
    b_meta: *const ArrayMetadata,
    out: *mut *mut NdArrayHandle,
    out_dtype_ptr: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if a.is_null()
        || b.is_null()
        || out.is_null()
        || out_dtype_ptr.is_null()
        || out_shape.is_null()
        || out_ndim.is_null()
        || a_meta.is_null()
        || b_meta.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let a_meta = &*a_meta;
        let b_meta = &*b_meta;

        let a_wrapper = NdArrayHandle::as_wrapper(a as *mut _);
        let b_wrapper = NdArrayHandle::as_wrapper(b as *mut _);

        let result_wrapper = binary_op_arithmetic!(a_wrapper, a_meta, b_wrapper, b_meta, minimum);

        if let Err(e) = write_output_metadata(
            &result_wrapper,
            out_dtype_ptr,
            out_ndim,
            out_shape,
            max_ndim,
        ) {
            set_last_error(e);
            return ERR_GENERIC;
        }
        *out = NdArrayHandle::from_wrapper(Box::new(result_wrapper));

        SUCCESS
    })
}

/// Element-wise minimum with a scalar.
#[no_mangle]
pub unsafe extern "C" fn ndarray_minimum_scalar(
    a: *const NdArrayHandle,
    meta: *const ArrayMetadata,
    scalar: f64,
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
        let a_wrapper = NdArrayHandle::as_wrapper(a as *mut _);
        let meta = &*meta;

        let result_wrapper = match a_wrapper.dtype {
            DType::Float64 => {
                let Some(arr) = extract_array_f64(a_wrapper, meta) else {
                    set_last_error("Failed to extract f64 array".to_string());
                    return ERR_GENERIC;
                };
                let result = arr.mapv(|x| x.min(scalar));
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let Some(arr) = extract_array_f32(a_wrapper, meta) else {
                    set_last_error("Failed to extract f32 array".to_string());
                    return ERR_GENERIC;
                };
                let result = arr.mapv(|x| x.min(scalar as f32));
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(result))),
                    dtype: DType::Float32,
                }
            }
            DType::Int64 => {
                let Some(arr) = extract_array_i64(a_wrapper, meta) else {
                    set_last_error("Failed to extract i64 array".to_string());
                    return ERR_GENERIC;
                };
                let s = scalar as i64;
                let result = arr.mapv(|x| x.min(s));
                NDArrayWrapper {
                    data: ArrayData::Int64(Arc::new(RwLock::new(result))),
                    dtype: DType::Int64,
                }
            }
            DType::Int32 => {
                let Some(arr) = extract_array_i32(a_wrapper, meta) else {
                    set_last_error("Failed to extract i32 array".to_string());
                    return ERR_GENERIC;
                };
                let s = scalar as i32;
                let result = arr.mapv(|x| x.min(s));
                NDArrayWrapper {
                    data: ArrayData::Int32(Arc::new(RwLock::new(result))),
                    dtype: DType::Int32,
                }
            }
            DType::Int16 => {
                let Some(arr) = extract_array_i16(a_wrapper, meta) else {
                    set_last_error("Failed to extract i16 array".to_string());
                    return ERR_GENERIC;
                };
                let s = scalar as i16;
                let result = arr.mapv(|x| x.min(s));
                NDArrayWrapper {
                    data: ArrayData::Int16(Arc::new(RwLock::new(result))),
                    dtype: DType::Int16,
                }
            }
            DType::Int8 => {
                let Some(arr) = extract_array_i8(a_wrapper, meta) else {
                    set_last_error("Failed to extract i8 array".to_string());
                    return ERR_GENERIC;
                };
                let s = scalar as i8;
                let result = arr.mapv(|x| x.min(s));
                NDArrayWrapper {
                    data: ArrayData::Int8(Arc::new(RwLock::new(result))),
                    dtype: DType::Int8,
                }
            }
            DType::Uint64 => {
                let Some(arr) = extract_array_u64(a_wrapper, meta) else {
                    set_last_error("Failed to extract u64 array".to_string());
                    return ERR_GENERIC;
                };
                let s = (scalar.max(0.0)) as u64;
                let result = arr.mapv(|x| x.min(s));
                NDArrayWrapper {
                    data: ArrayData::Uint64(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint64,
                }
            }
            DType::Uint32 => {
                let Some(arr) = extract_array_u32(a_wrapper, meta) else {
                    set_last_error("Failed to extract u32 array".to_string());
                    return ERR_GENERIC;
                };
                let s = (scalar.max(0.0)) as u32;
                let result = arr.mapv(|x| x.min(s));
                NDArrayWrapper {
                    data: ArrayData::Uint32(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint32,
                }
            }
            DType::Uint16 => {
                let Some(arr) = extract_array_u16(a_wrapper, meta) else {
                    set_last_error("Failed to extract u16 array".to_string());
                    return ERR_GENERIC;
                };
                let s = (scalar.max(0.0)) as u16;
                let result = arr.mapv(|x| x.min(s));
                NDArrayWrapper {
                    data: ArrayData::Uint16(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint16,
                }
            }
            DType::Uint8 => {
                let Some(arr) = extract_array_u8(a_wrapper, meta) else {
                    set_last_error("Failed to extract u8 array".to_string());
                    return ERR_GENERIC;
                };
                let s = (scalar.max(0.0)) as u8;
                let result = arr.mapv(|x| x.min(s));
                NDArrayWrapper {
                    data: ArrayData::Uint8(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint8,
                }
            }
            DType::Bool => {
                set_last_error("minimum_scalar() not supported for Bool type".to_string());
                return ERR_GENERIC;
            }
            DType::Complex64 | DType::Complex128 => {
                set_last_error("minimum_scalar() not supported for complex dtypes".to_string());
                return ERR_GENERIC;
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
