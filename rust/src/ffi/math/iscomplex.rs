//! Complex number predicates: iscomplex and isreal.
//!
//! iscomplex: Returns true where imaginary part is non-zero.
//! isreal:   Returns true where imaginary part is zero (or for real dtypes).

use crate::helpers::error::{set_last_error, ERR_GENERIC, SUCCESS};
use crate::helpers::write_output_metadata;
use crate::helpers::{extract_view_c128, extract_view_c64};
use crate::types::dtype::DType;
use crate::types::{ArrayData, ArrayMetadata, NDArrayWrapper, NdArrayHandle};
use ndarray::{ArrayD, IxDyn};
use parking_lot::RwLock;
use std::sync::Arc;

/// Returns bool array: true where element has non-zero imaginary part.
#[no_mangle]
pub unsafe extern "C" fn ndarray_iscomplex(
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
            DType::Complex64 => {
                let Some(view) = extract_view_c64(a_wrapper, meta) else {
                    set_last_error("Failed to extract Complex64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.mapv(|x| if x.im != 0.0 { 1u8 } else { 0u8 });
                NDArrayWrapper {
                    data: ArrayData::Bool(Arc::new(RwLock::new(result))),
                    dtype: DType::Bool,
                }
            }
            DType::Complex128 => {
                let Some(view) = extract_view_c128(a_wrapper, meta) else {
                    set_last_error("Failed to extract Complex128 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.mapv(|x| if x.im != 0.0 { 1u8 } else { 0u8 });
                NDArrayWrapper {
                    data: ArrayData::Bool(Arc::new(RwLock::new(result))),
                    dtype: DType::Bool,
                }
            }
            _ => {
                // For all real dtypes, return all false (all zeros)
                let shape = IxDyn(meta.shape_slice());
                let result = ArrayD::zeros(shape);
                NDArrayWrapper {
                    data: ArrayData::Bool(Arc::new(RwLock::new(result))),
                    dtype: DType::Bool,
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

/// Returns bool array: true where element has zero imaginary part (or is real dtype).
#[no_mangle]
pub unsafe extern "C" fn ndarray_isreal(
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
            DType::Complex64 => {
                let Some(view) = extract_view_c64(a_wrapper, meta) else {
                    set_last_error("Failed to extract Complex64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.mapv(|x| if x.im == 0.0 { 1u8 } else { 0u8 });
                NDArrayWrapper {
                    data: ArrayData::Bool(Arc::new(RwLock::new(result))),
                    dtype: DType::Bool,
                }
            }
            DType::Complex128 => {
                let Some(view) = extract_view_c128(a_wrapper, meta) else {
                    set_last_error("Failed to extract Complex128 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.mapv(|x| if x.im == 0.0 { 1u8 } else { 0u8 });
                NDArrayWrapper {
                    data: ArrayData::Bool(Arc::new(RwLock::new(result))),
                    dtype: DType::Bool,
                }
            }
            _ => {
                // For all real dtypes, return all true (all ones)
                let shape = IxDyn(meta.shape_slice());
                let result = ArrayD::from_elem(shape, 1u8);
                NDArrayWrapper {
                    data: ArrayData::Bool(Arc::new(RwLock::new(result))),
                    dtype: DType::Bool,
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
