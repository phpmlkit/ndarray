//! Extract imaginary part element-wise.
//!
//! For complex arrays, returns the imaginary component as float.
//! For real arrays, returns zeros with the same dtype.

use crate::helpers::error::{set_last_error, ERR_GENERIC, SUCCESS};
use crate::helpers::write_output_metadata;
use crate::helpers::{extract_view_c128, extract_view_c64};
use crate::types::dtype::DType;
use crate::types::{ArrayData, ArrayMetadata, NDArrayWrapper, NdArrayHandle};
use ndarray::{ArrayD, IxDyn};
use parking_lot::RwLock;
use std::sync::Arc;

/// Extract imaginary part element-wise.
#[no_mangle]
pub unsafe extern "C" fn ndarray_imag(
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
                let shape = IxDyn(meta.shape_slice());
                let result = ArrayD::zeros(shape);
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let shape = IxDyn(meta.shape_slice());
                let result = ArrayD::zeros(shape);
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(result))),
                    dtype: DType::Float32,
                }
            }
            DType::Int64 => {
                let shape = IxDyn(meta.shape_slice());
                let result = ArrayD::zeros(shape);
                NDArrayWrapper {
                    data: ArrayData::Int64(Arc::new(RwLock::new(result))),
                    dtype: DType::Int64,
                }
            }
            DType::Int32 => {
                let shape = IxDyn(meta.shape_slice());
                let result = ArrayD::zeros(shape);
                NDArrayWrapper {
                    data: ArrayData::Int32(Arc::new(RwLock::new(result))),
                    dtype: DType::Int32,
                }
            }
            DType::Int16 => {
                let shape = IxDyn(meta.shape_slice());
                let result = ArrayD::zeros(shape);
                NDArrayWrapper {
                    data: ArrayData::Int16(Arc::new(RwLock::new(result))),
                    dtype: DType::Int16,
                }
            }
            DType::Int8 => {
                let shape = IxDyn(meta.shape_slice());
                let result = ArrayD::zeros(shape);
                NDArrayWrapper {
                    data: ArrayData::Int8(Arc::new(RwLock::new(result))),
                    dtype: DType::Int8,
                }
            }
            DType::Uint64 => {
                let shape = IxDyn(meta.shape_slice());
                let result = ArrayD::zeros(shape);
                NDArrayWrapper {
                    data: ArrayData::Uint64(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint64,
                }
            }
            DType::Uint32 => {
                let shape = IxDyn(meta.shape_slice());
                let result = ArrayD::zeros(shape);
                NDArrayWrapper {
                    data: ArrayData::Uint32(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint32,
                }
            }
            DType::Uint16 => {
                let shape = IxDyn(meta.shape_slice());
                let result = ArrayD::zeros(shape);
                NDArrayWrapper {
                    data: ArrayData::Uint16(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint16,
                }
            }
            DType::Uint8 => {
                let shape = IxDyn(meta.shape_slice());
                let result = ArrayD::zeros(shape);
                NDArrayWrapper {
                    data: ArrayData::Uint8(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint8,
                }
            }
            DType::Bool => {
                let shape = IxDyn(meta.shape_slice());
                let result = ArrayD::zeros(shape);
                NDArrayWrapper {
                    data: ArrayData::Bool(Arc::new(RwLock::new(result))),
                    dtype: DType::Bool,
                }
            }
            DType::Complex64 => {
                let Some(view) = extract_view_c64(a_wrapper, meta) else {
                    set_last_error("Failed to extract Complex64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.mapv(|x| x.im);
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(result))),
                    dtype: DType::Float32,
                }
            }
            DType::Complex128 => {
                let Some(view) = extract_view_c128(a_wrapper, meta) else {
                    set_last_error("Failed to extract Complex128 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.mapv(|x| x.im);
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
