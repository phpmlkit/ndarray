//! Sum reduction.

use std::ffi::c_void;

use crate::ffi::reductions::helpers::{write_reduction_scalar, ReductionScalar};
use crate::helpers::error::{set_last_error, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::helpers::normalize_axis;
use crate::helpers::write_output_metadata;
use crate::helpers::{
    extract_view_c128, extract_view_c64, extract_view_f32, extract_view_f64, extract_view_i16,
    extract_view_i32, extract_view_i64, extract_view_i8, extract_view_u16, extract_view_u32,
    extract_view_u64, extract_view_u8,
};
use crate::types::dtype::DType;
use crate::types::{ArrayData, ArrayMetadata, NDArrayWrapper, NdArrayHandle};
use ndarray::Axis;
use parking_lot::RwLock;
use std::sync::Arc;

/// Compute the sum of all elements in the array.
#[no_mangle]
pub unsafe extern "C" fn ndarray_sum(
    handle: *const NdArrayHandle,
    meta: *const ArrayMetadata,
    out_value: *mut c_void,
    out_dtype: *mut u8,
) -> i32 {
    if handle.is_null() || meta.is_null() || out_value.is_null() || out_dtype.is_null() {
        return ERR_GENERIC;
    }

    let meta = &*meta;

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);

        let scalar = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, meta) else {
                    set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                ReductionScalar::F64(view.sum())
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, meta) else {
                    set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                ReductionScalar::F32(view.sum())
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, meta) else {
                    set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                ReductionScalar::I64(view.sum())
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, meta) else {
                    set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                ReductionScalar::I32(view.sum())
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, meta) else {
                    set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                ReductionScalar::I16(view.sum())
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, meta) else {
                    set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                ReductionScalar::I8(view.sum())
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, meta) else {
                    set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                ReductionScalar::U64(view.sum())
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, meta) else {
                    set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                ReductionScalar::U32(view.sum())
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, meta) else {
                    set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                ReductionScalar::U16(view.sum())
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, meta) else {
                    set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                ReductionScalar::U8(view.sum())
            }
            DType::Complex64 => {
                let Some(view) = extract_view_c64(wrapper, meta) else {
                    set_last_error("Failed to extract Complex64 view".to_string());
                    return ERR_GENERIC;
                };
                ReductionScalar::C64(view.sum())
            }
            DType::Complex128 => {
                let Some(view) = extract_view_c128(wrapper, meta) else {
                    set_last_error("Failed to extract Complex128 view".to_string());
                    return ERR_GENERIC;
                };
                ReductionScalar::C128(view.sum())
            }
            DType::Bool => {
                set_last_error("sum() not supported for Bool type".to_string());
                return ERR_GENERIC;
            }
        };

        write_reduction_scalar(out_value, out_dtype, scalar);
        SUCCESS
    })
}

/// Compute the sum along an axis.
#[no_mangle]
pub unsafe extern "C" fn ndarray_sum_axis(
    handle: *const NdArrayHandle,
    meta: *const ArrayMetadata,
    axis: i32,
    keepdims: bool,
    out_handle: *mut *mut NdArrayHandle,
    out_dtype: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if handle.is_null()
        || out_handle.is_null()
        || meta.is_null()
        || out_dtype.is_null()
        || out_ndim.is_null()
        || out_shape.is_null()
    {
        return ERR_GENERIC;
    }

    let meta = &*meta;

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = meta.shape_slice();

        // Validate axis
        let axis_usize = match normalize_axis(shape_slice, axis, false) {
            Ok(a) => a,
            Err(e) => {
                set_last_error(e);
                return ERR_SHAPE;
            }
        };

        // Match on dtype, extract view, compute sum along axis, and create result wrapper
        let result_wrapper = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, meta) else {
                    set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.sum_axis(Axis(axis_usize));
                let final_arr = if keepdims {
                    result.insert_axis(Axis(axis_usize))
                } else {
                    result
                };
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(final_arr))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, meta) else {
                    set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.sum_axis(Axis(axis_usize));
                let final_arr = if keepdims {
                    result.insert_axis(Axis(axis_usize))
                } else {
                    result
                };
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(final_arr))),
                    dtype: DType::Float32,
                }
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, meta) else {
                    set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.sum_axis(Axis(axis_usize));
                let final_arr = if keepdims {
                    result.insert_axis(Axis(axis_usize))
                } else {
                    result
                };
                NDArrayWrapper {
                    data: ArrayData::Int64(Arc::new(RwLock::new(final_arr))),
                    dtype: DType::Int64,
                }
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, meta) else {
                    set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.sum_axis(Axis(axis_usize));
                let final_arr = if keepdims {
                    result.insert_axis(Axis(axis_usize))
                } else {
                    result
                };
                NDArrayWrapper {
                    data: ArrayData::Int32(Arc::new(RwLock::new(final_arr))),
                    dtype: DType::Int32,
                }
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, meta) else {
                    set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.sum_axis(Axis(axis_usize));
                let final_arr = if keepdims {
                    result.insert_axis(Axis(axis_usize))
                } else {
                    result
                };
                NDArrayWrapper {
                    data: ArrayData::Int16(Arc::new(RwLock::new(final_arr))),
                    dtype: DType::Int16,
                }
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, meta) else {
                    set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.sum_axis(Axis(axis_usize));
                let final_arr = if keepdims {
                    result.insert_axis(Axis(axis_usize))
                } else {
                    result
                };
                NDArrayWrapper {
                    data: ArrayData::Int8(Arc::new(RwLock::new(final_arr))),
                    dtype: DType::Int8,
                }
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, meta) else {
                    set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.sum_axis(Axis(axis_usize));
                let final_arr = if keepdims {
                    result.insert_axis(Axis(axis_usize))
                } else {
                    result
                };
                NDArrayWrapper {
                    data: ArrayData::Uint64(Arc::new(RwLock::new(final_arr))),
                    dtype: DType::Uint64,
                }
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, meta) else {
                    set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.sum_axis(Axis(axis_usize));
                let final_arr = if keepdims {
                    result.insert_axis(Axis(axis_usize))
                } else {
                    result
                };
                NDArrayWrapper {
                    data: ArrayData::Uint32(Arc::new(RwLock::new(final_arr))),
                    dtype: DType::Uint32,
                }
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, meta) else {
                    set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.sum_axis(Axis(axis_usize));
                let final_arr = if keepdims {
                    result.insert_axis(Axis(axis_usize))
                } else {
                    result
                };
                NDArrayWrapper {
                    data: ArrayData::Uint16(Arc::new(RwLock::new(final_arr))),
                    dtype: DType::Uint16,
                }
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, meta) else {
                    set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.sum_axis(Axis(axis_usize));
                let final_arr = if keepdims {
                    result.insert_axis(Axis(axis_usize))
                } else {
                    result
                };
                NDArrayWrapper {
                    data: ArrayData::Uint8(Arc::new(RwLock::new(final_arr))),
                    dtype: DType::Uint8,
                }
            }
            DType::Complex64 => {
                let Some(view) = extract_view_c64(wrapper, meta) else {
                    set_last_error("Failed to extract Complex64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.sum_axis(Axis(axis_usize));
                let final_arr = if keepdims {
                    result.insert_axis(Axis(axis_usize))
                } else {
                    result
                };
                NDArrayWrapper {
                    data: ArrayData::Complex64(Arc::new(RwLock::new(final_arr))),
                    dtype: DType::Complex64,
                }
            }
            DType::Complex128 => {
                let Some(view) = extract_view_c128(wrapper, meta) else {
                    set_last_error("Failed to extract Complex128 view".to_string());
                    return ERR_GENERIC;
                };
                let result = view.sum_axis(Axis(axis_usize));
                let final_arr = if keepdims {
                    result.insert_axis(Axis(axis_usize))
                } else {
                    result
                };
                NDArrayWrapper {
                    data: ArrayData::Complex128(Arc::new(RwLock::new(final_arr))),
                    dtype: DType::Complex128,
                }
            }
            DType::Bool => {
                set_last_error("sum_axis() not supported for Bool type".to_string());
                return ERR_GENERIC;
            }
        };

        if let Err(e) =
            write_output_metadata(&result_wrapper, out_dtype, out_ndim, out_shape, max_ndim)
        {
            set_last_error(e);
            return ERR_GENERIC;
        }
        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
