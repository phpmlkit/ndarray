//! Boolean all reduction (logical AND).

use std::ffi::c_void;
use std::sync::Arc;

use ndarray::Axis;
use parking_lot::RwLock;

use crate::ffi::reductions::helpers::{write_reduction_scalar, ReductionScalar};
use crate::helpers::error::{set_last_error, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::helpers::extract_array_as_bool;
use crate::helpers::normalize_axis;
use crate::helpers::write_output_metadata;
use crate::types::dtype::DType;
use crate::types::{ArrayData, ArrayMetadata, NDArrayWrapper, NdArrayHandle};

/// Compute whether all elements are truthy (scalar).
#[no_mangle]
pub unsafe extern "C" fn ndarray_all(
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

        let Some(arr) = extract_array_as_bool(wrapper, meta) else {
            set_last_error("Failed to extract view".to_string());
            return ERR_GENERIC;
        };

        let result: u8 = arr.iter().all(|&x| x != 0) as u8;
        write_reduction_scalar(out_value, out_dtype, ReductionScalar::Bool(result));
        SUCCESS
    })
}

/// Compute whether all elements are truthy along an axis.
#[no_mangle]
pub unsafe extern "C" fn ndarray_all_axis(
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

        let axis_usize = match normalize_axis(shape_slice, axis, false) {
            Ok(a) => a,
            Err(e) => {
                set_last_error(e);
                return ERR_SHAPE;
            }
        };

        let Some(arr) = extract_array_as_bool(wrapper, meta) else {
            set_last_error("Failed to extract view".to_string());
            return ERR_GENERIC;
        };

        let result = arr.fold_axis(Axis(axis_usize), 1u8, |&acc, &x| acc & x);
        let final_arr: ndarray::ArrayD<u8> = if keepdims {
            result.insert_axis(Axis(axis_usize)).into_dyn()
        } else {
            result.into_dyn()
        };

        let result_wrapper = NDArrayWrapper {
            data: ArrayData::Bool(Arc::new(RwLock::new(final_arr))),
            dtype: DType::Bool,
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
