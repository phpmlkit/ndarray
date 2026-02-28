//! Element-wise minimum and maximum operations.
//!
//! Provides element-wise minimum and maximum operations with broadcasting support.
//! These return element-wise minimum/maximum of two arrays (like NumPy's np.minimum/np.maximum).

use crate::core::view_helpers::{
    extract_view_as_f32, extract_view_as_f64, extract_view_as_i16, extract_view_as_i32,
    extract_view_as_i64, extract_view_as_i8, extract_view_as_u16, extract_view_as_u32,
    extract_view_as_u64, extract_view_as_u8,
};
use crate::dtype::DType;
use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::{write_output_metadata, NdArrayHandle, ViewMetadata};
use crate::minmax_op_arm;
use crate::ArrayData;

/// Element-wise minimum operation with broadcasting.
/// Returns an array of the same shape containing the smaller value at each position.
#[no_mangle]
pub unsafe extern "C" fn ndarray_minimum(
    a: *const NdArrayHandle,
    a_meta: *const ViewMetadata,
    b: *const NdArrayHandle,
    b_meta: *const ViewMetadata,
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

        let out_dtype = DType::promote(a_wrapper.dtype, b_wrapper.dtype);

        let (result_wrapper, result_shape) = match out_dtype {
            DType::Float64 => minmax_op_arm!(
                a_wrapper,
                a_meta,
                b_wrapper,
                b_meta,
                DType::Float64,
                extract_view_as_f64,
                ArrayData::Float64,
                min
            ),
            DType::Float32 => minmax_op_arm!(
                a_wrapper,
                a_meta,
                b_wrapper,
                b_meta,
                DType::Float32,
                extract_view_as_f32,
                ArrayData::Float32,
                min
            ),
            DType::Int64 => minmax_op_arm!(
                a_wrapper,
                a_meta,
                b_wrapper,
                b_meta,
                DType::Int64,
                extract_view_as_i64,
                ArrayData::Int64,
                min
            ),
            DType::Int32 => minmax_op_arm!(
                a_wrapper,
                a_meta,
                b_wrapper,
                b_meta,
                DType::Int32,
                extract_view_as_i32,
                ArrayData::Int32,
                min
            ),
            DType::Int16 => minmax_op_arm!(
                a_wrapper,
                a_meta,
                b_wrapper,
                b_meta,
                DType::Int16,
                extract_view_as_i16,
                ArrayData::Int16,
                min
            ),
            DType::Int8 => minmax_op_arm!(
                a_wrapper,
                a_meta,
                b_wrapper,
                b_meta,
                DType::Int8,
                extract_view_as_i8,
                ArrayData::Int8,
                min
            ),
            DType::Uint64 => minmax_op_arm!(
                a_wrapper,
                a_meta,
                b_wrapper,
                b_meta,
                DType::Uint64,
                extract_view_as_u64,
                ArrayData::Uint64,
                min
            ),
            DType::Uint32 => minmax_op_arm!(
                a_wrapper,
                a_meta,
                b_wrapper,
                b_meta,
                DType::Uint32,
                extract_view_as_u32,
                ArrayData::Uint32,
                min
            ),
            DType::Uint16 => minmax_op_arm!(
                a_wrapper,
                a_meta,
                b_wrapper,
                b_meta,
                DType::Uint16,
                extract_view_as_u16,
                ArrayData::Uint16,
                min
            ),
            DType::Uint8 => minmax_op_arm!(
                a_wrapper,
                a_meta,
                b_wrapper,
                b_meta,
                DType::Uint8,
                extract_view_as_u8,
                ArrayData::Uint8,
                min
            ),
            DType::Bool => {
                crate::error::set_last_error("Minimum not supported for Bool type".to_string());
                return ERR_GENERIC;
            }
        };

        let _ = result_shape;
        if let Err(e) = write_output_metadata(
            &result_wrapper,
            out_dtype_ptr,
            out_ndim,
            out_shape,
            max_ndim,
        ) {
            crate::error::set_last_error(e);
            return ERR_GENERIC;
        }
        *out = NdArrayHandle::from_wrapper(Box::new(result_wrapper));

        SUCCESS
    })
}

/// Element-wise maximum operation with broadcasting.
/// Returns an array of the same shape containing the larger value at each position.
#[no_mangle]
pub unsafe extern "C" fn ndarray_maximum(
    a: *const NdArrayHandle,
    a_meta: *const ViewMetadata,
    b: *const NdArrayHandle,
    b_meta: *const ViewMetadata,
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

        let out_dtype = DType::promote(a_wrapper.dtype, b_wrapper.dtype);

        let (result_wrapper, result_shape) = match out_dtype {
            DType::Float64 => minmax_op_arm!(
                a_wrapper,
                a_meta,
                b_wrapper,
                b_meta,
                DType::Float64,
                extract_view_as_f64,
                ArrayData::Float64,
                max
            ),
            DType::Float32 => minmax_op_arm!(
                a_wrapper,
                a_meta,
                b_wrapper,
                b_meta,
                DType::Float32,
                extract_view_as_f32,
                ArrayData::Float32,
                max
            ),
            DType::Int64 => minmax_op_arm!(
                a_wrapper,
                a_meta,
                b_wrapper,
                b_meta,
                DType::Int64,
                extract_view_as_i64,
                ArrayData::Int64,
                max
            ),
            DType::Int32 => minmax_op_arm!(
                a_wrapper,
                a_meta,
                b_wrapper,
                b_meta,
                DType::Int32,
                extract_view_as_i32,
                ArrayData::Int32,
                max
            ),
            DType::Int16 => minmax_op_arm!(
                a_wrapper,
                a_meta,
                b_wrapper,
                b_meta,
                DType::Int16,
                extract_view_as_i16,
                ArrayData::Int16,
                max
            ),
            DType::Int8 => minmax_op_arm!(
                a_wrapper,
                a_meta,
                b_wrapper,
                b_meta,
                DType::Int8,
                extract_view_as_i8,
                ArrayData::Int8,
                max
            ),
            DType::Uint64 => minmax_op_arm!(
                a_wrapper,
                a_meta,
                b_wrapper,
                b_meta,
                DType::Uint64,
                extract_view_as_u64,
                ArrayData::Uint64,
                max
            ),
            DType::Uint32 => minmax_op_arm!(
                a_wrapper,
                a_meta,
                b_wrapper,
                b_meta,
                DType::Uint32,
                extract_view_as_u32,
                ArrayData::Uint32,
                max
            ),
            DType::Uint16 => minmax_op_arm!(
                a_wrapper,
                a_meta,
                b_wrapper,
                b_meta,
                DType::Uint16,
                extract_view_as_u16,
                ArrayData::Uint16,
                max
            ),
            DType::Uint8 => minmax_op_arm!(
                a_wrapper,
                a_meta,
                b_wrapper,
                b_meta,
                DType::Uint8,
                extract_view_as_u8,
                ArrayData::Uint8,
                max
            ),
            DType::Bool => {
                crate::error::set_last_error("Maximum not supported for Bool type".to_string());
                return ERR_GENERIC;
            }
        };

        let _ = result_shape;
        if let Err(e) = write_output_metadata(
            &result_wrapper,
            out_dtype_ptr,
            out_ndim,
            out_shape,
            max_ndim,
        ) {
            crate::error::set_last_error(e);
            return ERR_GENERIC;
        }
        *out = NdArrayHandle::from_wrapper(Box::new(result_wrapper));

        SUCCESS
    })
}

/// Macro to generate a min/max operation match arm for FFI functions.
///
/// Extracts both arrays as the target type, broadcasts to compatible shape,
/// applies the min/max operation element-wise, and returns the wrapper and shape.
#[macro_export]
macro_rules! minmax_op_arm {
    (
        $a_wrapper:expr, $a_meta:expr,
        $b_wrapper:expr, $b_meta:expr,
        $dtype:path, $extract_fn:ident, $variant:path, $op:ident
    ) => {{
        let Some(a_view) = $extract_fn($a_wrapper, $a_meta) else {
            crate::error::set_last_error(format!("Failed to extract a as {}", stringify!($dtype)));
            return crate::error::ERR_GENERIC;
        };
        let Some(b_view) = $extract_fn($b_wrapper, $b_meta) else {
            crate::error::set_last_error(format!("Failed to extract b as {}", stringify!($dtype)));
            return crate::error::ERR_GENERIC;
        };
        let broadcast_shape =
            match crate::core::view_helpers::broadcast_shape(a_view.shape(), b_view.shape()) {
                Some(s) => s,
                None => {
                    crate::error::set_last_error("incompatible shapes for min/max".to_string());
                    return crate::error::ERR_SHAPE;
                }
            };
        let a_bc = match a_view.broadcast(broadcast_shape.as_slice()) {
            Some(v) => v,
            None => {
                crate::error::set_last_error("incompatible shapes for min/max".to_string());
                return crate::error::ERR_SHAPE;
            }
        };
        let b_bc = match b_view.broadcast(broadcast_shape.as_slice()) {
            Some(v) => v,
            None => {
                crate::error::set_last_error("incompatible shapes for min/max".to_string());
                return crate::error::ERR_SHAPE;
            }
        };
        let result = ndarray::Zip::from(&a_bc).and(&b_bc).map_collect(|a, b| {
            if stringify!($op) == "min" {
                if *a < *b {
                    *a
                } else {
                    *b
                }
            } else {
                if *a > *b {
                    *a
                } else {
                    *b
                }
            }
        });
        let shape = result.shape().to_vec();
        (
            crate::core::NDArrayWrapper {
                data: $variant(::std::sync::Arc::new(::parking_lot::RwLock::new(result))),
                dtype: $dtype,
            },
            shape,
        )
    }};
}
