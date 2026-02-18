//! where(condition, x, y) operation with broadcasting.

use crate::core::view_helpers::{
    broadcast_shape, extract_view_as_f32, extract_view_as_f64, extract_view_as_i16,
    extract_view_as_i32, extract_view_as_i64, extract_view_as_i8, extract_view_as_u16,
    extract_view_as_u32, extract_view_as_u64, extract_view_as_u8, extract_view_bool,
};
use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{self, ERR_DTYPE, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::ffi::{write_output_metadata, NdArrayHandle};
use ndarray::{ArrayD, ArrayViewD, IxDyn};
use parking_lot::RwLock;
use std::slice;
use std::sync::Arc;

fn where_impl<T: Copy>(
    cond: ArrayViewD<'_, u8>,
    x: ArrayViewD<'_, T>,
    y: ArrayViewD<'_, T>,
) -> Result<(ArrayD<T>, Vec<usize>), String> {
    let out_xy = broadcast_shape(x.shape(), y.shape()).ok_or_else(|| {
        format!(
            "x shape {:?} and y shape {:?} are not broadcast-compatible",
            x.shape(),
            y.shape()
        )
    })?;
    let out_shape = broadcast_shape(&out_xy, cond.shape()).ok_or_else(|| {
        format!(
            "condition shape {:?} is not broadcast-compatible with x/y shape {:?}",
            cond.shape(),
            out_xy
        )
    })?;

    let out_dyn = IxDyn(&out_shape);
    let xb = x
        .broadcast(out_dyn.clone())
        .ok_or_else(|| "Failed to broadcast x in where()".to_string())?;
    let yb = y
        .broadcast(out_dyn.clone())
        .ok_or_else(|| "Failed to broadcast y in where()".to_string())?;
    let cb = cond
        .broadcast(out_dyn.clone())
        .ok_or_else(|| "Failed to broadcast condition in where()".to_string())?;

    // Use ndarray::Zip for vectorized, cache-efficient element-wise selection
    let result = ndarray::Zip::from(&cb)
        .and(&xb)
        .and(&yb)
        .map_collect(|&c, &x, &y| if c != 0 { x } else { y });

    Ok((result, out_shape))
}

/// Select values from x and y depending on condition.
#[no_mangle]
pub unsafe extern "C" fn ndarray_where(
    cond_handle: *const NdArrayHandle,
    cond_offset: usize,
    cond_shape: *const usize,
    cond_strides: *const usize,
    cond_ndim: usize,
    x_handle: *const NdArrayHandle,
    x_offset: usize,
    x_shape: *const usize,
    x_strides: *const usize,
    x_ndim: usize,
    y_handle: *const NdArrayHandle,
    y_offset: usize,
    y_shape: *const usize,
    y_strides: *const usize,
    y_ndim: usize,
    out_handle: *mut *mut NdArrayHandle,
    out_dtype_ptr: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if cond_handle.is_null()
        || cond_shape.is_null()
        || cond_strides.is_null()
        || x_handle.is_null()
        || x_shape.is_null()
        || x_strides.is_null()
        || y_handle.is_null()
        || y_shape.is_null()
        || y_strides.is_null()
        || out_handle.is_null()
        || out_dtype_ptr.is_null()
        || out_ndim.is_null()
        || out_shape.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let cond_wrapper = NdArrayHandle::as_wrapper(cond_handle as *mut _);
        let x_wrapper = NdArrayHandle::as_wrapper(x_handle as *mut _);
        let y_wrapper = NdArrayHandle::as_wrapper(y_handle as *mut _);

        if cond_wrapper.dtype != DType::Bool {
            error::set_last_error("where() condition must have Bool dtype".to_string());
            return ERR_DTYPE;
        }

        let cond_shape_slice = slice::from_raw_parts(cond_shape, cond_ndim);
        let cond_strides_slice = slice::from_raw_parts(cond_strides, cond_ndim);
        let x_shape_slice = slice::from_raw_parts(x_shape, x_ndim);
        let x_strides_slice = slice::from_raw_parts(x_strides, x_ndim);
        let y_shape_slice = slice::from_raw_parts(y_shape, y_ndim);
        let y_strides_slice = slice::from_raw_parts(y_strides, y_ndim);

        let Some(cond_view) = extract_view_bool(
            cond_wrapper,
            cond_offset,
            cond_shape_slice,
            cond_strides_slice,
        ) else {
            error::set_last_error("Failed to extract Bool condition view".to_string());
            return ERR_GENERIC;
        };

        let out_dtype = DType::promote(x_wrapper.dtype, y_wrapper.dtype);

        let (result_wrapper, result_shape_vec) = match out_dtype {
            DType::Float64 => {
                let Some(xv) =
                    extract_view_as_f64(x_wrapper, x_offset, x_shape_slice, x_strides_slice)
                else {
                    error::set_last_error("Failed to extract x as f64".to_string());
                    return ERR_GENERIC;
                };
                let Some(yv) =
                    extract_view_as_f64(y_wrapper, y_offset, y_shape_slice, y_strides_slice)
                else {
                    error::set_last_error("Failed to extract y as f64".to_string());
                    return ERR_GENERIC;
                };
                let (out, shape_vec) = match where_impl(cond_view, xv.view(), yv.view()) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                (
                    NDArrayWrapper {
                        data: ArrayData::Float64(Arc::new(RwLock::new(out))),
                        dtype: DType::Float64,
                    },
                    shape_vec,
                )
            }
            DType::Float32 => {
                let Some(xv) =
                    extract_view_as_f32(x_wrapper, x_offset, x_shape_slice, x_strides_slice)
                else {
                    error::set_last_error("Failed to extract x as f32".to_string());
                    return ERR_GENERIC;
                };
                let Some(yv) =
                    extract_view_as_f32(y_wrapper, y_offset, y_shape_slice, y_strides_slice)
                else {
                    error::set_last_error("Failed to extract y as f32".to_string());
                    return ERR_GENERIC;
                };
                let (out, shape_vec) = match where_impl(cond_view, xv.view(), yv.view()) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                (
                    NDArrayWrapper {
                        data: ArrayData::Float32(Arc::new(RwLock::new(out))),
                        dtype: DType::Float32,
                    },
                    shape_vec,
                )
            }
            DType::Int64 => {
                let Some(xv) =
                    extract_view_as_i64(x_wrapper, x_offset, x_shape_slice, x_strides_slice)
                else {
                    error::set_last_error("Failed to extract x as i64".to_string());
                    return ERR_GENERIC;
                };
                let Some(yv) =
                    extract_view_as_i64(y_wrapper, y_offset, y_shape_slice, y_strides_slice)
                else {
                    error::set_last_error("Failed to extract y as i64".to_string());
                    return ERR_GENERIC;
                };
                let (out, shape_vec) = match where_impl(cond_view, xv.view(), yv.view()) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                (
                    NDArrayWrapper {
                        data: ArrayData::Int64(Arc::new(RwLock::new(out))),
                        dtype: DType::Int64,
                    },
                    shape_vec,
                )
            }
            DType::Int32 => {
                let Some(xv) =
                    extract_view_as_i32(x_wrapper, x_offset, x_shape_slice, x_strides_slice)
                else {
                    error::set_last_error("Failed to extract x as i32".to_string());
                    return ERR_GENERIC;
                };
                let Some(yv) =
                    extract_view_as_i32(y_wrapper, y_offset, y_shape_slice, y_strides_slice)
                else {
                    error::set_last_error("Failed to extract y as i32".to_string());
                    return ERR_GENERIC;
                };
                let (out, shape_vec) = match where_impl(cond_view, xv.view(), yv.view()) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                (
                    NDArrayWrapper {
                        data: ArrayData::Int32(Arc::new(RwLock::new(out))),
                        dtype: DType::Int32,
                    },
                    shape_vec,
                )
            }
            DType::Int16 => {
                let Some(xv) =
                    extract_view_as_i16(x_wrapper, x_offset, x_shape_slice, x_strides_slice)
                else {
                    error::set_last_error("Failed to extract x as i16".to_string());
                    return ERR_GENERIC;
                };
                let Some(yv) =
                    extract_view_as_i16(y_wrapper, y_offset, y_shape_slice, y_strides_slice)
                else {
                    error::set_last_error("Failed to extract y as i16".to_string());
                    return ERR_GENERIC;
                };
                let (out, shape_vec) = match where_impl(cond_view, xv.view(), yv.view()) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                (
                    NDArrayWrapper {
                        data: ArrayData::Int16(Arc::new(RwLock::new(out))),
                        dtype: DType::Int16,
                    },
                    shape_vec,
                )
            }
            DType::Int8 => {
                let Some(xv) =
                    extract_view_as_i8(x_wrapper, x_offset, x_shape_slice, x_strides_slice)
                else {
                    error::set_last_error("Failed to extract x as i8".to_string());
                    return ERR_GENERIC;
                };
                let Some(yv) =
                    extract_view_as_i8(y_wrapper, y_offset, y_shape_slice, y_strides_slice)
                else {
                    error::set_last_error("Failed to extract y as i8".to_string());
                    return ERR_GENERIC;
                };
                let (out, shape_vec) = match where_impl(cond_view, xv.view(), yv.view()) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                (
                    NDArrayWrapper {
                        data: ArrayData::Int8(Arc::new(RwLock::new(out))),
                        dtype: DType::Int8,
                    },
                    shape_vec,
                )
            }
            DType::Uint64 => {
                let Some(xv) =
                    extract_view_as_u64(x_wrapper, x_offset, x_shape_slice, x_strides_slice)
                else {
                    error::set_last_error("Failed to extract x as u64".to_string());
                    return ERR_GENERIC;
                };
                let Some(yv) =
                    extract_view_as_u64(y_wrapper, y_offset, y_shape_slice, y_strides_slice)
                else {
                    error::set_last_error("Failed to extract y as u64".to_string());
                    return ERR_GENERIC;
                };
                let (out, shape_vec) = match where_impl(cond_view, xv.view(), yv.view()) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                (
                    NDArrayWrapper {
                        data: ArrayData::Uint64(Arc::new(RwLock::new(out))),
                        dtype: DType::Uint64,
                    },
                    shape_vec,
                )
            }
            DType::Uint32 => {
                let Some(xv) =
                    extract_view_as_u32(x_wrapper, x_offset, x_shape_slice, x_strides_slice)
                else {
                    error::set_last_error("Failed to extract x as u32".to_string());
                    return ERR_GENERIC;
                };
                let Some(yv) =
                    extract_view_as_u32(y_wrapper, y_offset, y_shape_slice, y_strides_slice)
                else {
                    error::set_last_error("Failed to extract y as u32".to_string());
                    return ERR_GENERIC;
                };
                let (out, shape_vec) = match where_impl(cond_view, xv.view(), yv.view()) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                (
                    NDArrayWrapper {
                        data: ArrayData::Uint32(Arc::new(RwLock::new(out))),
                        dtype: DType::Uint32,
                    },
                    shape_vec,
                )
            }
            DType::Uint16 => {
                let Some(xv) =
                    extract_view_as_u16(x_wrapper, x_offset, x_shape_slice, x_strides_slice)
                else {
                    error::set_last_error("Failed to extract x as u16".to_string());
                    return ERR_GENERIC;
                };
                let Some(yv) =
                    extract_view_as_u16(y_wrapper, y_offset, y_shape_slice, y_strides_slice)
                else {
                    error::set_last_error("Failed to extract y as u16".to_string());
                    return ERR_GENERIC;
                };
                let (out, shape_vec) = match where_impl(cond_view, xv.view(), yv.view()) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                (
                    NDArrayWrapper {
                        data: ArrayData::Uint16(Arc::new(RwLock::new(out))),
                        dtype: DType::Uint16,
                    },
                    shape_vec,
                )
            }
            DType::Uint8 => {
                let Some(xv) =
                    extract_view_as_u8(x_wrapper, x_offset, x_shape_slice, x_strides_slice)
                else {
                    error::set_last_error("Failed to extract x as u8".to_string());
                    return ERR_GENERIC;
                };
                let Some(yv) =
                    extract_view_as_u8(y_wrapper, y_offset, y_shape_slice, y_strides_slice)
                else {
                    error::set_last_error("Failed to extract y as u8".to_string());
                    return ERR_GENERIC;
                };
                let (out, shape_vec) = match where_impl(cond_view, xv.view(), yv.view()) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                (
                    NDArrayWrapper {
                        data: ArrayData::Uint8(Arc::new(RwLock::new(out))),
                        dtype: DType::Uint8,
                    },
                    shape_vec,
                )
            }
            DType::Bool => {
                let Some(xv) =
                    extract_view_as_u8(x_wrapper, x_offset, x_shape_slice, x_strides_slice)
                else {
                    error::set_last_error("Failed to extract x as bool".to_string());
                    return ERR_GENERIC;
                };
                let Some(yv) =
                    extract_view_as_u8(y_wrapper, y_offset, y_shape_slice, y_strides_slice)
                else {
                    error::set_last_error("Failed to extract y as bool".to_string());
                    return ERR_GENERIC;
                };
                let (out, shape_vec) = match where_impl(cond_view, xv.view(), yv.view()) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                (
                    NDArrayWrapper {
                        data: ArrayData::Bool(Arc::new(RwLock::new(out))),
                        dtype: DType::Bool,
                    },
                    shape_vec,
                )
            }
        };

        let _ = result_shape_vec;
        if let Err(e) = write_output_metadata(
            &result_wrapper,
            out_dtype_ptr,
            out_ndim,
            out_shape,
            max_ndim,
        ) {
            error::set_last_error(e);
            return ERR_GENERIC;
        }
        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
