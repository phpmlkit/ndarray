//! bincount operation for integer arrays.

use crate::core::view_helpers::{
    extract_view_bool, extract_view_i16, extract_view_i32, extract_view_i64, extract_view_i8,
    extract_view_u16, extract_view_u32, extract_view_u64, extract_view_u8,
};
use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{self, ERR_DTYPE, ERR_GENERIC, ERR_INDEX, SUCCESS};
use crate::ffi::NdArrayHandle;
use ndarray::{ArrayD, IxDyn};
use parking_lot::RwLock;
use std::slice;
use std::sync::Arc;

fn bincount_from_iter<I>(iter: I, minlength: usize) -> Result<ArrayD<i64>, String>
where
    I: IntoIterator<Item = i64>,
{
    let mut values: Vec<i64> = Vec::new();
    let mut max_val: usize = 0;

    for v in iter {
        if v < 0 {
            return Err(format!("bincount only supports non-negative values, got {}", v));
        }
        let vu = v as usize;
        if vu > max_val {
            max_val = vu;
        }
        values.push(v);
    }

    let out_len = (max_val + 1).max(minlength);
    let mut counts = vec![0i64; out_len];
    for v in values {
        counts[v as usize] += 1;
    }

    ArrayD::from_shape_vec(IxDyn(&[out_len]), counts)
        .map_err(|e| format!("Failed to create bincount output: {}", e))
}

/// Count occurrences of non-negative integer values in flattened input.
#[no_mangle]
pub unsafe extern "C" fn ndarray_bincount(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
    minlength: usize,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || shape.is_null() || strides.is_null() || out_handle.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = slice::from_raw_parts(shape, ndim);
        let strides_slice = slice::from_raw_parts(strides, ndim);

        let out = match wrapper.dtype {
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, offset, shape_slice, strides_slice) else {
                    error::set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                match bincount_from_iter(view.iter().copied(), minlength) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_INDEX;
                    }
                }
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, offset, shape_slice, strides_slice) else {
                    error::set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                match bincount_from_iter(view.iter().map(|&x| x as i64), minlength) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_INDEX;
                    }
                }
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, offset, shape_slice, strides_slice) else {
                    error::set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                match bincount_from_iter(view.iter().map(|&x| x as i64), minlength) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_INDEX;
                    }
                }
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, offset, shape_slice, strides_slice) else {
                    error::set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                match bincount_from_iter(view.iter().map(|&x| x as i64), minlength) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_INDEX;
                    }
                }
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, offset, shape_slice, strides_slice) else {
                    error::set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                match bincount_from_iter(view.iter().map(|&x| x as i64), minlength) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_INDEX;
                    }
                }
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, offset, shape_slice, strides_slice) else {
                    error::set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                match bincount_from_iter(view.iter().map(|&x| x as i64), minlength) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_INDEX;
                    }
                }
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, offset, shape_slice, strides_slice) else {
                    error::set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                match bincount_from_iter(view.iter().map(|&x| x as i64), minlength) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_INDEX;
                    }
                }
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, offset, shape_slice, strides_slice) else {
                    error::set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                match bincount_from_iter(view.iter().map(|&x| x as i64), minlength) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_INDEX;
                    }
                }
            }
            DType::Bool => {
                let Some(view) = extract_view_bool(wrapper, offset, shape_slice, strides_slice) else {
                    error::set_last_error("Failed to extract bool view".to_string());
                    return ERR_GENERIC;
                };
                match bincount_from_iter(view.iter().map(|&x| x as i64), minlength) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_INDEX;
                    }
                }
            }
            DType::Float32 | DType::Float64 => {
                error::set_last_error("bincount requires integer or bool dtype".to_string());
                return ERR_DTYPE;
            }
        };

        let wrapper_out = NDArrayWrapper {
            data: ArrayData::Int64(Arc::new(RwLock::new(out))),
            dtype: DType::Int64,
        };
        *out_handle = NdArrayHandle::from_wrapper(Box::new(wrapper_out));
        SUCCESS
    })
}
