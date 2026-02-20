//! Pad operations.

use crate::core::view_helpers::{
    extract_view_bool, extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32,
    extract_view_i64, extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64,
    extract_view_u8,
};
use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{self, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::ffi::shape_ops::helpers::PadMode;
use crate::ffi::write_output_metadata;
use crate::ffi::NdArrayHandle;
use crate::ffi::ViewMetadata;
use ndarray::{ArrayD, ArrayViewD, IxDyn};
use parking_lot::RwLock;
use std::slice;
use std::sync::Arc;

#[inline]
fn map_index(idx: isize, dim: usize, mode: PadMode) -> usize {
    if dim <= 1 {
        return 0;
    }

    match mode {
        PadMode::Reflect => {
            let period = (2 * dim - 2) as isize;
            let mut x = idx % period;
            if x < 0 {
                x += period;
            }
            if x < dim as isize {
                x as usize
            } else {
                (period - x) as usize
            }
        }
        PadMode::Symmetric => {
            let period = (2 * dim) as isize;
            let mut x = idx % period;
            if x < 0 {
                x += period;
            }
            if x < dim as isize {
                x as usize
            } else {
                (period - 1 - x) as usize
            }
        }
        PadMode::Constant => 0,
    }
}

#[inline]
fn unravel_index(mut linear: usize, shape: &[usize], out: &mut [usize]) {
    for axis in (0..shape.len()).rev() {
        let dim = shape[axis];
        out[axis] = linear % dim;
        linear /= dim;
    }
}

fn parse_constants<T: Copy>(raw: &[f64], ndim: usize, cast: fn(f64) -> T) -> Vec<(T, T)> {
    if raw.is_empty() {
        return vec![(cast(0.0), cast(0.0)); ndim];
    }
    if raw.len() == 1 {
        let v = cast(raw[0]);
        return vec![(v, v); ndim];
    }
    if raw.len() == 2 {
        let before = cast(raw[0]);
        let after = cast(raw[1]);
        return vec![(before, after); ndim];
    }
    if raw.len() == ndim * 2 {
        let mut out = Vec::with_capacity(ndim);
        for axis in 0..ndim {
            out.push((cast(raw[axis * 2]), cast(raw[axis * 2 + 1])));
        }
        return out;
    }

    let v = cast(raw[0]);
    vec![(v, v); ndim]
}

fn pad_view<T: Copy>(
    view: ArrayViewD<'_, T>,
    pad_pairs: &[(usize, usize)],
    mode: PadMode,
    constants: &[(T, T)],
) -> Result<ArrayD<T>, String> {
    let in_shape = view.shape();
    let ndim = in_shape.len();

    if ndim == 0 {
        return Err("pad() requires at least 1 dimension".to_string());
    }

    let mut out_shape = Vec::with_capacity(ndim);
    for axis in 0..ndim {
        out_shape.push(in_shape[axis] + pad_pairs[axis].0 + pad_pairs[axis].1);
    }

    let out_len: usize = out_shape.iter().product();
    let mut out_data = Vec::with_capacity(out_len);
    let mut out_idx = vec![0usize; ndim];
    let mut in_idx = vec![0usize; ndim];

    for linear in 0..out_len {
        unravel_index(linear, &out_shape, &mut out_idx);
        let mut outside_axis: Option<(usize, bool)> = None;

        for axis in 0..ndim {
            let before = pad_pairs[axis].0 as isize;
            let src = out_idx[axis] as isize - before;
            let dim = in_shape[axis];

            if src < 0 || src >= dim as isize {
                match mode {
                    PadMode::Constant => {
                        if outside_axis.is_none() {
                            outside_axis = Some((axis, src < 0));
                        }
                        in_idx[axis] = 0;
                    }
                    PadMode::Symmetric | PadMode::Reflect => {
                        in_idx[axis] = map_index(src, dim, mode);
                    }
                }
            } else {
                in_idx[axis] = src as usize;
            }
        }

        let value = if let Some((axis, is_before)) = outside_axis {
            if is_before {
                constants[axis].0
            } else {
                constants[axis].1
            }
        } else {
            match view.get(IxDyn(&in_idx)) {
                Some(v) => *v,
                None => return Err("Failed to read source element during pad".to_string()),
            }
        };

        out_data.push(value);
    }

    ArrayD::from_shape_vec(IxDyn(&out_shape), out_data)
        .map_err(|e| format!("Failed to create padded array: {}", e))
}

/// Pad an array with constant/symmetric/reflect mode.
#[no_mangle]
pub unsafe extern "C" fn ndarray_pad(
    handle: *const NdArrayHandle,
    meta: *const ViewMetadata,
    pad_width: *const usize,
    mode: i32,
    constant_values: *const f64,
    constant_values_len: usize,
    out_handle: *mut *mut NdArrayHandle,
    out_dtype: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if handle.is_null()
        || meta.is_null()
        || pad_width.is_null()
        || out_handle.is_null()
        || out_dtype.is_null()
        || out_shape.is_null()
        || out_ndim.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let meta_ref = &*meta;
        let ndim = meta_ref.ndim;
        let pad_slice = slice::from_raw_parts(pad_width, ndim * 2);
        let constant_slice = if constant_values.is_null() || constant_values_len == 0 {
            &[][..]
        } else {
            slice::from_raw_parts(constant_values, constant_values_len)
        };

        let pad_mode = match PadMode::from_i32(mode) {
            Ok(m) => m,
            Err(e) => {
                error::set_last_error(e);
                return ERR_GENERIC;
            }
        };

        if ndim == 0 {
            error::set_last_error("pad() requires at least 1 dimension".to_string());
            return ERR_SHAPE;
        }

        let mut pad_pairs = Vec::with_capacity(ndim);
        for axis in 0..ndim {
            pad_pairs.push((pad_slice[axis * 2], pad_slice[axis * 2 + 1]));
        }

        let result_wrapper = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, meta_ref)
                else {
                    error::set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                let consts = parse_constants(constant_slice, ndim, |x| x);
                let result = match pad_view(view, &pad_pairs, pad_mode, &consts) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, meta_ref)
                else {
                    error::set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                let consts = parse_constants(constant_slice, ndim, |x| x as f32);
                let result = match pad_view(view, &pad_pairs, pad_mode, &consts) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(result))),
                    dtype: DType::Float32,
                }
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, meta_ref)
                else {
                    error::set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                let consts = parse_constants(constant_slice, ndim, |x| x as i64);
                let result = match pad_view(view, &pad_pairs, pad_mode, &consts) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Int64(Arc::new(RwLock::new(result))),
                    dtype: DType::Int64,
                }
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, meta_ref)
                else {
                    error::set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                let consts = parse_constants(constant_slice, ndim, |x| x as i32);
                let result = match pad_view(view, &pad_pairs, pad_mode, &consts) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Int32(Arc::new(RwLock::new(result))),
                    dtype: DType::Int32,
                }
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, meta_ref)
                else {
                    error::set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                let consts = parse_constants(constant_slice, ndim, |x| x as i16);
                let result = match pad_view(view, &pad_pairs, pad_mode, &consts) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Int16(Arc::new(RwLock::new(result))),
                    dtype: DType::Int16,
                }
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, meta_ref)
                else {
                    error::set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                let consts = parse_constants(constant_slice, ndim, |x| x as i8);
                let result = match pad_view(view, &pad_pairs, pad_mode, &consts) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Int8(Arc::new(RwLock::new(result))),
                    dtype: DType::Int8,
                }
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, meta_ref)
                else {
                    error::set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                let consts = parse_constants(constant_slice, ndim, |x| x as u64);
                let result = match pad_view(view, &pad_pairs, pad_mode, &consts) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Uint64(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint64,
                }
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, meta_ref)
                else {
                    error::set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                let consts = parse_constants(constant_slice, ndim, |x| x as u32);
                let result = match pad_view(view, &pad_pairs, pad_mode, &consts) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Uint32(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint32,
                }
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, meta_ref)
                else {
                    error::set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                let consts = parse_constants(constant_slice, ndim, |x| x as u16);
                let result = match pad_view(view, &pad_pairs, pad_mode, &consts) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Uint16(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint16,
                }
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, meta_ref)
                else {
                    error::set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                let consts = parse_constants(constant_slice, ndim, |x| x as u8);
                let result = match pad_view(view, &pad_pairs, pad_mode, &consts) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Uint8(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint8,
                }
            }
            DType::Bool => {
                let Some(view) = extract_view_bool(wrapper, meta_ref)
                else {
                    error::set_last_error("Failed to extract bool view".to_string());
                    return ERR_GENERIC;
                };
                let consts =
                    parse_constants(constant_slice, ndim, |x| if x != 0.0 { 1u8 } else { 0u8 });
                let result = match pad_view(view, &pad_pairs, pad_mode, &consts) {
                    Ok(v) => v,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Bool(Arc::new(RwLock::new(result))),
                    dtype: DType::Bool,
                }
            }
        };

        if let Err(e) =
            write_output_metadata(&result_wrapper, out_dtype, out_ndim, out_shape, max_ndim)
        {
            error::set_last_error(e);
            return ERR_GENERIC;
        }

        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
