//! Vector and matrix norms.

use std::ffi::c_void;
use std::sync::Arc;

use ndarray::{ArrayD, ArrayViewD, Axis, IxDyn};
use num_complex::{Complex32, Complex64};
use num_traits::ToPrimitive;
use parking_lot::RwLock;

use crate::helpers::error::{self, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::helpers::normalize_axis;
use crate::helpers::{
    extract_view_bool, extract_view_c128, extract_view_c64, extract_view_f32, extract_view_f64,
    extract_view_i16, extract_view_i32, extract_view_i64, extract_view_i8, extract_view_u16,
    extract_view_u32, extract_view_u64, extract_view_u8,
};
use crate::types::dtype::DType;
use crate::types::{ArrayData, ArrayMetadata, NDArrayWrapper, NdArrayHandle};

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NormOrd {
    One = 1,
    Two = 2,
    Inf = 3,
    NegInf = 4,
    Fro = 5,
}

impl NormOrd {
    fn from_i32(v: i32) -> Result<Self, String> {
        match v {
            1 => Ok(Self::One),
            2 => Ok(Self::Two),
            3 => Ok(Self::Inf),
            4 => Ok(Self::NegInf),
            5 => Ok(Self::Fro),
            _ => Err(format!("Invalid norm order code: {}", v)),
        }
    }
}

fn norm_scalar_real<T>(view: ArrayViewD<T>, shape: &[usize], ord: NormOrd) -> Result<f64, String>
where
    T: Copy + ToPrimitive,
{
    let to = |x: T| x.to_f64().unwrap_or(0.0);
    match ord {
        NormOrd::One => Ok(view.iter().map(|&x| to(x).abs()).sum()),
        NormOrd::Two => Ok(view
            .iter()
            .map(|&x| {
                let v = to(x);
                v * v
            })
            .sum::<f64>()
            .sqrt()),
        NormOrd::Inf => {
            Ok(view
                .iter()
                .map(|&x| to(x).abs())
                .fold(0.0_f64, |acc, v| if v > acc { v } else { acc }))
        }
        NormOrd::NegInf => view
            .iter()
            .map(|&x| to(x).abs())
            .reduce(|a, b| if a < b { a } else { b })
            .ok_or_else(|| "Cannot compute norm of empty input".to_string()),
        NormOrd::Fro => {
            if shape.len() != 2 {
                return Err("fro norm requires a 2D matrix input".to_string());
            }
            Ok(view
                .iter()
                .map(|&x| {
                    let v = to(x);
                    v * v
                })
                .sum::<f64>()
                .sqrt())
        }
    }
}

fn norm_scalar_complex32(
    view: ArrayViewD<Complex32>,
    shape: &[usize],
    ord: NormOrd,
) -> Result<f64, String> {
    match ord {
        NormOrd::One => Ok(view.iter().map(|x| x.norm() as f64).sum()),
        NormOrd::Two => Ok(view
            .iter()
            .map(|x| {
                let n = x.norm() as f64;
                n * n
            })
            .sum::<f64>()
            .sqrt()),
        NormOrd::Inf => Ok(view
            .iter()
            .map(|x| x.norm() as f64)
            .fold(0.0_f64, |acc, v| if v > acc { v } else { acc })),
        NormOrd::NegInf => view
            .iter()
            .map(|x| x.norm() as f64)
            .reduce(|a, b| if a < b { a } else { b })
            .ok_or_else(|| "Cannot compute norm of empty input".to_string()),
        NormOrd::Fro => {
            if shape.len() != 2 {
                return Err("fro norm requires a 2D matrix input".to_string());
            }
            Ok(view
                .iter()
                .map(|x| {
                    let n = x.norm() as f64;
                    n * n
                })
                .sum::<f64>()
                .sqrt())
        }
    }
}

fn norm_scalar_complex64(
    view: ArrayViewD<Complex64>,
    shape: &[usize],
    ord: NormOrd,
) -> Result<f64, String> {
    match ord {
        NormOrd::One => Ok(view.iter().map(|x| x.norm()).sum()),
        NormOrd::Two => Ok(view.iter().map(|x| x.norm().powi(2)).sum::<f64>().sqrt()),
        NormOrd::Inf => {
            Ok(view
                .iter()
                .map(|x| x.norm())
                .fold(0.0_f64, |acc, v| if v > acc { v } else { acc }))
        }
        NormOrd::NegInf => view
            .iter()
            .map(|x| x.norm())
            .reduce(|a, b| if a < b { a } else { b })
            .ok_or_else(|| "Cannot compute norm of empty input".to_string()),
        NormOrd::Fro => {
            if shape.len() != 2 {
                return Err("fro norm requires a 2D matrix input".to_string());
            }
            Ok(view.iter().map(|x| x.norm().powi(2)).sum::<f64>().sqrt())
        }
    }
}

fn finalize_axis_output(
    reduced: ArrayD<f64>,
    shape: &[usize],
    axis: usize,
    keepdims: bool,
) -> Result<ArrayD<f64>, String> {
    if keepdims {
        let mut kd_shape = shape.to_vec();
        kd_shape[axis] = 1;
        return reduced
            .into_shape_with_order(IxDyn(&kd_shape))
            .map_err(|e| format!("Failed to apply keepdims shape: {}", e));
    }
    if shape.len() == 1 {
        return reduced
            .into_shape_with_order(IxDyn(&[]))
            .map_err(|e| format!("Failed to shape scalar output: {}", e));
    }
    Ok(reduced)
}

fn norm_axis_real<T>(
    view: ArrayViewD<T>,
    shape: &[usize],
    axis: usize,
    keepdims: bool,
    ord: NormOrd,
) -> Result<ArrayD<f64>, String>
where
    T: Copy + ToPrimitive,
{
    let to = |x: T| x.to_f64().unwrap_or(0.0);
    let mut out_shape = shape.to_vec();
    let lane_len = out_shape[axis];
    out_shape.remove(axis);
    if out_shape.is_empty() {
        out_shape.push(1);
    }

    let mut out: Vec<f64> = Vec::with_capacity(shape.iter().product::<usize>() / lane_len.max(1));
    for lane in view.lanes(Axis(axis)) {
        let v = match ord {
            NormOrd::One => lane.iter().map(|&x| to(x).abs()).sum(),
            NormOrd::Two => lane
                .iter()
                .map(|&x| {
                    let v = to(x);
                    v * v
                })
                .sum::<f64>()
                .sqrt(),
            NormOrd::Inf => {
                lane.iter()
                    .map(|&x| to(x).abs())
                    .fold(0.0_f64, |acc, val| if val > acc { val } else { acc })
            }
            NormOrd::NegInf => lane
                .iter()
                .map(|&x| to(x).abs())
                .reduce(|a, b| if a < b { a } else { b })
                .unwrap_or(0.0),
            NormOrd::Fro => {
                return Err("fro norm is only supported with axis=None".to_string());
            }
        };
        out.push(v);
    }

    let reduced = ArrayD::from_shape_vec(IxDyn(&out_shape), out)
        .map_err(|e| format!("Failed to build norm output: {}", e))?;

    finalize_axis_output(reduced, shape, axis, keepdims)
}

fn norm_axis_complex32(
    view: ArrayViewD<Complex32>,
    shape: &[usize],
    axis: usize,
    keepdims: bool,
    ord: NormOrd,
) -> Result<ArrayD<f64>, String> {
    let mut out_shape = shape.to_vec();
    let lane_len = out_shape[axis];
    out_shape.remove(axis);
    if out_shape.is_empty() {
        out_shape.push(1);
    }

    let mut out: Vec<f64> = Vec::with_capacity(shape.iter().product::<usize>() / lane_len.max(1));
    for lane in view.lanes(Axis(axis)) {
        let v = match ord {
            NormOrd::One => lane.iter().map(|x| x.norm() as f64).sum(),
            NormOrd::Two => lane
                .iter()
                .map(|x| {
                    let n = x.norm() as f64;
                    n * n
                })
                .sum::<f64>()
                .sqrt(),
            NormOrd::Inf => lane
                .iter()
                .map(|x| x.norm() as f64)
                .fold(0.0_f64, |acc, val| if val > acc { val } else { acc }),
            NormOrd::NegInf => lane
                .iter()
                .map(|x| x.norm() as f64)
                .reduce(|a, b| if a < b { a } else { b })
                .unwrap_or(0.0),
            NormOrd::Fro => {
                return Err("fro norm is only supported with axis=None".to_string());
            }
        };
        out.push(v);
    }

    let reduced = ArrayD::from_shape_vec(IxDyn(&out_shape), out)
        .map_err(|e| format!("Failed to build norm output: {}", e))?;

    finalize_axis_output(reduced, shape, axis, keepdims)
}

fn norm_axis_complex64(
    view: ArrayViewD<Complex64>,
    shape: &[usize],
    axis: usize,
    keepdims: bool,
    ord: NormOrd,
) -> Result<ArrayD<f64>, String> {
    let mut out_shape = shape.to_vec();
    let lane_len = out_shape[axis];
    out_shape.remove(axis);
    if out_shape.is_empty() {
        out_shape.push(1);
    }

    let mut out: Vec<f64> = Vec::with_capacity(shape.iter().product::<usize>() / lane_len.max(1));
    for lane in view.lanes(Axis(axis)) {
        let v = match ord {
            NormOrd::One => lane.iter().map(|x| x.norm()).sum(),
            NormOrd::Two => lane.iter().map(|x| x.norm().powi(2)).sum::<f64>().sqrt(),
            NormOrd::Inf => {
                lane.iter()
                    .map(|x| x.norm())
                    .fold(0.0_f64, |acc, val| if val > acc { val } else { acc })
            }
            NormOrd::NegInf => lane
                .iter()
                .map(|x| x.norm())
                .reduce(|a, b| if a < b { a } else { b })
                .unwrap_or(0.0),
            NormOrd::Fro => {
                return Err("fro norm is only supported with axis=None".to_string());
            }
        };
        out.push(v);
    }

    let reduced = ArrayD::from_shape_vec(IxDyn(&out_shape), out)
        .map_err(|e| format!("Failed to build norm output: {}", e))?;

    finalize_axis_output(reduced, shape, axis, keepdims)
}

/// Compute scalar norm.
#[no_mangle]
pub unsafe extern "C" fn ndarray_norm(
    handle: *const NdArrayHandle,
    meta: *const ArrayMetadata,
    ord: i32,
    out_value: *mut c_void,
    out_dtype: *mut u8,
) -> i32 {
    if handle.is_null() || meta.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let meta = &*meta;
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);

        let ord = match NormOrd::from_i32(ord) {
            Ok(o) => o,
            Err(e) => {
                error::set_last_error(e);
                return ERR_GENERIC;
            }
        };

        let shape = unsafe { meta.shape_slice() };

        let norm_result = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, meta) else {
                    error::set_last_error("Failed to extract Float64 view for norm".to_string());
                    return ERR_GENERIC;
                };
                norm_scalar_real(view, shape, ord)
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, meta) else {
                    error::set_last_error("Failed to extract Float32 view for norm".to_string());
                    return ERR_GENERIC;
                };
                norm_scalar_real(view, shape, ord)
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, meta) else {
                    error::set_last_error("Failed to extract Int64 view for norm".to_string());
                    return ERR_GENERIC;
                };
                norm_scalar_real(view, shape, ord)
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, meta) else {
                    error::set_last_error("Failed to extract Int32 view for norm".to_string());
                    return ERR_GENERIC;
                };
                norm_scalar_real(view, shape, ord)
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, meta) else {
                    error::set_last_error("Failed to extract Int16 view for norm".to_string());
                    return ERR_GENERIC;
                };
                norm_scalar_real(view, shape, ord)
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, meta) else {
                    error::set_last_error("Failed to extract Int8 view for norm".to_string());
                    return ERR_GENERIC;
                };
                norm_scalar_real(view, shape, ord)
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, meta) else {
                    error::set_last_error("Failed to extract Uint64 view for norm".to_string());
                    return ERR_GENERIC;
                };
                norm_scalar_real(view, shape, ord)
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, meta) else {
                    error::set_last_error("Failed to extract Uint32 view for norm".to_string());
                    return ERR_GENERIC;
                };
                norm_scalar_real(view, shape, ord)
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, meta) else {
                    error::set_last_error("Failed to extract Uint16 view for norm".to_string());
                    return ERR_GENERIC;
                };
                norm_scalar_real(view, shape, ord)
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, meta) else {
                    error::set_last_error("Failed to extract Uint8 view for norm".to_string());
                    return ERR_GENERIC;
                };
                norm_scalar_real(view, shape, ord)
            }
            DType::Bool => {
                let Some(view) = extract_view_bool(wrapper, meta) else {
                    error::set_last_error("Failed to extract Bool view for norm".to_string());
                    return ERR_GENERIC;
                };
                norm_scalar_real(view, shape, ord)
            }
            DType::Complex64 => {
                let Some(view) = extract_view_c64(wrapper, meta) else {
                    error::set_last_error("Failed to extract Complex64 view for norm".to_string());
                    return ERR_GENERIC;
                };
                norm_scalar_complex32(view, shape, ord)
            }
            DType::Complex128 => {
                let Some(view) = extract_view_c128(wrapper, meta) else {
                    error::set_last_error("Failed to extract Complex128 view for norm".to_string());
                    return ERR_GENERIC;
                };
                norm_scalar_complex64(view, shape, ord)
            }
        };

        match norm_result {
            Ok(v) => {
                if !out_value.is_null() {
                    *(out_value as *mut f64) = v;
                }
                if !out_dtype.is_null() {
                    *out_dtype = DType::Float64 as u8;
                }
                SUCCESS
            }
            Err(e) => {
                error::set_last_error(e);
                ERR_SHAPE
            }
        }
    })
}

/// Compute norm along an axis.
#[no_mangle]
pub unsafe extern "C" fn ndarray_norm_axis(
    handle: *const NdArrayHandle,
    meta: *const ArrayMetadata,
    axis: i32,
    keepdims: bool,
    ord: i32,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || meta.is_null() || out_handle.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let meta = &*meta;
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = meta.shape_slice();

        let ord = match NormOrd::from_i32(ord) {
            Ok(o) => o,
            Err(e) => {
                error::set_last_error(e);
                return ERR_GENERIC;
            }
        };

        if matches!(ord, NormOrd::Fro) {
            error::set_last_error("fro norm is only supported with axis=None".to_string());
            return ERR_SHAPE;
        }

        let axis = match normalize_axis(shape_slice, axis, false) {
            Ok(v) => v,
            Err(e) => {
                error::set_last_error(e);
                return ERR_SHAPE;
            }
        };

        let shape = unsafe { meta.shape_slice() };

        let axis_norm_result = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, meta) else {
                    error::set_last_error("Failed to extract Float64 view for norm".to_string());
                    return ERR_GENERIC;
                };
                norm_axis_real(view, shape, axis, keepdims, ord)
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, meta) else {
                    error::set_last_error("Failed to extract Float32 view for norm".to_string());
                    return ERR_GENERIC;
                };
                norm_axis_real(view, shape, axis, keepdims, ord)
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, meta) else {
                    error::set_last_error("Failed to extract Int64 view for norm".to_string());
                    return ERR_GENERIC;
                };
                norm_axis_real(view, shape, axis, keepdims, ord)
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, meta) else {
                    error::set_last_error("Failed to extract Int32 view for norm".to_string());
                    return ERR_GENERIC;
                };
                norm_axis_real(view, shape, axis, keepdims, ord)
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, meta) else {
                    error::set_last_error("Failed to extract Int16 view for norm".to_string());
                    return ERR_GENERIC;
                };
                norm_axis_real(view, shape, axis, keepdims, ord)
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, meta) else {
                    error::set_last_error("Failed to extract Int8 view for norm".to_string());
                    return ERR_GENERIC;
                };
                norm_axis_real(view, shape, axis, keepdims, ord)
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, meta) else {
                    error::set_last_error("Failed to extract Uint64 view for norm".to_string());
                    return ERR_GENERIC;
                };
                norm_axis_real(view, shape, axis, keepdims, ord)
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, meta) else {
                    error::set_last_error("Failed to extract Uint32 view for norm".to_string());
                    return ERR_GENERIC;
                };
                norm_axis_real(view, shape, axis, keepdims, ord)
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, meta) else {
                    error::set_last_error("Failed to extract Uint16 view for norm".to_string());
                    return ERR_GENERIC;
                };
                norm_axis_real(view, shape, axis, keepdims, ord)
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, meta) else {
                    error::set_last_error("Failed to extract Uint8 view for norm".to_string());
                    return ERR_GENERIC;
                };
                norm_axis_real(view, shape, axis, keepdims, ord)
            }
            DType::Bool => {
                let Some(view) = extract_view_bool(wrapper, meta) else {
                    error::set_last_error("Failed to extract Bool view for norm".to_string());
                    return ERR_GENERIC;
                };
                norm_axis_real(view, shape, axis, keepdims, ord)
            }
            DType::Complex64 => {
                let Some(view) = extract_view_c64(wrapper, meta) else {
                    error::set_last_error("Failed to extract Complex64 view for norm".to_string());
                    return ERR_GENERIC;
                };
                norm_axis_complex32(view, shape, axis, keepdims, ord)
            }
            DType::Complex128 => {
                let Some(view) = extract_view_c128(wrapper, meta) else {
                    error::set_last_error("Failed to extract Complex128 view for norm".to_string());
                    return ERR_GENERIC;
                };
                norm_axis_complex64(view, shape, axis, keepdims, ord)
            }
        };

        match axis_norm_result {
            Ok(arr) => {
                let out = NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(arr))),
                    dtype: DType::Float64,
                };
                *out_handle = NdArrayHandle::from_wrapper(Box::new(out));
                SUCCESS
            }
            Err(e) => {
                error::set_last_error(e);
                ERR_SHAPE
            }
        }
    })
}
