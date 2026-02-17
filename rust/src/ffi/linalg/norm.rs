//! Vector and matrix norms.

use crate::core::view_helpers::extract_view_as_f64;
use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{self, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::ffi::reductions::helpers::validate_axis;
use crate::ffi::NdArrayHandle;
use ndarray::{ArrayD, Axis, IxDyn};
use parking_lot::RwLock;
use std::ffi::c_void;
use std::slice;
use std::sync::Arc;

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

/// Compute scalar norm (axis=None).
#[no_mangle]
pub unsafe extern "C" fn ndarray_norm(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
    ord: i32,
    out_value: *mut c_void,
    out_dtype: *mut u8,
) -> i32 {
    if handle.is_null() || shape.is_null() || strides.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = slice::from_raw_parts(shape, ndim);
        let strides_slice = slice::from_raw_parts(strides, ndim);

        let ord = match NormOrd::from_i32(ord) {
            Ok(o) => o,
            Err(e) => {
                error::set_last_error(e);
                return ERR_GENERIC;
            }
        };

        match scalar_norm(wrapper, offset, shape_slice, strides_slice, ord) {
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

/// Compute norm along axis.
#[no_mangle]
pub unsafe extern "C" fn ndarray_norm_axis(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
    axis: i32,
    keepdims: bool,
    ord: i32,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || shape.is_null() || strides.is_null() || out_handle.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = slice::from_raw_parts(shape, ndim);
        let strides_slice = slice::from_raw_parts(strides, ndim);

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

        let axis = match validate_axis(shape_slice, axis) {
            Ok(v) => v,
            Err(e) => {
                error::set_last_error(e);
                return ERR_SHAPE;
            }
        };

        match axis_norm(wrapper, offset, shape_slice, strides_slice, axis, keepdims, ord) {
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

fn scalar_norm(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
    ord: NormOrd,
) -> Result<f64, String> {
    let view = extract_view_as_f64(wrapper, offset, shape, strides)
        .ok_or_else(|| "Failed to extract view for norm".to_string())?;

    match ord {
        NormOrd::One => Ok(view.iter().map(|x| x.abs()).sum()),
        NormOrd::Two => Ok(view.iter().map(|x| x * x).sum::<f64>().sqrt()),
        NormOrd::Inf => Ok(view
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f64, |acc, v| if v > acc { v } else { acc })),
        NormOrd::NegInf => view
            .iter()
            .map(|x| x.abs())
            .reduce(|a, b| if a < b { a } else { b })
            .ok_or_else(|| "Cannot compute norm of empty input".to_string()),
        NormOrd::Fro => {
            if shape.len() != 2 {
                return Err("fro norm requires a 2D matrix input".to_string());
            }
            Ok(view.iter().map(|x| x * x).sum::<f64>().sqrt())
        }
    }
}

fn axis_norm(
    wrapper: &NDArrayWrapper,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: usize,
    keepdims: bool,
    ord: NormOrd,
) -> Result<ArrayD<f64>, String> {
    let view = extract_view_as_f64(wrapper, offset, shape, strides)
        .ok_or_else(|| "Failed to extract view for norm".to_string())?;

    let mut out_shape = shape.to_vec();
    let lane_len = out_shape[axis];
    out_shape.remove(axis);
    if out_shape.is_empty() {
        out_shape.push(1);
    }

    let mut out: Vec<f64> = Vec::with_capacity(shape.iter().product::<usize>() / lane_len.max(1));
    for lane in view.lanes(Axis(axis)) {
        let v = match ord {
            NormOrd::One => lane.iter().map(|x| x.abs()).sum(),
            NormOrd::Two => lane.iter().map(|x| x * x).sum::<f64>().sqrt(),
            NormOrd::Inf => lane
                .iter()
                .map(|x| x.abs())
                .fold(0.0_f64, |acc, val| if val > acc { val } else { acc }),
            NormOrd::NegInf => lane
                .iter()
                .map(|x| x.abs())
                .reduce(|a, b| if a < b { a } else { b })
                .unwrap_or(0.0),
            NormOrd::Fro => {
                return Err("fro norm is only supported with axis=None".to_string())
            }
        };
        out.push(v);
    }

    let reduced = ArrayD::from_shape_vec(IxDyn(&out_shape), out)
        .map_err(|e| format!("Failed to build norm output: {}", e))?;

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
