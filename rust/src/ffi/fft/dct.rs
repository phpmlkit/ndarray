//! Discrete cosine transforms (DCT types I–IV) and inverses via `ndrustfft`.
//!
//! Normalization matches `scipy.fft.dct` / `idct` (`norm`: backward, ortho, forward, none).
//! Kernel pairing: type **1** self-inverse; **2** ↔ **3**; **4** self-inverse.

use std::sync::Arc;

use ndarray::{ArrayD, Axis};
use ndrustfft::{nddct1, nddct2, nddct3, nddct4, DctHandler, Normalization};
use parking_lot::RwLock;

use crate::helpers::error::{self, ERR_DTYPE, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::helpers::{
    dct1_backward_post_inverse_f64, dct1_ortho_lane_f64, dct1_ortho_lane_inverse_f64,
    dct234_backward_post_inverse_f64, dct2_ortho_lane_f64, dct2_ortho_lane_inverse_f64,
    dct3_ortho_lane_f64, dct3_ortho_lane_inverse_f64, dct4_ortho_scale_from_backward_f64,
    dct_forward_scale_from_backward_c, dct_inverse_forward_scale_from_backward_f64,
    dct_ndrust_norm_is_default, map_axis_lane_f64, scale_axis_f32, scale_axis_f64,
};
use crate::helpers::{
    extract_view_as_f32, extract_view_as_f64, normalize_axis, resize_along_axis_f32,
    resize_along_axis_f64, write_output_metadata,
};
use crate::types::dtype::DType;
use crate::types::{ArrayData, ArrayMetadata, NDArrayWrapper, NdArrayHandle};

fn resolve_axes(
    shape: &[usize],
    axes_ptr: *const i32,
    n_axes: usize,
) -> Result<Vec<usize>, String> {
    let ndim = shape.len();
    if n_axes == 0 {
        return Ok((0..ndim).collect());
    }
    if axes_ptr.is_null() {
        return Err("dctn: axes is null but n_axes > 0".to_string());
    }
    let mut out = Vec::with_capacity(n_axes);
    for i in 0..n_axes {
        let ax = unsafe { *axes_ptr.add(i) };
        out.push(normalize_axis(shape, ax, false)?);
    }
    Ok(out)
}

fn validate_dct_type(t: u8) -> Result<(), String> {
    if (1..=4).contains(&t) {
        Ok(())
    } else {
        Err(format!("dct type must be 1, 2, 3, or 4, got {}", t))
    }
}

fn run_dct_backward_f64(
    input: &ArrayD<f64>,
    axis: usize,
    dct_type: u8,
) -> Result<ArrayD<f64>, String> {
    validate_dct_type(dct_type)?;
    let n = input.shape()[axis];
    let mut output = ArrayD::<f64>::zeros(input.raw_dim());
    let norm = if dct_ndrust_norm_is_default(dct_type, true) {
        Normalization::Default
    } else {
        Normalization::None
    };
    let handler = DctHandler::<f64>::new(n).normalization(norm);
    match dct_type {
        1 => nddct1(input, &mut output, &handler, axis),
        2 => nddct2(input, &mut output, &handler, axis),
        3 => nddct3(input, &mut output, &handler, axis),
        4 => nddct4(input, &mut output, &handler, axis),
        _ => unreachable!(),
    }
    Ok(output)
}

fn run_dct_raw_f64(input: &ArrayD<f64>, axis: usize, dct_type: u8) -> Result<ArrayD<f64>, String> {
    validate_dct_type(dct_type)?;
    let n = input.shape()[axis];
    let mut output = ArrayD::<f64>::zeros(input.raw_dim());
    let handler = DctHandler::<f64>::new(n).normalization(Normalization::None);
    match dct_type {
        1 => nddct1(input, &mut output, &handler, axis),
        2 => nddct2(input, &mut output, &handler, axis),
        3 => nddct3(input, &mut output, &handler, axis),
        4 => nddct4(input, &mut output, &handler, axis),
        _ => unreachable!(),
    }
    Ok(output)
}

fn run_dct_ortho_f64(
    input: &ArrayD<f64>,
    axis: usize,
    dct_type: u8,
) -> Result<ArrayD<f64>, String> {
    validate_dct_type(dct_type)?;
    let n = input.shape()[axis];
    let mut output = ArrayD::<f64>::zeros(input.raw_dim());
    match dct_type {
        1 => map_axis_lane_f64(input, &mut output, axis, |x, o| {
            dct1_ortho_lane_f64(x, o);
        }),
        2 => map_axis_lane_f64(input, &mut output, axis, |x, o| {
            dct2_ortho_lane_f64(x, o);
        }),
        3 => map_axis_lane_f64(input, &mut output, axis, |x, o| {
            dct3_ortho_lane_f64(x, o);
        }),
        4 => {
            let mut tmp = run_dct_backward_f64(input, axis, 4)?;
            scale_axis_f64(&mut tmp, axis, dct4_ortho_scale_from_backward_f64(n));
            output = tmp;
        }
        _ => unreachable!(),
    }
    Ok(output)
}

fn run_dct_f64(
    input: &ArrayD<f64>,
    axis: usize,
    dct_type: u8,
    norm: u8,
) -> Result<ArrayD<f64>, String> {
    let n = input.shape()[axis];
    match norm {
        3 => run_dct_raw_f64(input, axis, dct_type),
        1 => run_dct_ortho_f64(input, axis, dct_type),
        2 => {
            let mut out = run_dct_backward_f64(input, axis, dct_type)?;
            scale_axis_f64(
                &mut out,
                axis,
                dct_forward_scale_from_backward_c(dct_type, n),
            );
            Ok(out)
        }
        _ => run_dct_backward_f64(input, axis, dct_type),
    }
}

fn run_idct_backward_f64(
    input: &ArrayD<f64>,
    axis: usize,
    dct_type: u8,
) -> Result<ArrayD<f64>, String> {
    validate_dct_type(dct_type)?;
    let n = input.shape()[axis];
    let mut output = ArrayD::<f64>::zeros(input.raw_dim());
    let handler = DctHandler::<f64>::new(n).normalization(Normalization::None);
    match dct_type {
        1 => nddct1(input, &mut output, &handler, axis),
        2 => nddct3(input, &mut output, &handler, axis),
        3 => nddct2(input, &mut output, &handler, axis),
        4 => nddct4(input, &mut output, &handler, axis),
        _ => unreachable!(),
    }
    let scale = match dct_type {
        1 => dct1_backward_post_inverse_f64(n),
        _ => dct234_backward_post_inverse_f64(n),
    };
    scale_axis_f64(&mut output, axis, scale);
    Ok(output)
}

fn run_idct_raw_f64(input: &ArrayD<f64>, axis: usize, dct_type: u8) -> Result<ArrayD<f64>, String> {
    validate_dct_type(dct_type)?;
    let n = input.shape()[axis];
    let mut output = ArrayD::<f64>::zeros(input.raw_dim());
    let handler = DctHandler::<f64>::new(n).normalization(Normalization::None);
    match dct_type {
        1 => nddct1(input, &mut output, &handler, axis),
        2 => nddct3(input, &mut output, &handler, axis),
        3 => nddct2(input, &mut output, &handler, axis),
        4 => nddct4(input, &mut output, &handler, axis),
        _ => unreachable!(),
    }
    Ok(output)
}

fn run_idct_ortho_f64(
    input: &ArrayD<f64>,
    axis: usize,
    dct_type: u8,
) -> Result<ArrayD<f64>, String> {
    validate_dct_type(dct_type)?;
    let n = input.shape()[axis];
    let mut output = ArrayD::<f64>::zeros(input.raw_dim());
    match dct_type {
        1 => map_axis_lane_f64(input, &mut output, axis, |x, o| {
            dct1_ortho_lane_inverse_f64(x, o);
        }),
        2 => map_axis_lane_f64(input, &mut output, axis, |x, o| {
            dct2_ortho_lane_inverse_f64(x, o);
        }),
        3 => map_axis_lane_f64(input, &mut output, axis, |x, o| {
            dct3_ortho_lane_inverse_f64(x, o);
        }),
        4 => {
            let mut tmp = run_dct_backward_f64(input, axis, 4)?;
            scale_axis_f64(&mut tmp, axis, dct4_ortho_scale_from_backward_f64(n));
            output = tmp;
        }
        _ => unreachable!(),
    }
    Ok(output)
}

fn run_idct_f64(
    input: &ArrayD<f64>,
    axis: usize,
    dct_type: u8,
    norm: u8,
) -> Result<ArrayD<f64>, String> {
    let n = input.shape()[axis];
    match norm {
        3 => run_idct_raw_f64(input, axis, dct_type),
        1 => run_idct_ortho_f64(input, axis, dct_type),
        2 => {
            let mut out = run_idct_backward_f64(input, axis, dct_type)?;
            scale_axis_f64(
                &mut out,
                axis,
                dct_inverse_forward_scale_from_backward_f64(dct_type, n),
            );
            Ok(out)
        }
        _ => run_idct_backward_f64(input, axis, dct_type),
    }
}

fn map_lane_f32_ortho(
    input: &ArrayD<f32>,
    output: &mut ArrayD<f32>,
    axis: usize,
    f: fn(&[f64], &mut [f64]),
) {
    assert_eq!(input.shape(), output.shape());
    for (sl, mut dl) in input
        .lanes(Axis(axis))
        .into_iter()
        .zip(output.lanes_mut(Axis(axis)))
    {
        let n = sl.len();
        let mut tmp_in = vec![0.0_f64; n];
        let mut tmp_out = vec![0.0_f64; n];
        for (i, &v) in sl.iter().enumerate() {
            tmp_in[i] = v as f64;
        }
        f(&tmp_in, &mut tmp_out);
        for (i, v) in dl.iter_mut().enumerate() {
            *v = tmp_out[i] as f32;
        }
    }
}

fn run_dct_backward_f32(
    input: &ArrayD<f32>,
    axis: usize,
    dct_type: u8,
) -> Result<ArrayD<f32>, String> {
    validate_dct_type(dct_type)?;
    let n = input.shape()[axis];
    let mut output = ArrayD::<f32>::zeros(input.raw_dim());
    let norm = if dct_ndrust_norm_is_default(dct_type, true) {
        Normalization::Default
    } else {
        Normalization::None
    };
    let handler = DctHandler::<f32>::new(n).normalization(norm);
    match dct_type {
        1 => nddct1(input, &mut output, &handler, axis),
        2 => nddct2(input, &mut output, &handler, axis),
        3 => nddct3(input, &mut output, &handler, axis),
        4 => nddct4(input, &mut output, &handler, axis),
        _ => unreachable!(),
    }
    Ok(output)
}

fn run_dct_raw_f32(input: &ArrayD<f32>, axis: usize, dct_type: u8) -> Result<ArrayD<f32>, String> {
    validate_dct_type(dct_type)?;
    let n = input.shape()[axis];
    let mut output = ArrayD::<f32>::zeros(input.raw_dim());
    let handler = DctHandler::<f32>::new(n).normalization(Normalization::None);
    match dct_type {
        1 => nddct1(input, &mut output, &handler, axis),
        2 => nddct2(input, &mut output, &handler, axis),
        3 => nddct3(input, &mut output, &handler, axis),
        4 => nddct4(input, &mut output, &handler, axis),
        _ => unreachable!(),
    }
    Ok(output)
}

fn run_dct_ortho_f32(
    input: &ArrayD<f32>,
    axis: usize,
    dct_type: u8,
) -> Result<ArrayD<f32>, String> {
    validate_dct_type(dct_type)?;
    let n = input.shape()[axis];
    let mut output = ArrayD::<f32>::zeros(input.raw_dim());
    match dct_type {
        1 => map_lane_f32_ortho(input, &mut output, axis, dct1_ortho_lane_f64),
        2 => map_lane_f32_ortho(input, &mut output, axis, dct2_ortho_lane_f64),
        3 => map_lane_f32_ortho(input, &mut output, axis, dct3_ortho_lane_f64),
        4 => {
            let mut tmp = run_dct_backward_f32(input, axis, 4)?;
            scale_axis_f32(&mut tmp, axis, dct4_ortho_scale_from_backward_f64(n) as f32);
            output = tmp;
        }
        _ => unreachable!(),
    }
    Ok(output)
}

fn run_dct_f32(
    input: &ArrayD<f32>,
    axis: usize,
    dct_type: u8,
    norm: u8,
) -> Result<ArrayD<f32>, String> {
    let n = input.shape()[axis];
    match norm {
        3 => run_dct_raw_f32(input, axis, dct_type),
        1 => run_dct_ortho_f32(input, axis, dct_type),
        2 => {
            let mut out = run_dct_backward_f32(input, axis, dct_type)?;
            scale_axis_f32(
                &mut out,
                axis,
                dct_forward_scale_from_backward_c(dct_type, n) as f32,
            );
            Ok(out)
        }
        _ => run_dct_backward_f32(input, axis, dct_type),
    }
}

fn run_idct_backward_f32(
    input: &ArrayD<f32>,
    axis: usize,
    dct_type: u8,
) -> Result<ArrayD<f32>, String> {
    validate_dct_type(dct_type)?;
    let n = input.shape()[axis];
    let mut output = ArrayD::<f32>::zeros(input.raw_dim());
    let handler = DctHandler::<f32>::new(n).normalization(Normalization::None);
    match dct_type {
        1 => nddct1(input, &mut output, &handler, axis),
        2 => nddct3(input, &mut output, &handler, axis),
        3 => nddct2(input, &mut output, &handler, axis),
        4 => nddct4(input, &mut output, &handler, axis),
        _ => unreachable!(),
    }
    let scale = match dct_type {
        1 => dct1_backward_post_inverse_f64(n) as f32,
        _ => dct234_backward_post_inverse_f64(n) as f32,
    };
    scale_axis_f32(&mut output, axis, scale);
    Ok(output)
}

fn run_idct_raw_f32(input: &ArrayD<f32>, axis: usize, dct_type: u8) -> Result<ArrayD<f32>, String> {
    validate_dct_type(dct_type)?;
    let n = input.shape()[axis];
    let mut output = ArrayD::<f32>::zeros(input.raw_dim());
    let handler = DctHandler::<f32>::new(n).normalization(Normalization::None);
    match dct_type {
        1 => nddct1(input, &mut output, &handler, axis),
        2 => nddct3(input, &mut output, &handler, axis),
        3 => nddct2(input, &mut output, &handler, axis),
        4 => nddct4(input, &mut output, &handler, axis),
        _ => unreachable!(),
    }
    Ok(output)
}

fn run_idct_ortho_f32(
    input: &ArrayD<f32>,
    axis: usize,
    dct_type: u8,
) -> Result<ArrayD<f32>, String> {
    validate_dct_type(dct_type)?;
    let n = input.shape()[axis];
    let mut output = ArrayD::<f32>::zeros(input.raw_dim());
    match dct_type {
        1 => map_lane_f32_ortho(input, &mut output, axis, dct1_ortho_lane_inverse_f64),
        2 => map_lane_f32_ortho(input, &mut output, axis, dct2_ortho_lane_inverse_f64),
        3 => map_lane_f32_ortho(input, &mut output, axis, dct3_ortho_lane_inverse_f64),
        4 => {
            let mut tmp = run_dct_backward_f32(input, axis, 4)?;
            scale_axis_f32(&mut tmp, axis, dct4_ortho_scale_from_backward_f64(n) as f32);
            output = tmp;
        }
        _ => unreachable!(),
    }
    Ok(output)
}

fn run_idct_f32(
    input: &ArrayD<f32>,
    axis: usize,
    dct_type: u8,
    norm: u8,
) -> Result<ArrayD<f32>, String> {
    let n = input.shape()[axis];
    match norm {
        3 => run_idct_raw_f32(input, axis, dct_type),
        1 => run_idct_ortho_f32(input, axis, dct_type),
        2 => {
            let mut out = run_idct_backward_f32(input, axis, dct_type)?;
            scale_axis_f32(
                &mut out,
                axis,
                dct_inverse_forward_scale_from_backward_f64(dct_type, n) as f32,
            );
            Ok(out)
        }
        _ => run_idct_backward_f32(input, axis, dct_type),
    }
}

fn dctn_f64(
    mut current: ArrayD<f64>,
    axes: &[usize],
    dct_type: u8,
    norm: u8,
) -> Result<ArrayD<f64>, String> {
    for &ax in axes {
        current = run_dct_f64(&current, ax, dct_type, norm)?;
    }
    Ok(current)
}

fn idctn_f64(
    mut current: ArrayD<f64>,
    axes: &[usize],
    dct_type: u8,
    norm: u8,
) -> Result<ArrayD<f64>, String> {
    for &ax in axes {
        current = run_idct_f64(&current, ax, dct_type, norm)?;
    }
    Ok(current)
}

fn dctn_f32(
    mut current: ArrayD<f32>,
    axes: &[usize],
    dct_type: u8,
    norm: u8,
) -> Result<ArrayD<f32>, String> {
    for &ax in axes {
        current = run_dct_f32(&current, ax, dct_type, norm)?;
    }
    Ok(current)
}

fn idctn_f32(
    mut current: ArrayD<f32>,
    axes: &[usize],
    dct_type: u8,
    norm: u8,
) -> Result<ArrayD<f32>, String> {
    for &ax in axes {
        current = run_idct_f32(&current, ax, dct_type, norm)?;
    }
    Ok(current)
}

/// Real DCT along one axis (`dct_type` 1–4).
#[no_mangle]
pub unsafe extern "C" fn ndarray_dct(
    handle: *const NdArrayHandle,
    meta: *const ArrayMetadata,
    axis: i32,
    n: usize,
    dct_type: u8,
    norm: u8,
    out_handle: *mut *mut NdArrayHandle,
    out_dtype_ptr: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if handle.is_null()
        || meta.is_null()
        || out_handle.is_null()
        || out_dtype_ptr.is_null()
        || out_ndim.is_null()
        || out_shape.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let meta_ref = &*meta;
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape = wrapper.shape();
        if shape.is_empty() {
            error::set_last_error("dct requires at least one dimension".to_string());
            return ERR_SHAPE;
        }

        if let Err(e) = validate_dct_type(dct_type) {
            error::set_last_error(e);
            return ERR_SHAPE;
        }

        let axis_n = match normalize_axis(&shape, axis, false) {
            Ok(a) => a,
            Err(e) => {
                error::set_last_error(e);
                return ERR_SHAPE;
            }
        };

        let len_along = shape[axis_n];
        let target_n = if n == 0 { len_along } else { n };

        let result_wrapper = match wrapper.dtype {
            DType::Float64
            | DType::Int8
            | DType::Int16
            | DType::Int32
            | DType::Int64
            | DType::Uint8
            | DType::Uint16
            | DType::Uint32
            | DType::Uint64 => {
                let arr = if let Some(v) = extract_view_as_f64(wrapper, meta_ref) {
                    v
                } else {
                    error::set_last_error("dct: failed to read real input".to_string());
                    return ERR_GENERIC;
                };
                let arr = resize_along_axis_f64(&arr, axis_n, target_n);
                let out = match run_dct_f64(&arr, axis_n, dct_type, norm) {
                    Ok(o) => o,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(out))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let arr = if let Some(v) = extract_view_as_f32(wrapper, meta_ref) {
                    v
                } else {
                    error::set_last_error("dct: failed to read Float32 input".to_string());
                    return ERR_GENERIC;
                };
                let arr = resize_along_axis_f32(&arr, axis_n, target_n);
                let out = match run_dct_f32(&arr, axis_n, dct_type, norm) {
                    Ok(o) => o,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(out))),
                    dtype: DType::Float32,
                }
            }
            DType::Complex64 | DType::Complex128 => {
                error::set_last_error("dct expects a real dtype".to_string());
                return ERR_DTYPE;
            }
            DType::Bool => {
                error::set_last_error("dct does not support bool dtype".to_string());
                return ERR_DTYPE;
            }
        };

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

/// Inverse DCT along one axis (`dct_type` 1–4, same convention as SciPy `idct`).
#[no_mangle]
pub unsafe extern "C" fn ndarray_idct(
    handle: *const NdArrayHandle,
    meta: *const ArrayMetadata,
    axis: i32,
    n: usize,
    dct_type: u8,
    norm: u8,
    out_handle: *mut *mut NdArrayHandle,
    out_dtype_ptr: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if handle.is_null()
        || meta.is_null()
        || out_handle.is_null()
        || out_dtype_ptr.is_null()
        || out_ndim.is_null()
        || out_shape.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let meta_ref = &*meta;
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape = wrapper.shape();
        if shape.is_empty() {
            error::set_last_error("idct requires at least one dimension".to_string());
            return ERR_SHAPE;
        }

        if let Err(e) = validate_dct_type(dct_type) {
            error::set_last_error(e);
            return ERR_SHAPE;
        }

        let axis_n = match normalize_axis(&shape, axis, false) {
            Ok(a) => a,
            Err(e) => {
                error::set_last_error(e);
                return ERR_SHAPE;
            }
        };

        let len_along = shape[axis_n];
        let target_n = if n == 0 { len_along } else { n };

        let result_wrapper = match wrapper.dtype {
            DType::Float64
            | DType::Int8
            | DType::Int16
            | DType::Int32
            | DType::Int64
            | DType::Uint8
            | DType::Uint16
            | DType::Uint32
            | DType::Uint64 => {
                let arr = if let Some(v) = extract_view_as_f64(wrapper, meta_ref) {
                    v
                } else {
                    error::set_last_error("idct: failed to read real input".to_string());
                    return ERR_GENERIC;
                };
                let arr = resize_along_axis_f64(&arr, axis_n, target_n);
                let out = match run_idct_f64(&arr, axis_n, dct_type, norm) {
                    Ok(o) => o,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(out))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let arr = if let Some(v) = extract_view_as_f32(wrapper, meta_ref) {
                    v
                } else {
                    error::set_last_error("idct: failed to read Float32 input".to_string());
                    return ERR_GENERIC;
                };
                let arr = resize_along_axis_f32(&arr, axis_n, target_n);
                let out = match run_idct_f32(&arr, axis_n, dct_type, norm) {
                    Ok(o) => o,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(out))),
                    dtype: DType::Float32,
                }
            }
            DType::Complex64 | DType::Complex128 => {
                error::set_last_error("idct expects a real dtype".to_string());
                return ERR_DTYPE;
            }
            DType::Bool => {
                error::set_last_error("idct does not support bool dtype".to_string());
                return ERR_DTYPE;
            }
        };

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

/// N-dimensional DCT over `axes` (empty = all axes).
#[no_mangle]
pub unsafe extern "C" fn ndarray_dctn(
    handle: *const NdArrayHandle,
    meta: *const ArrayMetadata,
    axes: *const i32,
    n_axes: usize,
    dct_type: u8,
    norm: u8,
    out_handle: *mut *mut NdArrayHandle,
    out_dtype_ptr: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if handle.is_null()
        || meta.is_null()
        || out_handle.is_null()
        || out_dtype_ptr.is_null()
        || out_ndim.is_null()
        || out_shape.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let meta_ref = &*meta;
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape = wrapper.shape();
        if shape.is_empty() {
            error::set_last_error("dctn requires at least one dimension".to_string());
            return ERR_SHAPE;
        }

        if let Err(e) = validate_dct_type(dct_type) {
            error::set_last_error(e);
            return ERR_SHAPE;
        }

        let axes_v = match resolve_axes(&shape, axes, n_axes) {
            Ok(a) => a,
            Err(e) => {
                error::set_last_error(e);
                return ERR_SHAPE;
            }
        };

        let result_wrapper = match wrapper.dtype {
            DType::Float64
            | DType::Int8
            | DType::Int16
            | DType::Int32
            | DType::Int64
            | DType::Uint8
            | DType::Uint16
            | DType::Uint32
            | DType::Uint64 => {
                let arr = if let Some(v) = extract_view_as_f64(wrapper, meta_ref) {
                    v
                } else {
                    error::set_last_error("dctn: failed to read real input".to_string());
                    return ERR_GENERIC;
                };
                let out = match dctn_f64(arr, &axes_v, dct_type, norm) {
                    Ok(o) => o,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(out))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let arr = if let Some(v) = extract_view_as_f32(wrapper, meta_ref) {
                    v
                } else {
                    error::set_last_error("dctn: failed to read Float32 input".to_string());
                    return ERR_GENERIC;
                };
                let out = match dctn_f32(arr, &axes_v, dct_type, norm) {
                    Ok(o) => o,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(out))),
                    dtype: DType::Float32,
                }
            }
            DType::Complex64 | DType::Complex128 => {
                error::set_last_error("dctn expects a real dtype".to_string());
                return ERR_DTYPE;
            }
            DType::Bool => {
                error::set_last_error("dctn does not support bool dtype".to_string());
                return ERR_DTYPE;
            }
        };

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

/// N-dimensional inverse DCT over `axes` (empty = all axes).
#[no_mangle]
pub unsafe extern "C" fn ndarray_idctn(
    handle: *const NdArrayHandle,
    meta: *const ArrayMetadata,
    axes: *const i32,
    n_axes: usize,
    dct_type: u8,
    norm: u8,
    out_handle: *mut *mut NdArrayHandle,
    out_dtype_ptr: *mut u8,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if handle.is_null()
        || meta.is_null()
        || out_handle.is_null()
        || out_dtype_ptr.is_null()
        || out_ndim.is_null()
        || out_shape.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let meta_ref = &*meta;
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape = wrapper.shape();
        if shape.is_empty() {
            error::set_last_error("idctn requires at least one dimension".to_string());
            return ERR_SHAPE;
        }

        if let Err(e) = validate_dct_type(dct_type) {
            error::set_last_error(e);
            return ERR_SHAPE;
        }

        let axes_v = match resolve_axes(&shape, axes, n_axes) {
            Ok(a) => a,
            Err(e) => {
                error::set_last_error(e);
                return ERR_SHAPE;
            }
        };

        let result_wrapper = match wrapper.dtype {
            DType::Float64
            | DType::Int8
            | DType::Int16
            | DType::Int32
            | DType::Int64
            | DType::Uint8
            | DType::Uint16
            | DType::Uint32
            | DType::Uint64 => {
                let arr = if let Some(v) = extract_view_as_f64(wrapper, meta_ref) {
                    v
                } else {
                    error::set_last_error("idctn: failed to read real input".to_string());
                    return ERR_GENERIC;
                };
                let out = match idctn_f64(arr, &axes_v, dct_type, norm) {
                    Ok(o) => o,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(out))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let arr = if let Some(v) = extract_view_as_f32(wrapper, meta_ref) {
                    v
                } else {
                    error::set_last_error("idctn: failed to read Float32 input".to_string());
                    return ERR_GENERIC;
                };
                let out = match idctn_f32(arr, &axes_v, dct_type, norm) {
                    Ok(o) => o,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(out))),
                    dtype: DType::Float32,
                }
            }
            DType::Complex64 | DType::Complex128 => {
                error::set_last_error("idctn expects a real dtype".to_string());
                return ERR_DTYPE;
            }
            DType::Bool => {
                error::set_last_error("idctn does not support bool dtype".to_string());
                return ERR_DTYPE;
            }
        };

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
