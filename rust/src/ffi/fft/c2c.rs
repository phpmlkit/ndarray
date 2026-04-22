//! Complex FFT / inverse FFT and n-dimensional variants (`fftn` / `ifftn`).

use std::sync::Arc;

use ndarray::ArrayD;
use ndrustfft::{ndfft, ndifft, Complex, FftHandler};
use parking_lot::RwLock;

use crate::helpers::error::{self, ERR_DTYPE, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::helpers::{
    extract_view_as_c128, extract_view_as_c64, extract_view_as_f32, extract_view_as_f64,
    extract_view_c128, extract_view_c64, fft_forward_scale_f32, fft_forward_scale_f64,
    fft_inverse_scale_f32, fft_inverse_scale_f64, fft_norm_f32, fft_norm_f64, normalize_axis,
    resize_along_axis_complex128, resize_along_axis_complex64, resize_along_axis_f32,
    resize_along_axis_f64, scale_axis_complex128, scale_axis_complex64, write_output_metadata,
};
use crate::types::dtype::DType;
use crate::types::{ArrayData, ArrayMetadata, NDArrayWrapper, NdArrayHandle};

type C64 = Complex<f32>;
type C128 = Complex<f64>;

fn resolve_fftn_axes(
    shape: &[usize],
    axes_ptr: *const i32,
    n_axes: usize,
) -> Result<Vec<usize>, String> {
    let ndim = shape.len();
    if n_axes == 0 {
        return Ok((0..ndim).collect());
    }
    if axes_ptr.is_null() {
        return Err("fftn: axes is null but n_axes > 0".to_string());
    }
    let mut out = Vec::with_capacity(n_axes);
    for i in 0..n_axes {
        let ax = unsafe { *axes_ptr.add(i) };
        out.push(normalize_axis(shape, ax, false)?);
    }
    Ok(out)
}

fn fftn_c2c_f64(mut current: ArrayD<C128>, axes: &[usize], norm: u8) -> ArrayD<C128> {
    let norm_v = fft_norm_f64(norm);
    for &axis in axes {
        let n = current.shape()[axis];
        let mut out = ArrayD::<C128>::zeros(current.raw_dim());
        let handler = FftHandler::<f64>::new(n).normalization(norm_v.clone());
        ndfft(&current.view(), &mut out.view_mut(), &handler, axis);
        scale_axis_complex128(&mut out, axis, fft_forward_scale_f64(norm, n));
        current = out;
    }
    current
}

fn ifftn_c2c_f64(mut current: ArrayD<C128>, axes: &[usize], norm: u8) -> ArrayD<C128> {
    let norm_v = fft_norm_f64(norm);
    for &axis in axes {
        let n = current.shape()[axis];
        let mut out = ArrayD::<C128>::zeros(current.raw_dim());
        let handler = FftHandler::<f64>::new(n).normalization(norm_v.clone());
        ndifft(&current.view(), &mut out.view_mut(), &handler, axis);
        scale_axis_complex128(&mut out, axis, fft_inverse_scale_f64(norm, n));
        current = out;
    }
    current
}

fn fftn_c2c_f32(mut current: ArrayD<C64>, axes: &[usize], norm: u8) -> ArrayD<C64> {
    let norm_v = fft_norm_f32(norm);
    for &axis in axes {
        let n = current.shape()[axis];
        let mut out = ArrayD::<C64>::zeros(current.raw_dim());
        let handler = FftHandler::<f32>::new(n).normalization(norm_v.clone());
        ndfft(&current.view(), &mut out.view_mut(), &handler, axis);
        scale_axis_complex64(&mut out, axis, fft_forward_scale_f32(norm, n));
        current = out;
    }
    current
}

fn ifftn_c2c_f32(mut current: ArrayD<C64>, axes: &[usize], norm: u8) -> ArrayD<C64> {
    let norm_v = fft_norm_f32(norm);
    for &axis in axes {
        let n = current.shape()[axis];
        let mut out = ArrayD::<C64>::zeros(current.raw_dim());
        let handler = FftHandler::<f32>::new(n).normalization(norm_v.clone());
        ndifft(&current.view(), &mut out.view_mut(), &handler, axis);
        scale_axis_complex64(&mut out, axis, fft_inverse_scale_f32(norm, n));
        current = out;
    }
    current
}

/// One-dimensional complex FFT along `axis`. Real inputs are promoted to complex.
/// `n == 0` keeps the current length along `axis` (optional padding length).
#[no_mangle]
pub unsafe extern "C" fn ndarray_fft(
    handle: *const NdArrayHandle,
    meta: *const ArrayMetadata,
    axis: i32,
    n: usize,
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
            error::set_last_error("fft requires at least one dimension".to_string());
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
            DType::Complex128 => {
                let arr = if let Some(v) = extract_view_c128(wrapper, meta_ref) {
                    v.to_owned()
                } else if let Some(v) = extract_view_as_c128(wrapper, meta_ref) {
                    v
                } else {
                    error::set_last_error("fft: failed to read Complex128 input".to_string());
                    return ERR_GENERIC;
                };
                let arr = resize_along_axis_complex128(&arr, axis_n, target_n);
                let out = {
                    let n_ax = arr.shape()[axis_n];
                    let mut out = ArrayD::<C128>::zeros(arr.raw_dim());
                    let handler = FftHandler::<f64>::new(n_ax).normalization(fft_norm_f64(norm));
                    ndfft(&arr.view(), &mut out.view_mut(), &handler, axis_n);
                    scale_axis_complex128(&mut out, axis_n, fft_forward_scale_f64(norm, n_ax));
                    out
                };
                NDArrayWrapper {
                    data: ArrayData::Complex128(Arc::new(RwLock::new(out))),
                    dtype: DType::Complex128,
                }
            }
            DType::Complex64 => {
                let arr = if let Some(v) = extract_view_c64(wrapper, meta_ref) {
                    v.to_owned()
                } else if let Some(v) = extract_view_as_c64(wrapper, meta_ref) {
                    v
                } else {
                    error::set_last_error("fft: failed to read Complex64 input".to_string());
                    return ERR_GENERIC;
                };
                let arr = resize_along_axis_complex64(&arr, axis_n, target_n);
                let out = {
                    let n_ax = arr.shape()[axis_n];
                    let mut out = ArrayD::<C64>::zeros(arr.raw_dim());
                    let handler = FftHandler::<f32>::new(n_ax).normalization(fft_norm_f32(norm));
                    ndfft(&arr.view(), &mut out.view_mut(), &handler, axis_n);
                    scale_axis_complex64(&mut out, axis_n, fft_forward_scale_f32(norm, n_ax));
                    out
                };
                NDArrayWrapper {
                    data: ArrayData::Complex64(Arc::new(RwLock::new(out))),
                    dtype: DType::Complex64,
                }
            }
            DType::Float64
            | DType::Float32
            | DType::Int8
            | DType::Int16
            | DType::Int32
            | DType::Int64
            | DType::Uint8
            | DType::Uint16
            | DType::Uint32
            | DType::Uint64 => {
                if matches!(wrapper.dtype, DType::Float32) {
                    let arr = if let Some(v) = extract_view_as_f32(wrapper, meta_ref) {
                        v
                    } else {
                        error::set_last_error("fft: failed to read Float32 input".to_string());
                        return ERR_GENERIC;
                    };
                    let arr = resize_along_axis_f32(&arr, axis_n, target_n);
                    let c: ArrayD<C64> = arr.mapv(|x| Complex::new(x, 0.0));
                    let n_ax = c.shape()[axis_n];
                    let mut out = ArrayD::<C64>::zeros(c.raw_dim());
                    let handler = FftHandler::<f32>::new(n_ax).normalization(fft_norm_f32(norm));
                    ndfft(&c.view(), &mut out.view_mut(), &handler, axis_n);
                    scale_axis_complex64(&mut out, axis_n, fft_forward_scale_f32(norm, n_ax));
                    NDArrayWrapper {
                        data: ArrayData::Complex64(Arc::new(RwLock::new(out))),
                        dtype: DType::Complex64,
                    }
                } else {
                    let arr = if let Some(v) = extract_view_as_f64(wrapper, meta_ref) {
                        v
                    } else {
                        error::set_last_error(
                            "fft: failed to read real input as float64".to_string(),
                        );
                        return ERR_GENERIC;
                    };
                    let arr = resize_along_axis_f64(&arr, axis_n, target_n);
                    let c: ArrayD<C128> = arr.mapv(|x| Complex::new(x, 0.0));
                    let n_ax = c.shape()[axis_n];
                    let mut out = ArrayD::<C128>::zeros(c.raw_dim());
                    let handler = FftHandler::<f64>::new(n_ax).normalization(fft_norm_f64(norm));
                    ndfft(&c.view(), &mut out.view_mut(), &handler, axis_n);
                    scale_axis_complex128(&mut out, axis_n, fft_forward_scale_f64(norm, n_ax));
                    NDArrayWrapper {
                        data: ArrayData::Complex128(Arc::new(RwLock::new(out))),
                        dtype: DType::Complex128,
                    }
                }
            }
            DType::Bool => {
                error::set_last_error("fft does not support bool dtype".to_string());
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

/// Inverse complex FFT along `axis`.
#[no_mangle]
pub unsafe extern "C" fn ndarray_ifft(
    handle: *const NdArrayHandle,
    meta: *const ArrayMetadata,
    axis: i32,
    n: usize,
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
            error::set_last_error("ifft requires at least one dimension".to_string());
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
            DType::Complex128 => {
                let arr = if let Some(v) = extract_view_c128(wrapper, meta_ref) {
                    v.to_owned()
                } else if let Some(v) = extract_view_as_c128(wrapper, meta_ref) {
                    v
                } else {
                    error::set_last_error("ifft: failed to read Complex128 input".to_string());
                    return ERR_GENERIC;
                };
                let arr = resize_along_axis_complex128(&arr, axis_n, target_n);
                let n_ax = arr.shape()[axis_n];
                let mut out = ArrayD::<C128>::zeros(arr.raw_dim());
                let handler = FftHandler::<f64>::new(n_ax).normalization(fft_norm_f64(norm));
                ndifft(&arr.view(), &mut out.view_mut(), &handler, axis_n);
                scale_axis_complex128(&mut out, axis_n, fft_inverse_scale_f64(norm, n_ax));
                NDArrayWrapper {
                    data: ArrayData::Complex128(Arc::new(RwLock::new(out))),
                    dtype: DType::Complex128,
                }
            }
            DType::Complex64 => {
                let arr = if let Some(v) = extract_view_c64(wrapper, meta_ref) {
                    v.to_owned()
                } else if let Some(v) = extract_view_as_c64(wrapper, meta_ref) {
                    v
                } else {
                    error::set_last_error("ifft: failed to read Complex64 input".to_string());
                    return ERR_GENERIC;
                };
                let arr = resize_along_axis_complex64(&arr, axis_n, target_n);
                let n_ax = arr.shape()[axis_n];
                let mut out = ArrayD::<C64>::zeros(arr.raw_dim());
                let handler = FftHandler::<f32>::new(n_ax).normalization(fft_norm_f32(norm));
                ndifft(&arr.view(), &mut out.view_mut(), &handler, axis_n);
                scale_axis_complex64(&mut out, axis_n, fft_inverse_scale_f32(norm, n_ax));
                NDArrayWrapper {
                    data: ArrayData::Complex64(Arc::new(RwLock::new(out))),
                    dtype: DType::Complex64,
                }
            }
            _ => {
                error::set_last_error("ifft expects a complex dtype".to_string());
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

/// N-dimensional complex FFT over `axes` (empty means all axes).
#[no_mangle]
pub unsafe extern "C" fn ndarray_fftn(
    handle: *const NdArrayHandle,
    meta: *const ArrayMetadata,
    axes: *const i32,
    n_axes: usize,
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
            error::set_last_error("fftn requires at least one dimension".to_string());
            return ERR_SHAPE;
        }

        let axes_v = match resolve_fftn_axes(&shape, axes, n_axes) {
            Ok(a) => a,
            Err(e) => {
                error::set_last_error(e);
                return ERR_SHAPE;
            }
        };

        let result_wrapper = match wrapper.dtype {
            DType::Complex128 => {
                let arr = if let Some(v) = extract_view_c128(wrapper, meta_ref) {
                    v.to_owned()
                } else if let Some(v) = extract_view_as_c128(wrapper, meta_ref) {
                    v
                } else {
                    error::set_last_error("fftn: failed to read Complex128 input".to_string());
                    return ERR_GENERIC;
                };
                let out = fftn_c2c_f64(arr, &axes_v, norm);
                NDArrayWrapper {
                    data: ArrayData::Complex128(Arc::new(RwLock::new(out))),
                    dtype: DType::Complex128,
                }
            }
            DType::Complex64 => {
                let arr = if let Some(v) = extract_view_c64(wrapper, meta_ref) {
                    v.to_owned()
                } else if let Some(v) = extract_view_as_c64(wrapper, meta_ref) {
                    v
                } else {
                    error::set_last_error("fftn: failed to read Complex64 input".to_string());
                    return ERR_GENERIC;
                };
                let out = fftn_c2c_f32(arr, &axes_v, norm);
                NDArrayWrapper {
                    data: ArrayData::Complex64(Arc::new(RwLock::new(out))),
                    dtype: DType::Complex64,
                }
            }
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
                    error::set_last_error("fftn: failed to read real input".to_string());
                    return ERR_GENERIC;
                };
                let c: ArrayD<C128> = arr.mapv(|x| Complex::new(x, 0.0));
                let out = fftn_c2c_f64(c, &axes_v, norm);
                NDArrayWrapper {
                    data: ArrayData::Complex128(Arc::new(RwLock::new(out))),
                    dtype: DType::Complex128,
                }
            }
            DType::Float32 => {
                let arr = if let Some(v) = extract_view_as_f32(wrapper, meta_ref) {
                    v
                } else {
                    error::set_last_error("fftn: failed to read Float32 input".to_string());
                    return ERR_GENERIC;
                };
                let c: ArrayD<C64> = arr.mapv(|x| Complex::new(x, 0.0));
                let out = fftn_c2c_f32(c, &axes_v, norm);
                NDArrayWrapper {
                    data: ArrayData::Complex64(Arc::new(RwLock::new(out))),
                    dtype: DType::Complex64,
                }
            }
            DType::Bool => {
                error::set_last_error("fftn does not support bool dtype".to_string());
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

/// N-dimensional inverse complex FFT over `axes` (empty means all axes).
#[no_mangle]
pub unsafe extern "C" fn ndarray_ifftn(
    handle: *const NdArrayHandle,
    meta: *const ArrayMetadata,
    axes: *const i32,
    n_axes: usize,
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
            error::set_last_error("ifftn requires at least one dimension".to_string());
            return ERR_SHAPE;
        }

        let axes_v = match resolve_fftn_axes(&shape, axes, n_axes) {
            Ok(a) => a,
            Err(e) => {
                error::set_last_error(e);
                return ERR_SHAPE;
            }
        };

        let result_wrapper = match wrapper.dtype {
            DType::Complex128 => {
                let arr = if let Some(v) = extract_view_c128(wrapper, meta_ref) {
                    v.to_owned()
                } else if let Some(v) = extract_view_as_c128(wrapper, meta_ref) {
                    v
                } else {
                    error::set_last_error("ifftn: failed to read Complex128 input".to_string());
                    return ERR_GENERIC;
                };
                let out = ifftn_c2c_f64(arr, &axes_v, norm);
                NDArrayWrapper {
                    data: ArrayData::Complex128(Arc::new(RwLock::new(out))),
                    dtype: DType::Complex128,
                }
            }
            DType::Complex64 => {
                let arr = if let Some(v) = extract_view_c64(wrapper, meta_ref) {
                    v.to_owned()
                } else if let Some(v) = extract_view_as_c64(wrapper, meta_ref) {
                    v
                } else {
                    error::set_last_error("ifftn: failed to read Complex64 input".to_string());
                    return ERR_GENERIC;
                };
                let out = ifftn_c2c_f32(arr, &axes_v, norm);
                NDArrayWrapper {
                    data: ArrayData::Complex64(Arc::new(RwLock::new(out))),
                    dtype: DType::Complex64,
                }
            }
            _ => {
                error::set_last_error("ifftn expects a complex dtype".to_string());
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
