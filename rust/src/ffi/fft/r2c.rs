//! Real FFT (`rfft`) and inverse real FFT (`irfft`).

use std::sync::Arc;

use ndarray::{ArrayD, IxDyn};
use ndrustfft::{ndfft_r2c, ndifft_r2c, Complex, R2cFftHandler};
use parking_lot::RwLock;

use crate::helpers::error::{self, ERR_DTYPE, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::helpers::{
    extract_view_as_c128, extract_view_as_c64, extract_view_as_f32, extract_view_as_f64,
    extract_view_c128, extract_view_c64, fft_forward_scale_f32, fft_forward_scale_f64,
    fft_inverse_scale_f32, fft_inverse_scale_f64, fft_norm_f32, fft_norm_f64, infer_irfft_real_len,
    normalize_axis, resize_along_axis_complex128, resize_along_axis_complex64,
    resize_along_axis_f32, resize_along_axis_f64, scale_axis_complex128, scale_axis_complex64,
    write_output_metadata,
};
use crate::types::dtype::DType;
use crate::types::{ArrayData, ArrayMetadata, NDArrayWrapper, NdArrayHandle};

type C64 = Complex<f32>;
type C128 = Complex<f64>;

/// Real-input FFT along `axis`. Output is complex with length `n//2+1` on that axis.
#[no_mangle]
pub unsafe extern "C" fn ndarray_rfft(
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
            error::set_last_error("rfft requires at least one dimension".to_string());
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
            DType::Float32 => {
                let arr = if let Some(v) = extract_view_as_f32(wrapper, meta_ref) {
                    v
                } else {
                    error::set_last_error("rfft: failed to read Float32 input".to_string());
                    return ERR_GENERIC;
                };
                let arr = resize_along_axis_f32(&arr, axis_n, target_n);
                let n_ax = arr.shape()[axis_n];
                let mut out_shape = arr.shape().to_vec();
                out_shape[axis_n] = n_ax / 2 + 1;
                let mut out = ArrayD::<C64>::zeros(IxDyn(&out_shape));
                let handler = R2cFftHandler::<f32>::new(n_ax).normalization(fft_norm_f32(norm));
                ndfft_r2c(&arr.view(), &mut out.view_mut(), &handler, axis_n);
                scale_axis_complex64(&mut out, axis_n, fft_forward_scale_f32(norm, n_ax));
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
                    error::set_last_error("rfft: failed to read real input".to_string());
                    return ERR_GENERIC;
                };
                let arr = resize_along_axis_f64(&arr, axis_n, target_n);
                let n_ax = arr.shape()[axis_n];
                let mut out_shape = arr.shape().to_vec();
                out_shape[axis_n] = n_ax / 2 + 1;
                let mut out = ArrayD::<C128>::zeros(IxDyn(&out_shape));
                let handler = R2cFftHandler::<f64>::new(n_ax).normalization(fft_norm_f64(norm));
                ndfft_r2c(&arr.view(), &mut out.view_mut(), &handler, axis_n);
                scale_axis_complex128(&mut out, axis_n, fft_forward_scale_f64(norm, n_ax));
                NDArrayWrapper {
                    data: ArrayData::Complex128(Arc::new(RwLock::new(out))),
                    dtype: DType::Complex128,
                }
            }
            DType::Complex64 | DType::Complex128 => {
                error::set_last_error("rfft expects a real (float or integer) dtype".to_string());
                return ERR_DTYPE;
            }
            DType::Bool => {
                error::set_last_error("rfft does not support bool dtype".to_string());
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

/// Inverse real FFT: complex Hermitian spectrum → real array. `n == 0` infers real length.
#[no_mangle]
pub unsafe extern "C" fn ndarray_irfft(
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
            error::set_last_error("irfft requires at least one dimension".to_string());
            return ERR_SHAPE;
        }

        let axis_n = match normalize_axis(&shape, axis, false) {
            Ok(a) => a,
            Err(e) => {
                error::set_last_error(e);
                return ERR_SHAPE;
            }
        };

        let m = shape[axis_n];
        let result_wrapper = match wrapper.dtype {
            DType::Complex128 => {
                let arr = if let Some(v) = extract_view_c128(wrapper, meta_ref) {
                    v.to_owned()
                } else if let Some(v) = extract_view_as_c128(wrapper, meta_ref) {
                    v
                } else {
                    error::set_last_error("irfft: failed to read Complex128 input".to_string());
                    return ERR_GENERIC;
                };
                let n_real = match infer_irfft_real_len(m, n) {
                    Ok(n) => n,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                let expected_m = n_real / 2 + 1;
                let mut arr = resize_along_axis_complex128(&arr, axis_n, expected_m);
                scale_axis_complex128(&mut arr, axis_n, fft_inverse_scale_f64(norm, n_real));
                let mut out_shape = arr.shape().to_vec();
                out_shape[axis_n] = n_real;
                let mut out = ArrayD::<f64>::zeros(IxDyn(&out_shape));
                let handler = R2cFftHandler::<f64>::new(n_real).normalization(fft_norm_f64(norm));
                ndifft_r2c(&arr.view(), &mut out.view_mut(), &handler, axis_n);
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(out))),
                    dtype: DType::Float64,
                }
            }
            DType::Complex64 => {
                let arr = if let Some(v) = extract_view_c64(wrapper, meta_ref) {
                    v.to_owned()
                } else if let Some(v) = extract_view_as_c64(wrapper, meta_ref) {
                    v
                } else {
                    error::set_last_error("irfft: failed to read Complex64 input".to_string());
                    return ERR_GENERIC;
                };
                let n_real = match infer_irfft_real_len(m, n) {
                    Ok(n) => n,
                    Err(e) => {
                        error::set_last_error(e);
                        return ERR_SHAPE;
                    }
                };
                let expected_m = n_real / 2 + 1;
                let mut arr = resize_along_axis_complex64(&arr, axis_n, expected_m);
                scale_axis_complex64(&mut arr, axis_n, fft_inverse_scale_f32(norm, n_real));
                let mut out_shape = arr.shape().to_vec();
                out_shape[axis_n] = n_real;
                let mut out = ArrayD::<f32>::zeros(IxDyn(&out_shape));
                let handler = R2cFftHandler::<f32>::new(n_real).normalization(fft_norm_f32(norm));
                ndifft_r2c(&arr.view(), &mut out.view_mut(), &handler, axis_n);
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(out))),
                    dtype: DType::Float32,
                }
            }
            _ => {
                error::set_last_error("irfft expects a complex dtype".to_string());
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
