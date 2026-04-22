//! Resize arrays along one axis for FFT/DCT padding, truncation, and normalization helpers.

use std::f64::consts::PI;

use ndarray::{indices, Array1, ArrayD, Axis, IxDyn};
use ndrustfft::Normalization;
use num_complex::Complex;

/// FFT norm: `0` = backward (default), `1` = ortho, `2` = forward, `3` = none (unnormalized).
/// Always use `Normalization::None` in `ndrustfft` and apply SciPy scaling explicitly.
#[inline]
pub fn fft_norm_f64(_code: u8) -> Normalization<Complex<f64>> {
    Normalization::None
}

#[inline]
pub fn fft_norm_f32(_code: u8) -> Normalization<Complex<f32>> {
    Normalization::None
}

/// Scale applied along an axis after `ndfft` (forward transform).
#[inline]
pub fn fft_forward_scale_f64(norm: u8, n: usize) -> f64 {
    match norm {
        0 => 1.0_f64,
        1 => 1.0 / (n as f64).sqrt(),
        2 => 1.0 / n as f64,
        _ => 1.0_f64,
    }
}

/// Scale applied along an axis after `ndifft` (inverse transform).
#[inline]
pub fn fft_inverse_scale_f64(norm: u8, n: usize) -> f64 {
    match norm {
        0 => 1.0 / n as f64,
        1 => 1.0 / (n as f64).sqrt(),
        2 => 1.0_f64,
        _ => 1.0_f64,
    }
}

#[inline]
pub fn fft_forward_scale_f32(norm: u8, n: usize) -> f32 {
    fft_forward_scale_f64(norm, n) as f32
}

#[inline]
pub fn fft_inverse_scale_f32(norm: u8, n: usize) -> f32 {
    fft_inverse_scale_f64(norm, n) as f32
}

pub fn scale_axis_complex64(arr: &mut ArrayD<Complex<f32>>, axis: usize, factor: f32) {
    if factor == 1.0 {
        return;
    }
    for mut lane in arr.lanes_mut(Axis(axis)) {
        if let Some(s) = lane.as_slice_mut() {
            for c in s {
                c.re *= factor;
                c.im *= factor;
            }
        } else {
            for c in lane.iter_mut() {
                c.re *= factor;
                c.im *= factor;
            }
        }
    }
}

pub fn scale_axis_complex128(arr: &mut ArrayD<Complex<f64>>, axis: usize, factor: f64) {
    if factor == 1.0 {
        return;
    }
    for mut lane in arr.lanes_mut(Axis(axis)) {
        if let Some(s) = lane.as_slice_mut() {
            for c in s {
                c.re *= factor;
                c.im *= factor;
            }
        } else {
            for c in lane.iter_mut() {
                c.re *= factor;
                c.im *= factor;
            }
        }
    }
}

/// Pad or truncate along `axis` to `new_len` (zeros on pad).
pub fn resize_along_axis_complex64(
    src: &ArrayD<Complex<f32>>,
    axis: usize,
    new_len: usize,
) -> ArrayD<Complex<f32>> {
    let mut new_shape = src.shape().to_vec();
    let old_len = new_shape[axis];
    if old_len == new_len {
        return src.clone();
    }
    new_shape[axis] = new_len;
    let mut dst = ArrayD::<Complex<f32>>::zeros(IxDyn(&new_shape));
    for idx in indices(dst.raw_dim()) {
        let ax = idx[axis];
        if ax < old_len {
            let i = idx.clone();
            dst[i.clone()] = src[i.clone()];
        } else {
            dst[idx] = Complex::new(0.0, 0.0);
        }
    }
    dst
}

/// Pad or truncate along `axis` to `new_len` (zeros on pad).
pub fn resize_along_axis_complex128(
    src: &ArrayD<Complex<f64>>,
    axis: usize,
    new_len: usize,
) -> ArrayD<Complex<f64>> {
    let mut new_shape = src.shape().to_vec();
    let old_len = new_shape[axis];
    if old_len == new_len {
        return src.clone();
    }
    new_shape[axis] = new_len;
    let mut dst = ArrayD::<Complex<f64>>::zeros(IxDyn(&new_shape));
    for idx in indices(dst.raw_dim()) {
        let ax = idx[axis];
        if ax < old_len {
            let i = idx.clone();
            dst[i.clone()] = src[i.clone()];
        } else {
            dst[idx] = Complex::new(0.0, 0.0);
        }
    }
    dst
}

/// Pad or truncate real `f32` array along `axis`.
pub fn resize_along_axis_f32(src: &ArrayD<f32>, axis: usize, new_len: usize) -> ArrayD<f32> {
    let mut new_shape = src.shape().to_vec();
    let old_len = new_shape[axis];
    if old_len == new_len {
        return src.clone();
    }
    new_shape[axis] = new_len;
    let mut dst = ArrayD::<f32>::zeros(IxDyn(&new_shape));
    for idx in indices(dst.raw_dim()) {
        let ax = idx[axis];
        if ax < old_len {
            let i = idx.clone();
            dst[i.clone()] = src[i.clone()];
        } else {
            dst[idx] = 0.0;
        }
    }
    dst
}

/// Pad or truncate real `f64` array along `axis`.
pub fn resize_along_axis_f64(src: &ArrayD<f64>, axis: usize, new_len: usize) -> ArrayD<f64> {
    let mut new_shape = src.shape().to_vec();
    let old_len = new_shape[axis];
    if old_len == new_len {
        return src.clone();
    }
    new_shape[axis] = new_len;
    let mut dst = ArrayD::<f64>::zeros(IxDyn(&new_shape));
    for idx in indices(dst.raw_dim()) {
        let ax = idx[axis];
        if ax < old_len {
            let i = idx.clone();
            dst[i.clone()] = src[i.clone()];
        } else {
            dst[idx] = 0.0;
        }
    }
    dst
}

/// Infer real output length for `irfft` when `n` is not specified.
/// Matches common `n = 2*(m-1)` for `m > 1`, with `m == 1` → `n = 2`.
pub fn infer_irfft_real_len(m: usize, n_param: usize) -> Result<usize, String> {
    if n_param > 0 {
        return Ok(n_param);
    }
    if m == 0 {
        return Err("irfft: zero-sized frequency axis".to_string());
    }
    if m == 1 {
        Ok(2)
    } else {
        Ok(2 * (m - 1))
    }
}

// -------------------------------------------------------------------------------------------------
// DCT / IDCT normalization (SciPy-style)
//
// Norm code: `0` = backward (default), `1` = ortho, `2` = forward, `3` = none (raw `rustdct`)
// -------------------------------------------------------------------------------------------------

#[inline]
pub fn dct_ndrust_norm_is_default(dct_type: u8, forward: bool) -> bool {
    match dct_type {
        1 => false,
        _ => forward,
    }
}

/// After `nddct1` with `Normalization::None`, scale for `norm='backward'` inverse DCT-I.
///
/// `rustdct` DCT-I is scaled differently from SciPy’s reference sum; this factor matches
/// `idct(dct(x), type=1, norm='backward') ≈ x` against SciPy.
#[inline]
pub fn dct1_backward_post_inverse_f64(n: usize) -> f64 {
    if n <= 1 {
        1.0
    } else {
        2.0 / (n as f64 - 1.0)
    }
}

/// Types II–IV: inverse scaling after raw inverse kernel (DCT-III / II / IV).
#[inline]
pub fn dct234_backward_post_inverse_f64(n: usize) -> f64 {
    1.0 / n as f64
}

/// Factor `c` such that `dct_forward(x) = c * dct_backward(x)` (pointwise).
#[inline]
pub fn dct_forward_scale_from_backward_c(dct_type: u8, n: usize) -> f64 {
    match dct_type {
        1 => {
            if n <= 1 {
                1.0
            } else {
                1.0 / (2.0 * (n as f64 - 1.0))
            }
        }
        _ => 1.0 / (2.0 * n as f64),
    }
}

/// `idct(..., norm='forward')` equals `idct(..., norm='backward') / c` with the same `c`.
#[inline]
pub fn dct_inverse_forward_scale_from_backward_f64(dct_type: u8, n: usize) -> f64 {
    1.0 / dct_forward_scale_from_backward_c(dct_type, n)
}

/// `norm='ortho'` DCT-IV from backward spectrum: multiply by `sqrt(2/N)/2`.
#[inline]
pub fn dct4_ortho_scale_from_backward_f64(n: usize) -> f64 {
    (2.0_f64 / n as f64).sqrt() / 2.0_f64
}

// --- Ortho DCT-II / III ------------------------------------------------------------------------

pub fn dct2_ortho_lane_f64(x: &[f64], out: &mut [f64]) {
    let n = x.len();
    assert_eq!(out.len(), n);
    let sn = (2.0_f64 / n as f64).sqrt();
    for k in 0..n {
        let mut s = 0.0_f64;
        for (i, &xi) in x.iter().enumerate() {
            s += xi * (PI * k as f64 * (i as f64 + 0.5) / n as f64).cos();
        }
        let ck = if k == 0 { 1.0 / 2.0_f64.sqrt() } else { 1.0 };
        out[k] = sn * ck * s;
    }
}

pub fn dct2_ortho_lane_inverse_f64(y: &[f64], out: &mut [f64]) {
    let n = y.len();
    assert_eq!(out.len(), n);
    let sn = (2.0_f64 / n as f64).sqrt();
    for j in 0..n {
        let mut s = 0.0_f64;
        for k in 0..n {
            let ck = if k == 0 { 1.0 / 2.0_f64.sqrt() } else { 1.0 };
            s += sn * ck * y[k] * (PI * k as f64 * (j as f64 + 0.5) / n as f64).cos();
        }
        out[j] = s;
    }
}

#[inline]
pub fn dct3_ortho_lane_f64(x: &[f64], out: &mut [f64]) {
    dct2_ortho_lane_inverse_f64(x, out);
}

#[inline]
pub fn dct3_ortho_lane_inverse_f64(y: &[f64], out: &mut [f64]) {
    dct2_ortho_lane_f64(y, out);
}

// --- Ortho DCT-I -------------------------------------------------------------------------------

pub fn dct1_ortho_lane_f64(x: &[f64], out: &mut [f64]) {
    let n = x.len();
    assert_eq!(out.len(), n);
    if n == 1 {
        out[0] = x[0];
        return;
    }
    let d = n as f64 - 1.0;
    let s = (2.0_f64 / d).sqrt();
    for k in 0..n {
        let ak = if k == 0 || k == n - 1 {
            1.0 / 2.0_f64.sqrt()
        } else {
            1.0
        };
        let mut acc = 0.0_f64;
        for j in 0..n {
            let aj = if j == 0 || j == n - 1 {
                1.0 / 2.0_f64.sqrt()
            } else {
                1.0
            };
            acc += s * ak * aj * x[j] * (PI * k as f64 * j as f64 / d).cos();
        }
        out[k] = acc;
    }
}

pub fn dct1_ortho_lane_inverse_f64(y: &[f64], out: &mut [f64]) {
    let n = y.len();
    assert_eq!(out.len(), n);
    if n == 1 {
        out[0] = y[0];
        return;
    }
    let d = n as f64 - 1.0;
    let s = (2.0_f64 / d).sqrt();
    for j in 0..n {
        let aj = if j == 0 || j == n - 1 {
            1.0 / 2.0_f64.sqrt()
        } else {
            1.0
        };
        let mut acc = 0.0_f64;
        for k in 0..n {
            let ak = if k == 0 || k == n - 1 {
                1.0 / 2.0_f64.sqrt()
            } else {
                1.0
            };
            acc += s * ak * aj * y[k] * (PI * k as f64 * j as f64 / d).cos();
        }
        out[j] = acc;
    }
}

// --- Axis helpers for real arrays --------------------------------------------------------------

pub fn scale_axis_f64(arr: &mut ArrayD<f64>, axis: usize, factor: f64) {
    if factor == 1.0 {
        return;
    }
    for mut lane in arr.lanes_mut(Axis(axis)) {
        if let Some(s) = lane.as_slice_mut() {
            for v in s {
                *v *= factor;
            }
        } else {
            for v in lane.iter_mut() {
                *v *= factor;
            }
        }
    }
}

pub fn scale_axis_f32(arr: &mut ArrayD<f32>, axis: usize, factor: f32) {
    if factor == 1.0 {
        return;
    }
    for mut lane in arr.lanes_mut(Axis(axis)) {
        if let Some(s) = lane.as_slice_mut() {
            for v in s {
                *v *= factor;
            }
        } else {
            for v in lane.iter_mut() {
                *v *= factor;
            }
        }
    }
}

/// Apply `f` to each contiguous lane along `axis` (read from `src`, write to `dst`).
pub fn map_axis_lane_f64<F>(src: &ArrayD<f64>, dst: &mut ArrayD<f64>, axis: usize, mut f: F)
where
    F: FnMut(&[f64], &mut [f64]),
{
    assert_eq!(src.shape(), dst.shape());
    for (sl, mut dl) in src
        .lanes(Axis(axis))
        .into_iter()
        .zip(dst.lanes_mut(Axis(axis)))
    {
        if let (Some(s), Some(d)) = (sl.as_slice(), dl.as_slice_mut()) {
            f(s, d);
        } else {
            let tmp_s = sl.to_vec();
            let mut tmp_d = vec![0.0_f64; tmp_s.len()];
            f(&tmp_s, &mut tmp_d);
            dl.assign(&Array1::from_vec(tmp_d));
        }
    }
}

pub fn map_axis_lane_f32<F>(src: &ArrayD<f32>, dst: &mut ArrayD<f32>, axis: usize, mut f: F)
where
    F: FnMut(&[f32], &mut [f32]),
{
    assert_eq!(src.shape(), dst.shape());
    for (sl, mut dl) in src
        .lanes(Axis(axis))
        .into_iter()
        .zip(dst.lanes_mut(Axis(axis)))
    {
        if let (Some(s), Some(d)) = (sl.as_slice(), dl.as_slice_mut()) {
            f(s, d);
        } else {
            let tmp_s = sl.to_vec();
            let mut tmp_d = vec![0.0_f32; tmp_s.len()];
            f(&tmp_s, &mut tmp_d);
            dl.assign(&Array1::from_vec(tmp_d));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dct1_ortho_matches_transpose_inverse() {
        let n = 4_usize;
        let x = vec![1.0_f64, 2.0, 3.0, 4.0];
        let mut y = vec![0.0_f64; n];
        let mut z = vec![0.0_f64; n];
        dct1_ortho_lane_f64(&x, &mut y);
        dct1_ortho_lane_inverse_f64(&y, &mut z);
        for i in 0..n {
            assert!((z[i] - x[i]).abs() < 1e-10, "i={} z={} x={}", i, z[i], x[i]);
        }
    }
}
