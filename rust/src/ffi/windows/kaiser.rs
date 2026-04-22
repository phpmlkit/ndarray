//! Kaiser window.
//!
//! Uses an approximation for the modified Bessel function I0.

use ndarray::{ArrayD, IxDyn};
use parking_lot::RwLock;
use std::sync::Arc;

use crate::helpers::error::{set_last_error, ERR_GENERIC, SUCCESS};
use crate::types::dtype::DType;
use crate::types::{ArrayData, NDArrayWrapper, NdArrayHandle};

#[inline]
fn i0(x: f64) -> f64 {
    // Cephes-like approximation (commonly used in numerical libraries).
    let ax = x.abs();
    if ax < 3.75 {
        let y = (x / 3.75).powi(2);
        1.0
            + y * (3.5156229
                + y * (3.0899424
                    + y * (1.2067492
                        + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))))
    } else {
        let y = 3.75 / ax;
        let ans = 0.39894228
            + y * (0.01328592
                + y * (0.00225319
                    + y * (-0.00157565
                        + y * (0.00916281
                            + y * (-0.02057706
                                + y * (0.02635537
                                    + y * (-0.01647633 + y * 0.00392377)))))));
        (ax.exp() / ax.sqrt()) * ans
    }
}

fn kaiser_symmetric(m: usize, beta: f64) -> Vec<f64> {
    if m == 0 {
        return vec![];
    }
    if m == 1 {
        return vec![1.0];
    }

    let denom = (m - 1) as f64;
    let inv_i0_beta = 1.0 / i0(beta);
    let mut w = Vec::with_capacity(m);
    for n in 0..m {
        let r = 2.0 * (n as f64) / denom - 1.0; // in [-1, 1]
        let t = (1.0 - r * r).max(0.0).sqrt();
        w.push(i0(beta * t) * inv_i0_beta);
    }
    w
}

fn kaiser_window(m: usize, beta: f64, periodic: bool) -> Vec<f64> {
    if periodic && m > 1 {
        let mut w = kaiser_symmetric(m + 1, beta);
        w.pop();
        w
    } else {
        kaiser_symmetric(m, beta)
    }
}

/// Generate a Kaiser window (Float64).
#[no_mangle]
pub unsafe extern "C" fn ndarray_kaiser(
    m: usize,
    beta: f64,
    periodic: bool,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if out_handle.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        if !beta.is_finite() {
            set_last_error("kaiser(): beta must be finite".to_string());
            return ERR_GENERIC;
        }

        let data = kaiser_window(m, beta, periodic);
        let arr: ArrayD<f64> = match ArrayD::from_shape_vec(IxDyn(&[m]), data) {
            Ok(a) => a,
            Err(e) => {
                set_last_error(format!("kaiser(): shape error: {}", e));
                return ERR_GENERIC;
            }
        };

        let wrapper = NDArrayWrapper {
            data: ArrayData::Float64(Arc::new(RwLock::new(arr))),
            dtype: DType::Float64,
        };

        *out_handle = NdArrayHandle::from_wrapper(Box::new(wrapper));
        SUCCESS
    })
}

