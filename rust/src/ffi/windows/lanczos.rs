//! Lanczos window.

use ndarray::{ArrayD, IxDyn};
use parking_lot::RwLock;
use std::f64::consts::PI;
use std::sync::Arc;

use crate::helpers::error::{set_last_error, ERR_GENERIC, SUCCESS};
use crate::types::dtype::DType;
use crate::types::{ArrayData, NDArrayWrapper, NdArrayHandle};

#[inline]
fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let pix = PI * x;
        pix.sin() / pix
    }
}

fn lanczos_symmetric(m: usize) -> Vec<f64> {
    if m == 0 {
        return vec![];
    }
    if m == 1 {
        return vec![1.0];
    }

    // w(n) = sinc(2n/(m-1) - 1), n=0..m-1
    let denom = (m - 1) as f64;
    let mut w = Vec::with_capacity(m);
    for n in 0..m {
        let x = 2.0 * (n as f64) / denom - 1.0;
        w.push(sinc(x));
    }
    w
}

fn lanczos_window(m: usize, periodic: bool) -> Vec<f64> {
    if periodic && m > 1 {
        let mut w = lanczos_symmetric(m + 1);
        w.pop();
        w
    } else {
        lanczos_symmetric(m)
    }
}

/// Generate a Lanczos window (Float64).
#[no_mangle]
pub unsafe extern "C" fn ndarray_lanczos(
    m: usize,
    periodic: bool,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if out_handle.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let data = lanczos_window(m, periodic);
        let arr: ArrayD<f64> = match ArrayD::from_shape_vec(IxDyn(&[m]), data) {
            Ok(a) => a,
            Err(e) => {
                set_last_error(format!("lanczos(): shape error: {}", e));
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

