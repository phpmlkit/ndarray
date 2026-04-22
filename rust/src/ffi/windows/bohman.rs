//! Bohman window.

use ndarray::{ArrayD, IxDyn};
use parking_lot::RwLock;
use std::f64::consts::PI;
use std::sync::Arc;

use crate::helpers::error::{set_last_error, ERR_GENERIC, SUCCESS};
use crate::types::dtype::DType;
use crate::types::{ArrayData, NDArrayWrapper, NdArrayHandle};

fn bohman_symmetric(m: usize) -> Vec<f64> {
    if m == 0 {
        return vec![];
    }
    if m == 1 {
        return vec![1.0];
    }

    // w(n) = (1 - |x|) cos(pi |x|) + sin(pi |x|)/pi, where x = 2n/(m-1) - 1
    let denom = (m - 1) as f64;
    let mut w = Vec::with_capacity(m);
    for n in 0..m {
        let x = (2.0 * (n as f64) / denom - 1.0).abs();
        let wv = (1.0 - x) * (PI * x).cos() + (PI * x).sin() / PI;
        w.push(wv);
    }
    w
}

fn bohman_window(m: usize, periodic: bool) -> Vec<f64> {
    if periodic && m > 1 {
        let mut w = bohman_symmetric(m + 1);
        w.pop();
        w
    } else {
        bohman_symmetric(m)
    }
}

/// Generate a Bohman window (Float64).
#[no_mangle]
pub unsafe extern "C" fn ndarray_bohman(
    m: usize,
    periodic: bool,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if out_handle.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let data = bohman_window(m, periodic);
        let arr: ArrayD<f64> = match ArrayD::from_shape_vec(IxDyn(&[m]), data) {
            Ok(a) => a,
            Err(e) => {
                set_last_error(format!("bohman(): shape error: {}", e));
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

