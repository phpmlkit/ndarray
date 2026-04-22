//! Bartlett (triangular) window.

use ndarray::{ArrayD, IxDyn};
use parking_lot::RwLock;
use std::sync::Arc;

use crate::helpers::error::{set_last_error, ERR_GENERIC, SUCCESS};
use crate::types::dtype::DType;
use crate::types::{ArrayData, NDArrayWrapper, NdArrayHandle};

fn bartlett_symmetric(m: usize) -> Vec<f64> {
    if m == 0 {
        return vec![];
    }
    if m == 1 {
        return vec![1.0];
    }

    let mm1 = (m - 1) as f64;
    let half = mm1 / 2.0;
    let mut w = Vec::with_capacity(m);
    for n in 0..m {
        let x = (n as f64 - half).abs() / half;
        w.push(1.0 - x);
    }
    w
}

fn bartlett_window(m: usize, periodic: bool) -> Vec<f64> {
    if periodic && m > 1 {
        let mut w = bartlett_symmetric(m + 1);
        w.pop();
        w
    } else {
        bartlett_symmetric(m)
    }
}

/// Generate a Bartlett window (Float64).
#[no_mangle]
pub unsafe extern "C" fn ndarray_bartlett(
    m: usize,
    periodic: bool,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if out_handle.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let data = bartlett_window(m, periodic);
        let arr: ArrayD<f64> = match ArrayD::from_shape_vec(IxDyn(&[m]), data) {
            Ok(a) => a,
            Err(e) => {
                set_last_error(format!("bartlett(): shape error: {}", e));
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

