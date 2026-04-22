//! Boxcar (rectangular) window.

use ndarray::{ArrayD, IxDyn};
use parking_lot::RwLock;
use std::sync::Arc;

use crate::helpers::error::{set_last_error, ERR_GENERIC, SUCCESS};
use crate::types::dtype::DType;
use crate::types::{ArrayData, NDArrayWrapper, NdArrayHandle};

fn boxcar_symmetric(m: usize) -> Vec<f64> {
    match m {
        0 => vec![],
        _ => vec![1.0_f64; m],
    }
}

fn boxcar_window(m: usize, periodic: bool) -> Vec<f64> {
    if periodic && m > 1 {
        let mut w = boxcar_symmetric(m + 1);
        w.pop();
        w
    } else {
        boxcar_symmetric(m)
    }
}

/// Generate a boxcar (rectangular) window (Float64).
#[no_mangle]
pub unsafe extern "C" fn ndarray_boxcar(
    m: usize,
    periodic: bool,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if out_handle.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let data = boxcar_window(m, periodic);
        let arr: ArrayD<f64> = match ArrayD::from_shape_vec(IxDyn(&[m]), data) {
            Ok(a) => a,
            Err(e) => {
                set_last_error(format!("boxcar(): shape error: {}", e));
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

