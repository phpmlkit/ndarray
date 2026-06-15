//! Einsum module: Einstein summation with deterministic accumulation.

pub mod kernels;
pub mod parser;

use std::ffi::CStr;
use std::ops::{Add, Mul};
use std::sync::Arc;

use ndarray::{ArrayD, IxDyn, ScalarOperand};
use num_traits::Zero;
use parking_lot::RwLock;

use crate::helpers::error::{set_last_error, ERR_DTYPE, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::helpers::write_output_metadata;
use crate::helpers::{extract_array_f32, extract_array_f64};
use crate::types::dtype::DType;
use crate::types::{ArrayData, ArrayMetadata, NDArrayWrapper, NdArrayHandle};

use self::parser::EinsumSpec;

fn dispatch<T>(a: &ArrayD<T>, b: &ArrayD<T>, spec: &EinsumSpec) -> Result<ArrayD<T>, String>
where
    T: Copy + ScalarOperand + Zero + Mul<Output = T> + Add<Output = T>,
{
    if spec.is_elementwise {
        return Ok(kernels::hadamard(a, b));
    }
    if spec.contracted_axes.len() == 1 {
        let (la, ra) = spec.contracted_axes[0];
        if spec.out_shape.len() == 2 && a.ndim() == 2 && b.ndim() == 2 && la == 1 && ra == 0 {
            return Ok(kernels::gemm(a, b));
        }
        if spec.out_shape.len() == 2 && a.ndim() == 2 && b.ndim() == 2 && la == 1 && ra == 1 {
            return Ok(kernels::gemm_transposed(a, b));
        }
        if spec.out_shape.is_empty() && a.ndim() == 1 && b.ndim() == 1 {
            let s = kernels::dot(a, b);
            return Ok(ArrayD::from_shape_vec(IxDyn(&[]), vec![s]).unwrap());
        }
    }
    if spec.contracted_axes.is_empty()
        && a.ndim() == 1
        && b.ndim() == 1
        && spec.out_shape.len() == 2
    {
        return Ok(kernels::outer(a, b));
    }
    generic_contract(a, b, spec)
}

fn dispatch_single<T>(a: &ArrayD<T>, spec: &EinsumSpec) -> Result<ArrayD<T>, String>
where
    T: Copy + ScalarOperand + Zero,
{
    if let Some(ref perm) = spec.permute_order {
        return Ok(kernels::transpose(a, perm));
    }
    if !spec.reduce_axes.is_empty() {
        if spec.reduce_axes.len() == 1 {
            return Ok(kernels::sum_over_axis(a, spec.reduce_axes[0]));
        }
        if spec.reduce_axes.len() == 2 && spec.reduce_axes[0] == spec.reduce_axes[1] {
            let s = kernels::trace(a);
            return Ok(ArrayD::from_shape_vec(IxDyn(&[]), vec![s]).unwrap());
        }
        let s = kernels::sum_all(a);
        return Ok(ArrayD::from_shape_vec(IxDyn(&[]), vec![s]).unwrap());
    }
    if spec.out_shape.len() == 1
        && spec.lhs_shape.len() == 2
        && spec.lhs_shape[0] == spec.lhs_shape[1]
    {
        return Ok(kernels::diagonal(a));
    }
    Err(format!(
        "unsupported: shape={:?} out={:?}",
        spec.lhs_shape, spec.out_shape
    ))
}

fn generic_contract<T>(a: &ArrayD<T>, b: &ArrayD<T>, spec: &EinsumSpec) -> Result<ArrayD<T>, String>
where
    T: Copy + Zero + Mul<Output = T> + Add<Output = T>,
{
    let total: usize = spec.out_shape.iter().product();
    let mut data = vec![T::zero(); total];
    for out_idx in 0..total {
        let mut rem = out_idx;
        let mut oc = vec![0usize; spec.out_shape.len()];
        for dim in (0..spec.out_shape.len()).rev() {
            oc[dim] = rem % spec.out_shape[dim];
            rem /= spec.out_shape[dim];
        }
        let mut ac = vec![0usize; a.ndim()];
        let mut bc = vec![0usize; b.ndim()];
        for (op, (src, ax)) in spec.output_map.iter().enumerate() {
            if *src {
                ac[*ax] = oc[op];
            } else {
                bc[*ax] = oc[op];
            }
        }
        if spec.contracted_axes.is_empty() {
            data[out_idx] = a.as_slice().unwrap()[flat_index(&ac, a.shape())]
                * b.as_slice().unwrap()[flat_index(&bc, b.shape())];
        } else {
            let (la, ra) = spec.contracted_axes[0];
            let cs = spec.lhs_shape[la];
            let mut sum = T::zero();
            for k in 0..cs {
                let mut ac2 = ac.clone();
                let mut bc2 = bc.clone();
                ac2[la] = k;
                bc2[ra] = k;
                sum = sum
                    + a.as_slice().unwrap()[flat_index(&ac2, a.shape())]
                        * b.as_slice().unwrap()[flat_index(&bc2, b.shape())];
            }
            data[out_idx] = sum;
        }
    }
    Ok(ArrayD::from_shape_vec(IxDyn(&spec.out_shape), data).unwrap())
}

fn flat_index(coords: &[usize], shape: &[usize]) -> usize {
    let mut idx = 0;
    for (i, &c) in coords.iter().enumerate().rev() {
        idx = idx * shape[i] + c;
    }
    idx
}

/// FFI entry point for einsum (1 or 2 operands, null b for single-op).
#[no_mangle]
pub unsafe extern "C" fn ndarray_einsum(
    a: *const NdArrayHandle,
    a_meta: *const ArrayMetadata,
    b: *const NdArrayHandle,
    b_meta: *const ArrayMetadata,
    subscripts: *const std::os::raw::c_char,
    out_handle: *mut *mut NdArrayHandle,
    out_dtype: *mut u8,
    out_ndim: *mut usize,
    out_shape_ptr: *mut usize,
    max_ndim: usize,
) -> i32 {
    if a.is_null()
        || subscripts.is_null()
        || out_handle.is_null()
        || out_dtype.is_null()
        || out_ndim.is_null()
        || out_shape_ptr.is_null()
        || a_meta.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let a_meta = &*a_meta;
        let a_shape = a_meta.shape_slice();
        let a_wrapper = NdArrayHandle::as_wrapper(a as *mut _);
        let has_b = !b.is_null() && !b_meta.is_null();
        let b_shape = if has_b {
            Some((&*b_meta).shape_slice())
        } else {
            None
        };

        let s = match CStr::from_ptr(subscripts as *const i8).to_str() {
            Ok(s) => s,
            Err(e) => {
                set_last_error(format!("Invalid subscripts: {}", e));
                return ERR_GENERIC;
            }
        };
        let spec = match parser::parse(s, a_shape, b_shape) {
            Ok(s) => s,
            Err(e) => {
                set_last_error(e);
                return ERR_SHAPE;
            }
        };

        let out_dtype_val = if has_b {
            DType::promote(
                a_wrapper.dtype,
                NdArrayHandle::as_wrapper(b as *mut _).dtype,
            )
        } else {
            a_wrapper.dtype
        };

        macro_rules! extract_and_dispatch {
            ($dtype:ident, $array_data:ident, $extract_fn:ident) => {{
                let Some(arr) = $extract_fn(a_wrapper, a_meta) else {
                    set_last_error("Failed to extract array A".to_string());
                    return ERR_GENERIC;
                };
                let result = if has_b {
                    let Some(arr_b) = $extract_fn(NdArrayHandle::as_wrapper(b as *mut _), &*b_meta)
                    else {
                        set_last_error("Failed to extract array B".to_string());
                        return ERR_GENERIC;
                    };
                    dispatch(&arr, &arr_b, &spec)
                } else {
                    dispatch_single(&arr, &spec)
                };
                match result {
                    Ok(r) => NDArrayWrapper {
                        data: ArrayData::$array_data(Arc::new(RwLock::new(r))),
                        dtype: DType::$dtype,
                    },
                    Err(e) => {
                        set_last_error(e);
                        return ERR_SHAPE;
                    }
                }
            }};
        }

        let result_wrapper = match out_dtype_val {
            DType::Float64 => extract_and_dispatch!(Float64, Float64, extract_array_f64),
            DType::Float32 => extract_and_dispatch!(Float32, Float32, extract_array_f32),
            other => {
                set_last_error(format!(
                    "einsum only supports Float32/Float64, got {:?}",
                    other
                ));
                return ERR_DTYPE;
            }
        };

        if let Err(e) = write_output_metadata(
            &result_wrapper,
            out_dtype,
            out_ndim,
            out_shape_ptr,
            max_ndim,
        ) {
            set_last_error(e);
            return ERR_GENERIC;
        }
        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
