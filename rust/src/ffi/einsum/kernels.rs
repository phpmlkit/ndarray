//! Optimized einsum contraction kernels with deterministic accumulation order.

use ndarray::{ArrayD, Axis, IxDyn, ScalarOperand};
use num_traits::Zero;
use std::ops::{Add, Mul};

/// Deterministic matrix multiplication `ij,jk->ik`.
pub fn gemm<T>(a: &ArrayD<T>, b: &ArrayD<T>) -> ArrayD<T>
where
    T: Copy + ScalarOperand + Zero + Mul<Output = T> + Add<Output = T>,
{
    let m = a.shape()[0];
    let n = a.shape()[1];
    let p = b.shape()[1];
    let a_flat = a.as_slice().unwrap();
    let b_flat = b.as_slice().unwrap();
    let mut c = ArrayD::zeros(IxDyn(&[m, p]));
    let c_flat = c.as_slice_mut().unwrap();
    for i in 0..m {
        for k in 0..p {
            let row = i * n;
            let mut s = T::zero();
            for j in 0..n {
                s = s + a_flat[row + j] * b_flat[j * p + k];
            }
            c_flat[i * p + k] = s;
        }
    }
    c
}

pub fn gemm_transposed<T>(a: &ArrayD<T>, b: &ArrayD<T>) -> ArrayD<T>
where
    T: Copy + ScalarOperand + Zero + Mul<Output = T> + Add<Output = T>,
{
    let m = a.shape()[0];
    let n = a.shape()[1];
    let p = b.shape()[0];
    let asl = a.as_slice().unwrap();
    let bsl = b.as_slice().unwrap();
    let mut c = ArrayD::zeros(IxDyn(&[m, p]));
    let csl = c.as_slice_mut().unwrap();
    for i in 0..m {
        for k in 0..p {
            let ar = i * n;
            let br = k * n;
            let mut s = T::zero();
            for j in 0..n {
                s = s + asl[ar + j] * bsl[br + j];
            }
            csl[i * p + k] = s;
        }
    }
    c
}

pub fn dot<T>(a: &ArrayD<T>, b: &ArrayD<T>) -> T
where
    T: Copy + ScalarOperand + Zero + Mul<Output = T> + Add<Output = T>,
{
    let n = a.len();
    let asl = a.as_slice().unwrap();
    let bsl = b.as_slice().unwrap();
    let mut s = T::zero();
    for i in 0..n {
        s = s + asl[i] * bsl[i];
    }
    s
}

pub fn outer<T>(a: &ArrayD<T>, b: &ArrayD<T>) -> ArrayD<T>
where
    T: Copy + ScalarOperand + Zero + Mul<Output = T> + Add<Output = T>,
{
    let m = a.len();
    let n = b.len();
    let asl = a.as_slice().unwrap();
    let bsl = b.as_slice().unwrap();
    let mut c = ArrayD::zeros(IxDyn(&[m, n]));
    let cf = c.as_slice_mut().unwrap();
    for i in 0..m {
        let ai = asl[i];
        let rs = i * n;
        for j in 0..n {
            cf[rs + j] = ai * bsl[j];
        }
    }
    c
}

pub fn hadamard<T>(a: &ArrayD<T>, b: &ArrayD<T>) -> ArrayD<T>
where
    T: Copy + ScalarOperand + Zero + Mul<Output = T> + Add<Output = T>,
{
    let n = a.len();
    let asl = a.as_slice().unwrap();
    let bsl = b.as_slice().unwrap();
    let mut d = Vec::with_capacity(n);
    for i in 0..n {
        d.push(asl[i] * bsl[i]);
    }
    ArrayD::from_shape_vec(IxDyn(a.shape()), d).unwrap()
}

pub fn trace<T>(a: &ArrayD<T>) -> T
where
    T: Copy + ScalarOperand + Zero + Add<Output = T>,
{
    a.diag().iter().copied().fold(T::zero(), |acc, x| acc + x)
}

pub fn diagonal<T>(a: &ArrayD<T>) -> ArrayD<T>
where
    T: Copy + ScalarOperand + Zero,
{
    a.diag().to_owned().into_dyn()
}

pub fn transpose<T>(a: &ArrayD<T>, perm: &[usize]) -> ArrayD<T>
where
    T: Copy + ScalarOperand + Zero,
{
    let view = a.view().permuted_axes(perm);
    let data: Vec<T> = view.iter().copied().collect();
    ArrayD::from_shape_vec(IxDyn(view.shape()), data).unwrap()
}

pub fn sum_over_axis<T>(a: &ArrayD<T>, axis: usize) -> ArrayD<T>
where
    T: Copy + ScalarOperand + Zero,
{
    a.sum_axis(Axis(axis))
}

pub fn sum_all<T>(a: &ArrayD<T>) -> T
where
    T: Copy + ScalarOperand + Zero,
{
    a.sum()
}
