//! Sort and argsort operations (axis-aware and flattened).

use std::cmp::Ordering;

use crate::core::view_helpers::{
    extract_view_bool, extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32,
    extract_view_i64, extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64,
    extract_view_u8,
};
use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::reductions::helpers::{validate_axis, SortKind};
use crate::ffi::NdArrayHandle;
use ndarray::{ArrayD, ArrayViewD, Axis, IxDyn};
use parking_lot::RwLock;
use std::slice;
use std::sync::Arc;

fn cmp_f64_asc_nan_last(a: &f64, b: &f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
    }
}

fn cmp_f32_asc_nan_last(a: &f32, b: &f32) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
    }
}

fn sift_down_by<T, F>(values: &mut [T], start: usize, end: usize, cmp: &mut F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    let mut root = start;
    loop {
        let child = root * 2 + 1;
        if child > end {
            return;
        }

        let mut swap_idx = root;
        if cmp(&values[swap_idx], &values[child]) == Ordering::Less {
            swap_idx = child;
        }
        if child < end && cmp(&values[swap_idx], &values[child + 1]) == Ordering::Less {
            swap_idx = child + 1;
        }
        if swap_idx == root {
            return;
        }

        values.swap(root, swap_idx);
        root = swap_idx;
    }
}

fn heapsort_by<T, F>(values: &mut [T], mut cmp: F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    if values.len() < 2 {
        return;
    }

    let mut start = (values.len() - 2) / 2;
    loop {
        sift_down_by(values, start, values.len() - 1, &mut cmp);
        if start == 0 {
            break;
        }
        start -= 1;
    }

    let mut end = values.len() - 1;
    while end > 0 {
        values.swap(0, end);
        end -= 1;
        sift_down_by(values, 0, end, &mut cmp);
    }
}

fn sort_by_kind<T, F>(values: &mut [T], kind: SortKind, mut cmp: F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    match kind {
        SortKind::QuickSort => values.sort_unstable_by(|a, b| cmp(a, b)),
        SortKind::MergeSort | SortKind::Stable => values.sort_by(|a, b| cmp(a, b)),
        SortKind::HeapSort => heapsort_by(values, cmp),
    }
}

fn sort_axis_generic<T, F>(view: ArrayViewD<'_, T>, axis: usize, kind: SortKind, cmp: F) -> ArrayD<T>
where
    T: Copy,
    F: Fn(&T, &T) -> Ordering + Copy,
{
    let mut result = view.to_owned();
    let mut scratch: Vec<T> = Vec::new();

    for mut lane in result.lanes_mut(Axis(axis)) {
        scratch.clear();
        scratch.extend(lane.iter().copied());
        sort_by_kind(&mut scratch, kind, |a, b| cmp(a, b));
        for (dst, src) in lane.iter_mut().zip(scratch.iter().copied()) {
            *dst = src;
        }
    }

    result
}

fn sort_flat_generic<T, F>(view: ArrayViewD<'_, T>, kind: SortKind, cmp: F) -> ArrayD<T>
where
    T: Copy,
    F: Fn(&T, &T) -> Ordering + Copy,
{
    let mut flat: Vec<T> = view.iter().copied().collect();
    sort_by_kind(&mut flat, kind, |a, b| cmp(a, b));
    ArrayD::from_shape_vec(IxDyn(&[flat.len()]), flat).expect("Failed to build flat sorted output")
}

fn argsort_axis_generic<T, F>(
    view: ArrayViewD<'_, T>,
    axis: usize,
    kind: SortKind,
    cmp: F,
) -> ArrayD<i64>
where
    T: Copy,
    F: Fn(&T, &T) -> Ordering + Copy,
{
    let mut result = ArrayD::<i64>::zeros(IxDyn(view.shape()));
    let mut idx_scratch: Vec<usize> = Vec::new();

    for (lane_in, mut lane_out) in view
        .lanes(Axis(axis))
        .into_iter()
        .zip(result.lanes_mut(Axis(axis)))
    {
        idx_scratch.clear();
        idx_scratch.extend(0..lane_in.len());
        sort_by_kind(&mut idx_scratch, kind, |a, b| cmp(&lane_in[*a], &lane_in[*b]));
        for (dst, src) in lane_out.iter_mut().zip(idx_scratch.iter().copied()) {
            *dst = src as i64;
        }
    }

    result
}

fn argsort_flat_generic<T, F>(view: ArrayViewD<'_, T>, kind: SortKind, cmp: F) -> ArrayD<i64>
where
    T: Copy,
    F: Fn(&T, &T) -> Ordering + Copy,
{
    let values: Vec<T> = view.iter().copied().collect();
    let mut indices: Vec<usize> = (0..values.len()).collect();
    sort_by_kind(&mut indices, kind, |a, b| cmp(&values[*a], &values[*b]));
    let out: Vec<i64> = indices.into_iter().map(|i| i as i64).collect();
    ArrayD::from_shape_vec(IxDyn(&[out.len()]), out).expect("Failed to build flat argsort output")
}

/// Sort along an axis. Output dtype matches input dtype and shape matches input shape.
#[no_mangle]
pub unsafe extern "C" fn ndarray_sort_axis(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
    axis: i32,
    kind: i32,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || out_handle.is_null() || shape.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = slice::from_raw_parts(shape, ndim);
        let strides_slice = slice::from_raw_parts(strides, ndim);

        let axis_usize = match validate_axis(shape_slice, axis) {
            Ok(a) => a,
            Err(e) => {
                crate::error::set_last_error(e);
                return ERR_GENERIC;
            }
        };

        let sort_kind = match SortKind::from_i32(kind) {
            Ok(k) => k,
            Err(e) => {
                crate::error::set_last_error(e);
                return ERR_GENERIC;
            }
        };

        let result_wrapper = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = sort_axis_generic(view, axis_usize, sort_kind, cmp_f64_asc_nan_last);
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                let result = sort_axis_generic(view, axis_usize, sort_kind, cmp_f32_asc_nan_last);
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(result))),
                    dtype: DType::Float32,
                }
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = sort_axis_generic(view, axis_usize, sort_kind, |a, b| a.cmp(b));
                NDArrayWrapper {
                    data: ArrayData::Int64(Arc::new(RwLock::new(result))),
                    dtype: DType::Int64,
                }
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                let result = sort_axis_generic(view, axis_usize, sort_kind, |a, b| a.cmp(b));
                NDArrayWrapper {
                    data: ArrayData::Int32(Arc::new(RwLock::new(result))),
                    dtype: DType::Int32,
                }
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                let result = sort_axis_generic(view, axis_usize, sort_kind, |a, b| a.cmp(b));
                NDArrayWrapper {
                    data: ArrayData::Int16(Arc::new(RwLock::new(result))),
                    dtype: DType::Int16,
                }
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                let result = sort_axis_generic(view, axis_usize, sort_kind, |a, b| a.cmp(b));
                NDArrayWrapper {
                    data: ArrayData::Int8(Arc::new(RwLock::new(result))),
                    dtype: DType::Int8,
                }
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = sort_axis_generic(view, axis_usize, sort_kind, |a, b| a.cmp(b));
                NDArrayWrapper {
                    data: ArrayData::Uint64(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint64,
                }
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                let result = sort_axis_generic(view, axis_usize, sort_kind, |a, b| a.cmp(b));
                NDArrayWrapper {
                    data: ArrayData::Uint32(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint32,
                }
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                let result = sort_axis_generic(view, axis_usize, sort_kind, |a, b| a.cmp(b));
                NDArrayWrapper {
                    data: ArrayData::Uint16(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint16,
                }
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                let result = sort_axis_generic(view, axis_usize, sort_kind, |a, b| a.cmp(b));
                NDArrayWrapper {
                    data: ArrayData::Uint8(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint8,
                }
            }
            DType::Bool => {
                let Some(view) = extract_view_bool(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract bool view".to_string());
                    return ERR_GENERIC;
                };
                let result = sort_axis_generic(view, axis_usize, sort_kind, |a, b| a.cmp(b));
                NDArrayWrapper {
                    data: ArrayData::Bool(Arc::new(RwLock::new(result))),
                    dtype: DType::Bool,
                }
            }
        };

        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}

/// Sort flattened array (axis=None behavior). Output is 1D.
#[no_mangle]
pub unsafe extern "C" fn ndarray_sort_flat(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
    kind: i32,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || out_handle.is_null() || shape.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = slice::from_raw_parts(shape, ndim);
        let strides_slice = slice::from_raw_parts(strides, ndim);

        let sort_kind = match SortKind::from_i32(kind) {
            Ok(k) => k,
            Err(e) => {
                crate::error::set_last_error(e);
                return ERR_GENERIC;
            }
        };

        let result_wrapper = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = sort_flat_generic(view, sort_kind, cmp_f64_asc_nan_last);
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(result))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                let result = sort_flat_generic(view, sort_kind, cmp_f32_asc_nan_last);
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(result))),
                    dtype: DType::Float32,
                }
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = sort_flat_generic(view, sort_kind, |a, b| a.cmp(b));
                NDArrayWrapper {
                    data: ArrayData::Int64(Arc::new(RwLock::new(result))),
                    dtype: DType::Int64,
                }
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                let result = sort_flat_generic(view, sort_kind, |a, b| a.cmp(b));
                NDArrayWrapper {
                    data: ArrayData::Int32(Arc::new(RwLock::new(result))),
                    dtype: DType::Int32,
                }
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                let result = sort_flat_generic(view, sort_kind, |a, b| a.cmp(b));
                NDArrayWrapper {
                    data: ArrayData::Int16(Arc::new(RwLock::new(result))),
                    dtype: DType::Int16,
                }
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                let result = sort_flat_generic(view, sort_kind, |a, b| a.cmp(b));
                NDArrayWrapper {
                    data: ArrayData::Int8(Arc::new(RwLock::new(result))),
                    dtype: DType::Int8,
                }
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                let result = sort_flat_generic(view, sort_kind, |a, b| a.cmp(b));
                NDArrayWrapper {
                    data: ArrayData::Uint64(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint64,
                }
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                let result = sort_flat_generic(view, sort_kind, |a, b| a.cmp(b));
                NDArrayWrapper {
                    data: ArrayData::Uint32(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint32,
                }
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                let result = sort_flat_generic(view, sort_kind, |a, b| a.cmp(b));
                NDArrayWrapper {
                    data: ArrayData::Uint16(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint16,
                }
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                let result = sort_flat_generic(view, sort_kind, |a, b| a.cmp(b));
                NDArrayWrapper {
                    data: ArrayData::Uint8(Arc::new(RwLock::new(result))),
                    dtype: DType::Uint8,
                }
            }
            DType::Bool => {
                let Some(view) = extract_view_bool(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract bool view".to_string());
                    return ERR_GENERIC;
                };
                let result = sort_flat_generic(view, sort_kind, |a, b| a.cmp(b));
                NDArrayWrapper {
                    data: ArrayData::Bool(Arc::new(RwLock::new(result))),
                    dtype: DType::Bool,
                }
            }
        };

        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}

/// Argsort along an axis. Output dtype is Int64 and shape matches input shape.
#[no_mangle]
pub unsafe extern "C" fn ndarray_argsort_axis(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
    axis: i32,
    kind: i32,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || out_handle.is_null() || shape.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = slice::from_raw_parts(shape, ndim);
        let strides_slice = slice::from_raw_parts(strides, ndim);

        let axis_usize = match validate_axis(shape_slice, axis) {
            Ok(a) => a,
            Err(e) => {
                crate::error::set_last_error(e);
                return ERR_GENERIC;
            }
        };

        let sort_kind = match SortKind::from_i32(kind) {
            Ok(k) => k,
            Err(e) => {
                crate::error::set_last_error(e);
                return ERR_GENERIC;
            }
        };

        let result = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_axis_generic(view, axis_usize, sort_kind, cmp_f64_asc_nan_last)
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_axis_generic(view, axis_usize, sort_kind, cmp_f32_asc_nan_last)
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_axis_generic(view, axis_usize, sort_kind, |a, b| a.cmp(b))
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_axis_generic(view, axis_usize, sort_kind, |a, b| a.cmp(b))
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_axis_generic(view, axis_usize, sort_kind, |a, b| a.cmp(b))
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_axis_generic(view, axis_usize, sort_kind, |a, b| a.cmp(b))
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_axis_generic(view, axis_usize, sort_kind, |a, b| a.cmp(b))
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_axis_generic(view, axis_usize, sort_kind, |a, b| a.cmp(b))
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_axis_generic(view, axis_usize, sort_kind, |a, b| a.cmp(b))
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_axis_generic(view, axis_usize, sort_kind, |a, b| a.cmp(b))
            }
            DType::Bool => {
                let Some(view) = extract_view_bool(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract bool view".to_string());
                    return ERR_GENERIC;
                };
                argsort_axis_generic(view, axis_usize, sort_kind, |a, b| a.cmp(b))
            }
        };

        let result_wrapper = NDArrayWrapper {
            data: ArrayData::Int64(Arc::new(RwLock::new(result))),
            dtype: DType::Int64,
        };
        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}

/// Argsort flattened array (axis=None behavior). Output is Int64 1D.
#[no_mangle]
pub unsafe extern "C" fn ndarray_argsort_flat(
    handle: *const NdArrayHandle,
    offset: usize,
    shape: *const usize,
    strides: *const usize,
    ndim: usize,
    kind: i32,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || out_handle.is_null() || shape.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let shape_slice = slice::from_raw_parts(shape, ndim);
        let strides_slice = slice::from_raw_parts(strides, ndim);

        let sort_kind = match SortKind::from_i32(kind) {
            Ok(k) => k,
            Err(e) => {
                crate::error::set_last_error(e);
                return ERR_GENERIC;
            }
        };

        let result = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_flat_generic(view, sort_kind, cmp_f64_asc_nan_last)
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_flat_generic(view, sort_kind, cmp_f32_asc_nan_last)
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_flat_generic(view, sort_kind, |a, b| a.cmp(b))
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_flat_generic(view, sort_kind, |a, b| a.cmp(b))
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_flat_generic(view, sort_kind, |a, b| a.cmp(b))
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_flat_generic(view, sort_kind, |a, b| a.cmp(b))
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_flat_generic(view, sort_kind, |a, b| a.cmp(b))
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_flat_generic(view, sort_kind, |a, b| a.cmp(b))
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_flat_generic(view, sort_kind, |a, b| a.cmp(b))
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                argsort_flat_generic(view, sort_kind, |a, b| a.cmp(b))
            }
            DType::Bool => {
                let Some(view) = extract_view_bool(wrapper, offset, shape_slice, strides_slice)
                else {
                    crate::error::set_last_error("Failed to extract bool view".to_string());
                    return ERR_GENERIC;
                };
                argsort_flat_generic(view, sort_kind, |a, b| a.cmp(b))
            }
        };

        let result_wrapper = NDArrayWrapper {
            data: ArrayData::Int64(Arc::new(RwLock::new(result))),
            dtype: DType::Int64,
        };
        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
