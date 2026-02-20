//! Shared helpers for sorting-family operations.

use std::cmp::Ordering;

use ndarray::{ArrayD, ArrayViewD, Axis, IxDyn};

/// Sorting algorithm selection for sort/argsort/topk operations.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortKind {
    QuickSort = 0,
    MergeSort = 1,
    HeapSort = 2,
    Stable = 3,
}

impl SortKind {
    /// Parse SortKind from FFI integer value.
    pub fn from_i32(value: i32) -> Result<Self, String> {
        match value {
            0 => Ok(SortKind::QuickSort),
            1 => Ok(SortKind::MergeSort),
            2 => Ok(SortKind::HeapSort),
            3 => Ok(SortKind::Stable),
            _ => Err(format!("Invalid sort kind: {}", value)),
        }
    }
}

pub fn cmp_f64_asc_nan_last(a: &f64, b: &f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
    }
}

pub fn cmp_f32_asc_nan_last(a: &f32, b: &f32) -> Ordering {
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

pub fn sort_by_kind<T, F>(values: &mut [T], kind: SortKind, mut cmp: F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    match kind {
        SortKind::QuickSort => values.sort_unstable_by(|a, b| cmp(a, b)),
        SortKind::MergeSort | SortKind::Stable => values.sort_by(|a, b| cmp(a, b)),
        SortKind::HeapSort => heapsort_by(values, cmp),
    }
}

pub fn sort_axis_generic<T, F>(
    view: ArrayViewD<'_, T>,
    axis: usize,
    kind: SortKind,
    cmp: F,
) -> ArrayD<T>
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

pub fn sort_flat_generic<T, F>(view: ArrayViewD<'_, T>, kind: SortKind, cmp: F) -> ArrayD<T>
where
    T: Copy,
    F: Fn(&T, &T) -> Ordering + Copy,
{
    let mut flat: Vec<T> = view.iter().copied().collect();
    sort_by_kind(&mut flat, kind, |a, b| cmp(a, b));
    ArrayD::from_shape_vec(IxDyn(&[flat.len()]), flat).expect("Failed to build flat sorted output")
}

pub fn argsort_axis_generic<T, F>(
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
        sort_by_kind(&mut idx_scratch, kind, |a, b| {
            cmp(&lane_in[*a], &lane_in[*b])
        });
        for (dst, src) in lane_out.iter_mut().zip(idx_scratch.iter().copied()) {
            *dst = src as i64;
        }
    }

    result
}

pub fn argsort_flat_generic<T, F>(view: ArrayViewD<'_, T>, kind: SortKind, cmp: F) -> ArrayD<i64>
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

pub fn topk_axis_generic<T, F>(
    view: ArrayViewD<'_, T>,
    axis: usize,
    k: usize,
    largest: bool,
    sorted: bool,
    kind: SortKind,
    cmp_asc: F,
) -> (ArrayD<T>, ArrayD<i64>)
where
    T: Copy + Default,
    F: Fn(&T, &T) -> Ordering + Copy,
{
    let mut out_shape = view.shape().to_vec();
    out_shape[axis] = k;

    let mut out_values = ArrayD::<T>::default(IxDyn(&out_shape));
    let mut out_indices = ArrayD::<i64>::zeros(IxDyn(&out_shape));

    let mut idx_scratch: Vec<usize> = Vec::new();

    for ((lane_in, mut lane_vals), mut lane_idxs) in view
        .lanes(Axis(axis))
        .into_iter()
        .zip(out_values.lanes_mut(Axis(axis)))
        .zip(out_indices.lanes_mut(Axis(axis)))
    {
        idx_scratch.clear();
        idx_scratch.extend(0..lane_in.len());

        if largest {
            sort_by_kind(&mut idx_scratch, kind, |a, b| {
                cmp_asc(&lane_in[*b], &lane_in[*a])
            });
        } else {
            sort_by_kind(&mut idx_scratch, kind, |a, b| {
                cmp_asc(&lane_in[*a], &lane_in[*b])
            });
        }

        if !sorted {
            idx_scratch[..k].sort_unstable();
        }

        for i in 0..k {
            let src_idx = idx_scratch[i];
            lane_vals[i] = lane_in[src_idx];
            lane_idxs[i] = src_idx as i64;
        }
    }

    (out_values, out_indices)
}

pub fn topk_flat_generic<T, F>(
    view: ArrayViewD<'_, T>,
    k: usize,
    largest: bool,
    sorted: bool,
    kind: SortKind,
    cmp_asc: F,
) -> (ArrayD<T>, ArrayD<i64>)
where
    T: Copy,
    F: Fn(&T, &T) -> Ordering + Copy,
{
    let flat: Vec<T> = view.iter().copied().collect();
    let mut indices: Vec<usize> = (0..flat.len()).collect();

    if largest {
        sort_by_kind(&mut indices, kind, |a, b| cmp_asc(&flat[*b], &flat[*a]));
    } else {
        sort_by_kind(&mut indices, kind, |a, b| cmp_asc(&flat[*a], &flat[*b]));
    }

    if !sorted {
        indices[..k].sort_unstable();
    }

    let mut out_vals: Vec<T> = Vec::with_capacity(k);
    let mut out_idxs: Vec<i64> = Vec::with_capacity(k);
    for &idx in indices.iter().take(k) {
        out_vals.push(flat[idx]);
        out_idxs.push(idx as i64);
    }

    (
        ArrayD::from_shape_vec(IxDyn(&[k]), out_vals).expect("Failed to build topk flat values"),
        ArrayD::from_shape_vec(IxDyn(&[k]), out_idxs).expect("Failed to build topk flat indices"),
    )
}
