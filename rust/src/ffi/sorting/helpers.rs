//! Shared helpers for sorting-family operations.

use std::cmp::Ordering;

use ndarray::{ArrayBase, ArrayD, Axis, Data, IxDyn};

use crate::types::SortKind;

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

pub fn sort_axis_generic<T, F>(view: &ArrayD<T>, axis: usize, kind: SortKind, cmp: F) -> ArrayD<T>
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

pub fn sort_flat_generic<T, F>(view: &ArrayD<T>, kind: SortKind, cmp: F) -> ArrayD<T>
where
    T: Copy,
    F: Fn(&T, &T) -> Ordering + Copy,
{
    let mut flat: Vec<T> = view.iter().copied().collect();
    sort_by_kind(&mut flat, kind, |a, b| cmp(a, b));
    ArrayD::from_shape_vec(IxDyn(&[flat.len()]), flat)
        .expect("Failed to build flat sorted output")
}

pub fn argsort_axis_generic<T, F>(
    view: &ArrayD<T>,
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

pub fn argsort_flat_generic<T, F>(view: &ArrayD<T>, kind: SortKind, cmp: F) -> ArrayD<i64>
where
    T: Copy,
    F: Fn(&T, &T) -> Ordering + Copy,
{
    let values: Vec<T> = view.iter().copied().collect();
    let mut indices: Vec<usize> = (0..values.len()).collect();
    sort_by_kind(&mut indices, kind, |a, b| cmp(&values[*a], &values[*b]));
    let out: Vec<i64> = indices.into_iter().map(|i| i as i64).collect();
    ArrayD::from_shape_vec(IxDyn(&[out.len()]), out)
        .expect("Failed to build flat argsort output")
}

// ---------------------------------------------------------------------------
// heap helpers
// ---------------------------------------------------------------------------

fn build_heap_by<T, F>(values: &mut [T], end: usize, cmp: &mut F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    if end < 2 {
        return;
    }
    let mut start = (end - 2) / 2;
    loop {
        sift_down_by(values, start, end, cmp);
        if start == 0 {
            break;
        }
        start -= 1;
    }
}

fn heap_topk<T, F>(
    data: &[T],
    k: usize,
    largest: bool,
    cmp_asc: &mut F,
) -> Vec<(T, usize)>
where
    T: Copy,
    F: FnMut(&T, &T) -> Ordering,
{
    let n = data.len();
    if k == 0 {
        return Vec::new();
    }
    if k >= n {
        let mut result: Vec<_> = data
            .iter()
            .copied()
            .enumerate()
            .map(|(i, v)| (v, i))
            .collect();
        if largest {
            result.sort_unstable_by(|a, b| cmp_asc(&b.0, &a.0));
        } else {
            result.sort_unstable_by(|a, b| cmp_asc(&a.0, &b.0));
        }
        return result;
    }

    let mut heap: Vec<(T, usize)> = Vec::with_capacity(k);

    for i in 0..k {
        heap.push((data[i], i));
    }

    if largest {
        let mut min_cmp = |a: &(T, usize), b: &(T, usize)| cmp_asc(&b.0, &a.0);
        build_heap_by(&mut heap, k - 1, &mut min_cmp);
    } else {
        let mut max_cmp = |a: &(T, usize), b: &(T, usize)| cmp_asc(&a.0, &b.0);
        build_heap_by(&mut heap, k - 1, &mut max_cmp);
    }

    for i in k..n {
        let cmp_root = cmp_asc(&heap[0].0, &data[i]);
        let replace = if largest {
            cmp_root == Ordering::Less
        } else {
            cmp_root == Ordering::Greater
        };
        if replace {
            heap[0] = (data[i], i);
            if largest {
                let mut min_cmp = |a: &(T, usize), b: &(T, usize)| cmp_asc(&b.0, &a.0);
                sift_down_by(&mut heap, 0, k - 1, &mut min_cmp);
            } else {
                let mut max_cmp = |a: &(T, usize), b: &(T, usize)| cmp_asc(&a.0, &b.0);
                sift_down_by(&mut heap, 0, k - 1, &mut max_cmp);
            }
        }
    }

    if largest {
        heap.sort_unstable_by(|a, b| cmp_asc(&b.0, &a.0));
    } else {
        heap.sort_unstable_by(|a, b| cmp_asc(&a.0, &b.0));
    }

    heap
}

// ---------------------------------------------------------------------------
// topk
// ---------------------------------------------------------------------------

pub fn topk_axis_generic<T, D, F>(
    view: &ArrayBase<D, IxDyn>,
    axis: usize,
    k: usize,
    largest: bool,
    sorted: bool,
    kind: SortKind,
    cmp_asc: F,
) -> (ArrayD<T>, ArrayD<i64>)
where
    T: Copy + Default,
    D: Data<Elem = T>,
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
        let n = lane_in.len();

        if k == 0 {
            continue;
        }

        if k == 1 {
            let mut best_idx = 0usize;
            let mut best_val = lane_in[0];
            for (i, &val) in lane_in.iter().enumerate().skip(1) {
                let better = if largest {
                    cmp_asc(&best_val, &val) == Ordering::Less
                } else {
                    cmp_asc(&val, &best_val) == Ordering::Less
                };
                if better {
                    best_val = val;
                    best_idx = i;
                }
            }
            lane_vals[0] = best_val;
            lane_idxs[0] = best_idx as i64;
            continue;
        }

        if k * 4 < n {
            let mut cmp = cmp_asc;
            let top_items: Vec<(T, usize)> = if let Some(slice) = lane_in.as_slice() {
                heap_topk(slice, k, largest, &mut cmp)
            } else {
                let tmp: Vec<T> = lane_in.iter().copied().collect();
                heap_topk(&tmp, k, largest, &mut cmp)
            };

            if !sorted {
                let mut items = top_items;
                items.sort_unstable_by_key(|&(_, idx)| idx);
                for (i, &(val, idx)) in items.iter().enumerate() {
                    lane_vals[i] = val;
                    lane_idxs[i] = idx as i64;
                }
            } else {
                for (i, &(val, idx)) in top_items.iter().enumerate() {
                    lane_vals[i] = val;
                    lane_idxs[i] = idx as i64;
                }
            }
            continue;
        }

        // Full sort for large k

        idx_scratch.clear();
        idx_scratch.extend(0..n);

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

pub fn topk_flat_generic<T, D, F>(
    view: &ArrayBase<D, IxDyn>,
    k: usize,
    largest: bool,
    sorted: bool,
    kind: SortKind,
    cmp_asc: F,
) -> (ArrayD<T>, ArrayD<i64>)
where
    T: Copy,
    D: Data<Elem = T>,
    F: Fn(&T, &T) -> Ordering + Copy,
{
    let flat: Vec<T> = view.iter().copied().collect();
    let n = flat.len();
    let out_k = k.min(n);

    if out_k == 0 {
        return (
            ArrayD::from_shape_vec(IxDyn(&[0]), Vec::new()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[0]), Vec::new()).unwrap(),
        );
    }

    let mut out_vals: Vec<T> = Vec::with_capacity(out_k);
    let mut out_idxs: Vec<i64> = Vec::with_capacity(out_k);

    if k * 4 < n {
        let mut cmp = cmp_asc;
        let top_items = heap_topk(&flat, k, largest, &mut cmp);

        let items: Vec<_> = if sorted {
            top_items
        } else {
            let mut items = top_items;
            items.sort_unstable_by_key(|&(_, idx)| idx);
            items
        };

        for &(val, idx) in items.iter().take(out_k) {
            out_vals.push(val);
            out_idxs.push(idx as i64);
        }
    } else {
        let mut indices: Vec<usize> = (0..n).collect();

        if largest {
            sort_by_kind(&mut indices, kind, |a, b| cmp_asc(&flat[*b], &flat[*a]));
        } else {
            sort_by_kind(&mut indices, kind, |a, b| cmp_asc(&flat[*a], &flat[*b]));
        }

        if !sorted {
            indices[..out_k].sort_unstable();
        }

        for &idx in indices.iter().take(out_k) {
            out_vals.push(flat[idx]);
            out_idxs.push(idx as i64);
        }
    }

    (
        ArrayD::from_shape_vec(IxDyn(&[out_k]), out_vals).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[out_k]), out_idxs).unwrap(),
    )
}
