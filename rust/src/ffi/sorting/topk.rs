//! Top-k along an axis (fast path for contiguous arrays).

use std::sync::Arc;

use parking_lot::RwLock;

use crate::ffi::sorting::helpers::{
    cmp_f32_asc_nan_last, cmp_f64_asc_nan_last, topk_axis_generic, topk_flat_generic,
};
use crate::helpers::error::{set_last_error, ERR_DTYPE, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::helpers::is_c_contiguous;
use crate::helpers::normalize_axis;
use crate::helpers::{
    extract_array_bool, extract_array_f32, extract_array_f64, extract_array_i16, extract_array_i32,
    extract_array_i64, extract_array_i8, extract_array_u16, extract_array_u32, extract_array_u64,
    extract_array_u8, extract_view_bool, extract_view_f32, extract_view_f64, extract_view_i16,
    extract_view_i32, extract_view_i64, extract_view_i8, extract_view_u16, extract_view_u32,
    extract_view_u64, extract_view_u8,
};
use crate::types::dtype::DType;
use crate::types::SortKind;
use crate::types::{ArrayData, ArrayMetadata, NDArrayWrapper, NdArrayHandle};

// ---------------------------------------------------------------------------
// Macros
// ---------------------------------------------------------------------------

macro_rules! topk_axis_arm {
    ($contig:expr, $wrapper:expr, $meta:expr,
     $view_fn:ident, $array_fn:ident,
     $cmp:expr, $variant:ident, $dtype:ident,
     $axis:expr, $k:expr, $largest:expr, $sorted:expr, $kind:expr) => {{
        if $contig {
            let Some(view) = $view_fn($wrapper, $meta) else {
                set_last_error(format!("Failed to extract {} view", stringify!($variant)));
                return ERR_DTYPE;
            };
            let (vals, idxs) = topk_axis_generic(
                &view, $axis, $k, $largest, $sorted, $kind, $cmp,
            );
            (
                NDArrayWrapper { data: ArrayData::$variant(Arc::new(RwLock::new(vals))), dtype: DType::$dtype },
                NDArrayWrapper { data: ArrayData::Int64(Arc::new(RwLock::new(idxs))), dtype: DType::Int64 },
            )
        } else {
            let Some(arr) = $array_fn($wrapper, $meta) else {
                set_last_error(format!("Failed to extract {} view", stringify!($variant)));
                return ERR_DTYPE;
            };
            let (vals, idxs) = topk_axis_generic(
                &arr, $axis, $k, $largest, $sorted, $kind, $cmp,
            );
            (
                NDArrayWrapper { data: ArrayData::$variant(Arc::new(RwLock::new(vals))), dtype: DType::$dtype },
                NDArrayWrapper { data: ArrayData::Int64(Arc::new(RwLock::new(idxs))), dtype: DType::Int64 },
            )
        }
    }};
}

macro_rules! topk_flat_arm {
    ($contig:expr, $wrapper:expr, $meta:expr,
     $view_fn:ident, $array_fn:ident,
     $cmp:expr, $variant:ident, $dtype:ident,
     $k:expr, $largest:expr, $sorted:expr, $kind:expr) => {{
        if $contig {
            let Some(view) = $view_fn($wrapper, $meta) else {
                set_last_error(format!("Failed to extract {} view", stringify!($variant)));
                return ERR_DTYPE;
            };
            let (vals, idxs) = topk_flat_generic(
                &view, $k, $largest, $sorted, $kind, $cmp,
            );
            (
                NDArrayWrapper { data: ArrayData::$variant(Arc::new(RwLock::new(vals))), dtype: DType::$dtype },
                NDArrayWrapper { data: ArrayData::Int64(Arc::new(RwLock::new(idxs))), dtype: DType::Int64 },
            )
        } else {
            let Some(arr) = $array_fn($wrapper, $meta) else {
                set_last_error(format!("Failed to extract {} view", stringify!($variant)));
                return ERR_DTYPE;
            };
            let (vals, idxs) = topk_flat_generic(
                &arr, $k, $largest, $sorted, $kind, $cmp,
            );
            (
                NDArrayWrapper { data: ArrayData::$variant(Arc::new(RwLock::new(vals))), dtype: DType::$dtype },
                NDArrayWrapper { data: ArrayData::Int64(Arc::new(RwLock::new(idxs))), dtype: DType::Int64 },
            )
        }
    }};
}

// ---------------------------------------------------------------------------
// axis topk
// ---------------------------------------------------------------------------

/// Compute the top-k values and indices along an axis in the array.
#[no_mangle]
pub unsafe extern "C" fn ndarray_topk_axis(
    handle: *const NdArrayHandle,
    meta: *const ArrayMetadata,
    axis: i32,
    k: usize,
    largest: bool,
    sorted: bool,
    kind: i32,
    out_values: *mut *mut NdArrayHandle,
    out_indices: *mut *mut NdArrayHandle,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if handle.is_null()
        || out_values.is_null()
        || out_indices.is_null()
        || meta.is_null()
        || out_shape.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let meta = &*meta;
        let shape_slice = meta.shape_slice();
        let strides_slice = meta.strides_slice();

        let axis_usize = match normalize_axis(shape_slice, axis, false) {
            Ok(a) => a,
            Err(e) => {
                set_last_error(e);
                return ERR_SHAPE;
            }
        };

        if k > shape_slice[axis_usize] {
            set_last_error(format!(
                "k={} is larger than selected axis size {}",
                k, shape_slice[axis_usize]
            ));
            return ERR_GENERIC;
        }

        let sort_kind = match SortKind::from_i32(kind) {
            Ok(k) => k,
            Err(e) => {
                set_last_error(e);
                return ERR_GENERIC;
            }
        };

        let contig = is_c_contiguous(shape_slice, strides_slice);

        let (values_wrapper, indices_wrapper) = match wrapper.dtype {
            DType::Float64 => topk_axis_arm!(
                contig, wrapper, meta,
                extract_view_f64, extract_array_f64,
                cmp_f64_asc_nan_last, Float64, Float64,
                axis_usize, k, largest, sorted, sort_kind
            ),
            DType::Float32 => topk_axis_arm!(
                contig, wrapper, meta,
                extract_view_f32, extract_array_f32,
                cmp_f32_asc_nan_last, Float32, Float32,
                axis_usize, k, largest, sorted, sort_kind
            ),
            DType::Int64 => topk_axis_arm!(
                contig, wrapper, meta,
                extract_view_i64, extract_array_i64,
                |a, b| a.cmp(b), Int64, Int64,
                axis_usize, k, largest, sorted, sort_kind
            ),
            DType::Int32 => topk_axis_arm!(
                contig, wrapper, meta,
                extract_view_i32, extract_array_i32,
                |a, b| a.cmp(b), Int32, Int32,
                axis_usize, k, largest, sorted, sort_kind
            ),
            DType::Int16 => topk_axis_arm!(
                contig, wrapper, meta,
                extract_view_i16, extract_array_i16,
                |a, b| a.cmp(b), Int16, Int16,
                axis_usize, k, largest, sorted, sort_kind
            ),
            DType::Int8 => topk_axis_arm!(
                contig, wrapper, meta,
                extract_view_i8, extract_array_i8,
                |a, b| a.cmp(b), Int8, Int8,
                axis_usize, k, largest, sorted, sort_kind
            ),
            DType::Uint64 => topk_axis_arm!(
                contig, wrapper, meta,
                extract_view_u64, extract_array_u64,
                |a, b| a.cmp(b), Uint64, Uint64,
                axis_usize, k, largest, sorted, sort_kind
            ),
            DType::Uint32 => topk_axis_arm!(
                contig, wrapper, meta,
                extract_view_u32, extract_array_u32,
                |a, b| a.cmp(b), Uint32, Uint32,
                axis_usize, k, largest, sorted, sort_kind
            ),
            DType::Uint16 => topk_axis_arm!(
                contig, wrapper, meta,
                extract_view_u16, extract_array_u16,
                |a, b| a.cmp(b), Uint16, Uint16,
                axis_usize, k, largest, sorted, sort_kind
            ),
            DType::Uint8 => topk_axis_arm!(
                contig, wrapper, meta,
                extract_view_u8, extract_array_u8,
                |a, b| a.cmp(b), Uint8, Uint8,
                axis_usize, k, largest, sorted, sort_kind
            ),
            DType::Bool => topk_axis_arm!(
                contig, wrapper, meta,
                extract_view_bool, extract_array_bool,
                |a, b| a.cmp(b), Bool, Bool,
                axis_usize, k, largest, sorted, sort_kind
            ),
            DType::Complex64 | DType::Complex128 => {
                set_last_error("Topk is not supported for complex dtypes".to_string());
                return ERR_DTYPE;
            }
        };

        // Compute output shape
        let mut result_shape = shape_slice.to_vec();
        result_shape[axis_usize] = k;
        let out_ndim = result_shape.len();
        if out_ndim > max_ndim {
            set_last_error(format!(
                "output ndim {} exceeds max_ndim {}",
                out_ndim, max_ndim
            ));
            return ERR_GENERIC;
        }
        for (i, &dim) in result_shape.iter().enumerate() {
            *out_shape.add(i) = dim;
        }

        *out_values = NdArrayHandle::from_wrapper(Box::new(values_wrapper));
        *out_indices = NdArrayHandle::from_wrapper(Box::new(indices_wrapper));
        SUCCESS
    })
}

// ---------------------------------------------------------------------------
// flat topk
// ---------------------------------------------------------------------------

/// Compute the top-k values and indices of the flattened array.
#[no_mangle]
pub unsafe extern "C" fn ndarray_topk_flat(
    handle: *const NdArrayHandle,
    meta: *const ArrayMetadata,
    k: usize,
    largest: bool,
    sorted: bool,
    kind: i32,
    out_values: *mut *mut NdArrayHandle,
    out_indices: *mut *mut NdArrayHandle,
    out_shape: *mut usize,
) -> i32 {
    if handle.is_null()
        || out_values.is_null()
        || out_indices.is_null()
        || meta.is_null()
        || out_shape.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let meta = &*meta;
        let shape_slice = meta.shape_slice();
        let total = shape_slice.iter().copied().product::<usize>();

        if k > total {
            set_last_error(format!("k={} is larger than flattened size {}", k, total));
            return ERR_GENERIC;
        }

        let sort_kind = match SortKind::from_i32(kind) {
            Ok(k) => k,
            Err(e) => {
                set_last_error(e);
                return ERR_GENERIC;
            }
        };

        let contig = is_c_contiguous(shape_slice, meta.strides_slice());

        let (values_wrapper, indices_wrapper) = match wrapper.dtype {
            DType::Float64 => topk_flat_arm!(
                contig, wrapper, meta,
                extract_view_f64, extract_array_f64,
                cmp_f64_asc_nan_last, Float64, Float64,
                k, largest, sorted, sort_kind
            ),
            DType::Float32 => topk_flat_arm!(
                contig, wrapper, meta,
                extract_view_f32, extract_array_f32,
                cmp_f32_asc_nan_last, Float32, Float32,
                k, largest, sorted, sort_kind
            ),
            DType::Int64 => topk_flat_arm!(
                contig, wrapper, meta,
                extract_view_i64, extract_array_i64,
                |a, b| a.cmp(b), Int64, Int64,
                k, largest, sorted, sort_kind
            ),
            DType::Int32 => topk_flat_arm!(
                contig, wrapper, meta,
                extract_view_i32, extract_array_i32,
                |a, b| a.cmp(b), Int32, Int32,
                k, largest, sorted, sort_kind
            ),
            DType::Int16 => topk_flat_arm!(
                contig, wrapper, meta,
                extract_view_i16, extract_array_i16,
                |a, b| a.cmp(b), Int16, Int16,
                k, largest, sorted, sort_kind
            ),
            DType::Int8 => topk_flat_arm!(
                contig, wrapper, meta,
                extract_view_i8, extract_array_i8,
                |a, b| a.cmp(b), Int8, Int8,
                k, largest, sorted, sort_kind
            ),
            DType::Uint64 => topk_flat_arm!(
                contig, wrapper, meta,
                extract_view_u64, extract_array_u64,
                |a, b| a.cmp(b), Uint64, Uint64,
                k, largest, sorted, sort_kind
            ),
            DType::Uint32 => topk_flat_arm!(
                contig, wrapper, meta,
                extract_view_u32, extract_array_u32,
                |a, b| a.cmp(b), Uint32, Uint32,
                k, largest, sorted, sort_kind
            ),
            DType::Uint16 => topk_flat_arm!(
                contig, wrapper, meta,
                extract_view_u16, extract_array_u16,
                |a, b| a.cmp(b), Uint16, Uint16,
                k, largest, sorted, sort_kind
            ),
            DType::Uint8 => topk_flat_arm!(
                contig, wrapper, meta,
                extract_view_u8, extract_array_u8,
                |a, b| a.cmp(b), Uint8, Uint8,
                k, largest, sorted, sort_kind
            ),
            DType::Bool => topk_flat_arm!(
                contig, wrapper, meta,
                extract_view_bool, extract_array_bool,
                |a, b| a.cmp(b), Bool, Bool,
                k, largest, sorted, sort_kind
            ),
            DType::Complex64 | DType::Complex128 => {
                set_last_error("Topk is not supported for complex dtypes".to_string());
                return ERR_DTYPE;
            }
        };

        *out_shape = k;
        *out_values = NdArrayHandle::from_wrapper(Box::new(values_wrapper));
        *out_indices = NdArrayHandle::from_wrapper(Box::new(indices_wrapper));
        SUCCESS
    })
}
