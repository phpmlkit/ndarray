//! Split array along axis at given indices.
//!
//! Returns view metadata (offset, shape, strides) for each part. No new allocations -
//! parts are views into the original. PHP creates NDArray objects with same handle.

use crate::error::{set_last_error, ERR_GENERIC, ERR_INDEX, ERR_SHAPE, SUCCESS};
use crate::ffi::stacking::helpers::resolve_axis;
use crate::ffi::{NdArrayHandle, ViewMetadata};

/// Split array along axis at the given indices.
///
/// indices[i] is the start of part i+1 (exclusive end of part i).
/// Parts: [0..indices[0]), [indices[0]..indices[1]), ..., [indices[n-1]..len].
/// Output count = num_indices + 1.
///
/// Writes to out_offsets (size num_indices+1), out_shapes and out_strides (size (num_indices+1)*ndim each).
#[no_mangle]
pub unsafe extern "C" fn ndarray_split(
    _handle: *const NdArrayHandle,
    meta: *const ViewMetadata,
    axis: i32,
    indices: *const usize,
    num_indices: usize,
    out_offsets: *mut usize,
    out_shapes: *mut usize,
    out_strides: *mut usize,
) -> i32 {
    if meta.is_null()
        || indices.is_null()
        || out_offsets.is_null()
        || out_shapes.is_null()
        || out_strides.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let meta_ref = &*meta;
        let shape_slice = std::slice::from_raw_parts(meta_ref.shape, meta_ref.ndim);
        let strides_slice = std::slice::from_raw_parts(meta_ref.strides, meta_ref.ndim);
        let indices_slice = std::slice::from_raw_parts(indices, num_indices);

        let axis_usize = match resolve_axis(shape_slice, axis) {
            Ok(a) => a,
            Err(e) => {
                set_last_error(e);
                return ERR_SHAPE;
            }
        };

        let axis_len = shape_slice[axis_usize];
        let axis_stride = strides_slice[axis_usize];

        // Validate indices: 0 <= i0 < i1 < ... < i_{n-1} <= axis_len
        let mut prev = 0usize;
        for &idx in indices_slice {
            if idx < prev || idx > axis_len {
                set_last_error(format!(
                    "Split index {} out of bounds for axis length {}",
                    idx, axis_len
                ));
                return ERR_INDEX;
            }
            prev = idx;
        }

        let num_parts = num_indices + 1;

        let mut part_starts = Vec::with_capacity(num_parts + 1);
        part_starts.push(0);
        part_starts.extend_from_slice(indices_slice);
        part_starts.push(axis_len);

        for (i, &start) in part_starts.iter().enumerate().take(num_parts) {
            let end = part_starts[i + 1];
            let part_len = end - start;

            // Offset for this part: base_offset + start * axis_stride
            let part_offset = meta_ref.offset + start * axis_stride;
            *out_offsets.add(i) = part_offset;

            // Shape: same as input but axis dimension = part_len
            for d in 0..meta_ref.ndim {
                let val = if d == axis_usize {
                    part_len
                } else {
                    shape_slice[d]
                };
                *out_shapes.add(i * meta_ref.ndim + d) = val;
            }

            // Strides: same as input (views share strides)
            for d in 0..meta_ref.ndim {
                *out_strides.add(i * meta_ref.ndim + d) = strides_slice[d];
            }
        }

        SUCCESS
    })
}
