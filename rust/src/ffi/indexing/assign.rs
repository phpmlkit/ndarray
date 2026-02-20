//! Slice assign operations.
//!
//! Provides assign operations between strided array views.

use crate::dtype::DType;
use crate::error::{self, ERR_GENERIC, SUCCESS};
use crate::ffi::{NdArrayHandle, ViewMetadata};

/// Assign values from source view to destination view.
///
/// # Arguments
/// * `dst` - Destination array handle
/// * `dst_meta` - Destination view metadata
/// * `src` - Source array handle
/// * `src_meta` - Source view metadata
#[no_mangle]
pub unsafe extern "C" fn ndarray_assign(
    dst: *const NdArrayHandle,
    dst_meta: *const ViewMetadata,
    src: *const NdArrayHandle,
    src_meta: *const ViewMetadata,
) -> i32 {
    if dst.is_null() || src.is_null() || dst_meta.is_null() || src_meta.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let dst_wrapper = NdArrayHandle::as_wrapper(dst as *mut _);
        let src_wrapper = NdArrayHandle::as_wrapper(src as *mut _);

        let dst_meta_ref = &*dst_meta;
        let src_meta_ref = &*src_meta;

        let dst_shape = std::slice::from_raw_parts(dst_meta_ref.shape, dst_meta_ref.ndim);
        let dst_strides = std::slice::from_raw_parts(dst_meta_ref.strides, dst_meta_ref.ndim);

        let src_shape = std::slice::from_raw_parts(src_meta_ref.shape, src_meta_ref.ndim);
        let src_strides = std::slice::from_raw_parts(src_meta_ref.strides, src_meta_ref.ndim);

        // Use destination dtype to determine which assign method to use
        let result = match dst_wrapper.dtype {
            DType::Int8 => dst_wrapper.assign_slice_i8(
                dst_meta_ref.offset,
                dst_shape,
                dst_strides,
                src_wrapper,
                src_meta_ref.offset,
                src_shape,
                src_strides,
            ),
            DType::Int16 => dst_wrapper.assign_slice_i16(
                dst_meta_ref.offset,
                dst_shape,
                dst_strides,
                src_wrapper,
                src_meta_ref.offset,
                src_shape,
                src_strides,
            ),
            DType::Int32 => dst_wrapper.assign_slice_i32(
                dst_meta_ref.offset,
                dst_shape,
                dst_strides,
                src_wrapper,
                src_meta_ref.offset,
                src_shape,
                src_strides,
            ),
            DType::Int64 => dst_wrapper.assign_slice_i64(
                dst_meta_ref.offset,
                dst_shape,
                dst_strides,
                src_wrapper,
                src_meta_ref.offset,
                src_shape,
                src_strides,
            ),
            DType::Uint8 => dst_wrapper.assign_slice_u8(
                dst_meta_ref.offset,
                dst_shape,
                dst_strides,
                src_wrapper,
                src_meta_ref.offset,
                src_shape,
                src_strides,
            ),
            DType::Uint16 => dst_wrapper.assign_slice_u16(
                dst_meta_ref.offset,
                dst_shape,
                dst_strides,
                src_wrapper,
                src_meta_ref.offset,
                src_shape,
                src_strides,
            ),
            DType::Uint32 => dst_wrapper.assign_slice_u32(
                dst_meta_ref.offset,
                dst_shape,
                dst_strides,
                src_wrapper,
                src_meta_ref.offset,
                src_shape,
                src_strides,
            ),
            DType::Uint64 => dst_wrapper.assign_slice_u64(
                dst_meta_ref.offset,
                dst_shape,
                dst_strides,
                src_wrapper,
                src_meta_ref.offset,
                src_shape,
                src_strides,
            ),
            DType::Float32 => dst_wrapper.assign_slice_f32(
                dst_meta_ref.offset,
                dst_shape,
                dst_strides,
                src_wrapper,
                src_meta_ref.offset,
                src_shape,
                src_strides,
            ),
            DType::Float64 => dst_wrapper.assign_slice_f64(
                dst_meta_ref.offset,
                dst_shape,
                dst_strides,
                src_wrapper,
                src_meta_ref.offset,
                src_shape,
                src_strides,
            ),
            DType::Bool => dst_wrapper.assign_slice_bool(
                dst_meta_ref.offset,
                dst_shape,
                dst_strides,
                src_wrapper,
                src_meta_ref.offset,
                src_shape,
                src_strides,
            ),
        };

        match result {
            Ok(()) => SUCCESS,
            Err(e) => {
                error::set_last_error(e);
                ERR_GENERIC
            }
        }
    })
}
