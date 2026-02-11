//! Slice assign operations.
//!
//! Provides assign operations between strided array views.

use crate::dtype::DType;
use crate::error::{self, ERR_GENERIC, SUCCESS};
use crate::ffi::NdArrayHandle;
use std::slice;

/// Assign values from source view to destination view.
///
/// # Arguments
/// * `dst` - Destination array handle
/// * `doff` - Destination offset
/// * `dshp` - Destination shape pointer
/// * `dstr` - Destination strides pointer
/// * `src` - Source array handle
/// * `soff` - Source offset
/// * `sshp` - Source shape pointer
/// * `sstr` - Source strides pointer
/// * `ndim` - Number of dimensions
#[no_mangle]
pub unsafe extern "C" fn ndarray_assign(
    dst: *const NdArrayHandle,
    doff: usize,
    dshp: *const usize,
    dstr: *const usize,
    src: *const NdArrayHandle,
    soff: usize,
    sshp: *const usize,
    sstr: *const usize,
    ndim: usize,
) -> i32 {
    if dst.is_null()
        || src.is_null()
        || dshp.is_null()
        || dstr.is_null()
        || sshp.is_null()
        || sstr.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let dst_wrapper = NdArrayHandle::as_wrapper(dst as *mut _);
        let src_wrapper = NdArrayHandle::as_wrapper(src as *mut _);

        let dst_shape = slice::from_raw_parts(dshp, ndim);
        let dst_strides = slice::from_raw_parts(dstr, ndim);

        let src_shape = slice::from_raw_parts(sshp, ndim);
        let src_strides = slice::from_raw_parts(sstr, ndim);

        // Use destination dtype to determine which assign method to use
        let result = match dst_wrapper.dtype {
            DType::Int8 => dst_wrapper.assign_slice_i8(
                doff,
                dst_shape,
                dst_strides,
                src_wrapper,
                soff,
                src_shape,
                src_strides,
            ),
            DType::Int16 => dst_wrapper.assign_slice_i16(
                doff,
                dst_shape,
                dst_strides,
                src_wrapper,
                soff,
                src_shape,
                src_strides,
            ),
            DType::Int32 => dst_wrapper.assign_slice_i32(
                doff,
                dst_shape,
                dst_strides,
                src_wrapper,
                soff,
                src_shape,
                src_strides,
            ),
            DType::Int64 => dst_wrapper.assign_slice_i64(
                doff,
                dst_shape,
                dst_strides,
                src_wrapper,
                soff,
                src_shape,
                src_strides,
            ),
            DType::Uint8 => dst_wrapper.assign_slice_u8(
                doff,
                dst_shape,
                dst_strides,
                src_wrapper,
                soff,
                src_shape,
                src_strides,
            ),
            DType::Uint16 => dst_wrapper.assign_slice_u16(
                doff,
                dst_shape,
                dst_strides,
                src_wrapper,
                soff,
                src_shape,
                src_strides,
            ),
            DType::Uint32 => dst_wrapper.assign_slice_u32(
                doff,
                dst_shape,
                dst_strides,
                src_wrapper,
                soff,
                src_shape,
                src_strides,
            ),
            DType::Uint64 => dst_wrapper.assign_slice_u64(
                doff,
                dst_shape,
                dst_strides,
                src_wrapper,
                soff,
                src_shape,
                src_strides,
            ),
            DType::Float32 => dst_wrapper.assign_slice_f32(
                doff,
                dst_shape,
                dst_strides,
                src_wrapper,
                soff,
                src_shape,
                src_strides,
            ),
            DType::Float64 => dst_wrapper.assign_slice_f64(
                doff,
                dst_shape,
                dst_strides,
                src_wrapper,
                soff,
                src_shape,
                src_strides,
            ),
            DType::Bool => dst_wrapper.assign_slice_bool(
                doff,
                dst_shape,
                dst_strides,
                src_wrapper,
                soff,
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
