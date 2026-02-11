//! Slice assign operations.
//!
//! Provides type-specific assign operations between strided array views.

use crate::core::NDArrayWrapper;
use crate::error::{self, ERR_GENERIC, SUCCESS};
use crate::ffi::NdArrayHandle;
use std::slice;

/// Helper for assigning slice values.
pub unsafe fn assign_slice_helper<F>(
    dst_handle: *const NdArrayHandle,
    dst_offset: usize,
    dst_shape_ptr: *const usize,
    dst_strides_ptr: *const usize,
    src_handle: *const NdArrayHandle,
    src_offset: usize,
    src_shape_ptr: *const usize,
    src_strides_ptr: *const usize,
    ndim: usize,
    method: F,
) -> i32
where
    F: Fn(
            &NDArrayWrapper,
            usize,
            &[usize],
            &[usize],
            &NDArrayWrapper,
            usize,
            &[usize],
            &[usize],
        ) -> Result<(), String>
        + std::panic::UnwindSafe,
{
    if dst_handle.is_null()
        || src_handle.is_null()
        || dst_shape_ptr.is_null()
        || dst_strides_ptr.is_null()
        || src_shape_ptr.is_null()
        || src_strides_ptr.is_null()
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let dst_wrapper = NdArrayHandle::as_wrapper(dst_handle as *mut _);
        let src_wrapper = NdArrayHandle::as_wrapper(src_handle as *mut _);

        let dst_shape = slice::from_raw_parts(dst_shape_ptr, ndim);
        let dst_strides = slice::from_raw_parts(dst_strides_ptr, ndim);

        let src_shape = slice::from_raw_parts(src_shape_ptr, ndim);
        let src_strides = slice::from_raw_parts(src_strides_ptr, ndim);

        match method(
            dst_wrapper,
            dst_offset,
            dst_shape,
            dst_strides,
            src_wrapper,
            src_offset,
            src_shape,
            src_strides,
        ) {
            Ok(()) => SUCCESS,
            Err(e) => {
                error::set_last_error(e);
                ERR_GENERIC
            }
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn ndarray_assign_int8(
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
    assign_slice_helper(
        dst,
        doff,
        dshp,
        dstr,
        src,
        soff,
        sshp,
        sstr,
        ndim,
        |d, do_, ds, dst_, s, so, ss, sst| d.assign_slice_i8(do_, ds, dst_, s, so, ss, sst),
    )
}

#[no_mangle]
pub unsafe extern "C" fn ndarray_assign_int16(
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
    assign_slice_helper(
        dst,
        doff,
        dshp,
        dstr,
        src,
        soff,
        sshp,
        sstr,
        ndim,
        |d, do_, ds, dst_, s, so, ss, sst| d.assign_slice_i16(do_, ds, dst_, s, so, ss, sst),
    )
}

#[no_mangle]
pub unsafe extern "C" fn ndarray_assign_int32(
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
    assign_slice_helper(
        dst,
        doff,
        dshp,
        dstr,
        src,
        soff,
        sshp,
        sstr,
        ndim,
        |d, do_, ds, dst_, s, so, ss, sst| d.assign_slice_i32(do_, ds, dst_, s, so, ss, sst),
    )
}

#[no_mangle]
pub unsafe extern "C" fn ndarray_assign_int64(
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
    assign_slice_helper(
        dst,
        doff,
        dshp,
        dstr,
        src,
        soff,
        sshp,
        sstr,
        ndim,
        |d, do_, ds, dst_, s, so, ss, sst| d.assign_slice_i64(do_, ds, dst_, s, so, ss, sst),
    )
}

#[no_mangle]
pub unsafe extern "C" fn ndarray_assign_uint8(
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
    assign_slice_helper(
        dst,
        doff,
        dshp,
        dstr,
        src,
        soff,
        sshp,
        sstr,
        ndim,
        |d, do_, ds, dst_, s, so, ss, sst| d.assign_slice_u8(do_, ds, dst_, s, so, ss, sst),
    )
}

#[no_mangle]
pub unsafe extern "C" fn ndarray_assign_uint16(
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
    assign_slice_helper(
        dst,
        doff,
        dshp,
        dstr,
        src,
        soff,
        sshp,
        sstr,
        ndim,
        |d, do_, ds, dst_, s, so, ss, sst| d.assign_slice_u16(do_, ds, dst_, s, so, ss, sst),
    )
}

#[no_mangle]
pub unsafe extern "C" fn ndarray_assign_uint32(
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
    assign_slice_helper(
        dst,
        doff,
        dshp,
        dstr,
        src,
        soff,
        sshp,
        sstr,
        ndim,
        |d, do_, ds, dst_, s, so, ss, sst| d.assign_slice_u32(do_, ds, dst_, s, so, ss, sst),
    )
}

#[no_mangle]
pub unsafe extern "C" fn ndarray_assign_uint64(
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
    assign_slice_helper(
        dst,
        doff,
        dshp,
        dstr,
        src,
        soff,
        sshp,
        sstr,
        ndim,
        |d, do_, ds, dst_, s, so, ss, sst| d.assign_slice_u64(do_, ds, dst_, s, so, ss, sst),
    )
}

#[no_mangle]
pub unsafe extern "C" fn ndarray_assign_float32(
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
    assign_slice_helper(
        dst,
        doff,
        dshp,
        dstr,
        src,
        soff,
        sshp,
        sstr,
        ndim,
        |d, do_, ds, dst_, s, so, ss, sst| d.assign_slice_f32(do_, ds, dst_, s, so, ss, sst),
    )
}

#[no_mangle]
pub unsafe extern "C" fn ndarray_assign_float64(
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
    assign_slice_helper(
        dst,
        doff,
        dshp,
        dstr,
        src,
        soff,
        sshp,
        sstr,
        ndim,
        |d, do_, ds, dst_, s, so, ss, sst| d.assign_slice_f64(do_, ds, dst_, s, so, ss, sst),
    )
}

#[no_mangle]
pub unsafe extern "C" fn ndarray_assign_bool(
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
    assign_slice_helper(
        dst,
        doff,
        dshp,
        dstr,
        src,
        soff,
        sshp,
        sstr,
        ndim,
        |d, do_, ds, dst_, s, so, ss, sst| d.assign_slice_bool(do_, ds, dst_, s, so, ss, sst),
    )
}
