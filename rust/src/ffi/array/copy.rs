//! Array copy FFI function.

use crate::core::dtype::DType;
use crate::core::error::{self, ERR_GENERIC, SUCCESS};
use crate::ffi::{ArrayMetadata, NdArrayHandle};

/// Create a deep copy of an array view.
#[no_mangle]
pub unsafe extern "C" fn ndarray_copy(
    handle: *const NdArrayHandle,
    meta: *const ArrayMetadata,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || meta.is_null() || out_handle.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let meta = &*meta;

        let result = match wrapper.dtype {
            DType::Int8 => wrapper.copy_view_i8(meta),
            DType::Int16 => wrapper.copy_view_i16(meta),
            DType::Int32 => wrapper.copy_view_i32(meta),
            DType::Int64 => wrapper.copy_view_i64(meta),
            DType::Uint8 => wrapper.copy_view_u8(meta),
            DType::Uint16 => wrapper.copy_view_u16(meta),
            DType::Uint32 => wrapper.copy_view_u32(meta),
            DType::Uint64 => wrapper.copy_view_u64(meta),
            DType::Float32 => wrapper.copy_view_f32(meta),
            DType::Float64 => wrapper.copy_view_f64(meta),
            DType::Bool => wrapper.copy_view_bool(meta),
        };

        match result {
            Ok(new_wrapper) => {
                *out_handle = NdArrayHandle::from_wrapper(Box::new(new_wrapper));
                SUCCESS
            }
            Err(e) => {
                error::set_last_error(e);
                ERR_GENERIC
            }
        }
    })
}
