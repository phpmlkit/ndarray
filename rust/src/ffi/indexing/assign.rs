//! Slice assign operations.
//!
//! Provides assign operations between strided array views.

use crate::helpers::error::{set_last_error, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::helpers::view::{
    extract_array_bool, extract_array_c128, extract_array_c64, extract_array_f32,
    extract_array_f64, extract_array_i16, extract_array_i32, extract_array_i64, extract_array_i8,
    extract_array_u16, extract_array_u32, extract_array_u64, extract_array_u8,
    extract_view_mut_bool, extract_view_mut_c128, extract_view_mut_c64, extract_view_mut_f32,
    extract_view_mut_f64, extract_view_mut_i16, extract_view_mut_i32, extract_view_mut_i64,
    extract_view_mut_i8, extract_view_mut_u16, extract_view_mut_u32, extract_view_mut_u64,
    extract_view_mut_u8, rhs_broadcasts_to_lhs,
};
use crate::types::dtype::DType;
use crate::types::{ArrayMetadata, NdArrayHandle};

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
    dst_meta: *const ArrayMetadata,
    src: *const NdArrayHandle,
    src_meta: *const ArrayMetadata,
) -> i32 {
    if dst.is_null() || src.is_null() || dst_meta.is_null() || src_meta.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let dst_wrapper = NdArrayHandle::as_wrapper(dst as *mut _);
        let src_wrapper = NdArrayHandle::as_wrapper(src as *mut _);

        let dst_meta = &*dst_meta;
        let src_meta = &*src_meta;

        // Validate dtypes match
        if dst_wrapper.dtype != src_wrapper.dtype {
            set_last_error(format!(
                "DType mismatch in assign: dst={:?}, src={:?}",
                dst_wrapper.dtype, src_wrapper.dtype
            ));
            return ERR_GENERIC;
        }

        let dst_shape = unsafe { dst_meta.shape_slice() };
        let src_shape = unsafe { src_meta.shape_slice() };
        if !rhs_broadcasts_to_lhs(dst_shape, src_shape) {
            set_last_error(format!(
                "Cannot assign: source shape {:?} cannot broadcast to destination shape {:?}",
                src_shape, dst_shape
            ));
            return ERR_SHAPE;
        }

        match dst_wrapper.dtype {
            DType::Int8 => {
                let src_arr = extract_array_i8(src_wrapper, src_meta).expect("Type mismatch");
                let mut dst_arr =
                    extract_view_mut_i8(dst_wrapper, dst_meta).expect("Type mismatch");
                dst_arr.assign(&src_arr);
            }
            DType::Int16 => {
                let src_arr = extract_array_i16(src_wrapper, src_meta).expect("Type mismatch");
                let mut dst_arr =
                    extract_view_mut_i16(dst_wrapper, dst_meta).expect("Type mismatch");
                dst_arr.assign(&src_arr);
            }
            DType::Int32 => {
                let src_arr = extract_array_i32(src_wrapper, src_meta).expect("Type mismatch");
                let mut dst_arr =
                    extract_view_mut_i32(dst_wrapper, dst_meta).expect("Type mismatch");
                dst_arr.assign(&src_arr);
            }
            DType::Int64 => {
                let src_arr = extract_array_i64(src_wrapper, src_meta).expect("Type mismatch");
                let mut dst_arr =
                    extract_view_mut_i64(dst_wrapper, dst_meta).expect("Type mismatch");
                dst_arr.assign(&src_arr);
            }
            DType::Uint8 => {
                let src_arr = extract_array_u8(src_wrapper, src_meta).expect("Type mismatch");
                let mut dst_arr =
                    extract_view_mut_u8(dst_wrapper, dst_meta).expect("Type mismatch");
                dst_arr.assign(&src_arr);
            }
            DType::Uint16 => {
                let src_arr = extract_array_u16(src_wrapper, src_meta).expect("Type mismatch");
                let mut dst_arr =
                    extract_view_mut_u16(dst_wrapper, dst_meta).expect("Type mismatch");
                dst_arr.assign(&src_arr);
            }
            DType::Uint32 => {
                let src_arr = extract_array_u32(src_wrapper, src_meta).expect("Type mismatch");
                let mut dst_arr =
                    extract_view_mut_u32(dst_wrapper, dst_meta).expect("Type mismatch");
                dst_arr.assign(&src_arr);
            }
            DType::Uint64 => {
                let src_arr = extract_array_u64(src_wrapper, src_meta).expect("Type mismatch");
                let mut dst_arr =
                    extract_view_mut_u64(dst_wrapper, dst_meta).expect("Type mismatch");
                dst_arr.assign(&src_arr);
            }
            DType::Float32 => {
                let src_arr = extract_array_f32(src_wrapper, src_meta).expect("Type mismatch");
                let mut dst_arr =
                    extract_view_mut_f32(dst_wrapper, dst_meta).expect("Type mismatch");
                dst_arr.assign(&src_arr);
            }
            DType::Float64 => {
                let src_arr = extract_array_f64(src_wrapper, src_meta).expect("Type mismatch");
                let mut dst_arr =
                    extract_view_mut_f64(dst_wrapper, dst_meta).expect("Type mismatch");
                dst_arr.assign(&src_arr);
            }
            DType::Complex64 => {
                let src_arr = extract_array_c64(src_wrapper, src_meta).expect("Type mismatch");
                let mut dst_arr =
                    extract_view_mut_c64(dst_wrapper, dst_meta).expect("Type mismatch");
                dst_arr.assign(&src_arr);
            }
            DType::Complex128 => {
                let src_arr = extract_array_c128(src_wrapper, src_meta).expect("Type mismatch");
                let mut dst_arr =
                    extract_view_mut_c128(dst_wrapper, dst_meta).expect("Type mismatch");
                dst_arr.assign(&src_arr);
            }
            DType::Bool => {
                let src_arr = extract_array_bool(src_wrapper, src_meta).expect("Type mismatch");
                let mut dst_arr =
                    extract_view_mut_bool(dst_wrapper, dst_meta).expect("Type mismatch");
                dst_arr.assign(&src_arr);
            }
        }

        SUCCESS
    })
}
