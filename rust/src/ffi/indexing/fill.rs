//! Slice fill operations.
//!
//! Provides unified fill operations for strided array views.

use std::ffi::c_void;

use crate::helpers::view::{
    extract_view_mut_bool, extract_view_mut_c128, extract_view_mut_c64, extract_view_mut_f32,
    extract_view_mut_f64, extract_view_mut_i16, extract_view_mut_i32, extract_view_mut_i64,
    extract_view_mut_i8, extract_view_mut_u16, extract_view_mut_u32, extract_view_mut_u64,
    extract_view_mut_u8,
};
use crate::types::dtype::DType;
use crate::types::{ArrayMetadata, NdArrayHandle};
use num_complex::Complex;

/// Fill a slice with a value.
///
/// # Arguments
/// * `handle` - Array handle
/// * `meta` - View metadata
/// * `value` - Pointer to the fill value (type depends on array dtype)
#[no_mangle]
pub unsafe extern "C" fn ndarray_fill(
    handle: *const NdArrayHandle,
    meta: *const ArrayMetadata,
    value: *const c_void,
) -> i32 {
    if handle.is_null() || value.is_null() || meta.is_null() {
        return crate::helpers::error::ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let meta = &*meta;

        match wrapper.dtype {
            DType::Int8 => {
                let v = *(value as *const i8);
                extract_view_mut_i8(wrapper, meta)
                    .expect("Type mismatch")
                    .fill(v);
            }
            DType::Int16 => {
                let v = *(value as *const i16);
                extract_view_mut_i16(wrapper, meta)
                    .expect("Type mismatch")
                    .fill(v);
            }
            DType::Int32 => {
                let v = *(value as *const i32);
                extract_view_mut_i32(wrapper, meta)
                    .expect("Type mismatch")
                    .fill(v);
            }
            DType::Int64 => {
                let v = *(value as *const i64);
                extract_view_mut_i64(wrapper, meta)
                    .expect("Type mismatch")
                    .fill(v);
            }
            DType::Uint8 => {
                let v = *(value as *const u8);
                extract_view_mut_u8(wrapper, meta)
                    .expect("Type mismatch")
                    .fill(v);
            }
            DType::Uint16 => {
                let v = *(value as *const u16);
                extract_view_mut_u16(wrapper, meta)
                    .expect("Type mismatch")
                    .fill(v);
            }
            DType::Uint32 => {
                let v = *(value as *const u32);
                extract_view_mut_u32(wrapper, meta)
                    .expect("Type mismatch")
                    .fill(v);
            }
            DType::Uint64 => {
                let v = *(value as *const u64);
                extract_view_mut_u64(wrapper, meta)
                    .expect("Type mismatch")
                    .fill(v);
            }
            DType::Float32 => {
                let v = *(value as *const f32);
                extract_view_mut_f32(wrapper, meta)
                    .expect("Type mismatch")
                    .fill(v);
            }
            DType::Float64 => {
                let v = *(value as *const f64);
                extract_view_mut_f64(wrapper, meta)
                    .expect("Type mismatch")
                    .fill(v);
            }
            DType::Complex64 => {
                let v = *(value as *const Complex<f32>);
                extract_view_mut_c64(wrapper, meta)
                    .expect("Type mismatch")
                    .fill(v);
            }
            DType::Complex128 => {
                let v = *(value as *const Complex<f64>);
                extract_view_mut_c128(wrapper, meta)
                    .expect("Type mismatch")
                    .fill(v);
            }
            DType::Bool => {
                let v = *(value as *const u8);
                extract_view_mut_bool(wrapper, meta)
                    .expect("Type mismatch")
                    .fill(v);
            }
        }

        crate::helpers::error::SUCCESS
    })
}
