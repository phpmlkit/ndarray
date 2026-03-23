//! Array copy FFI function.

use crate::helpers::view::{
    extract_view_bool, extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32,
    extract_view_i64, extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64,
    extract_view_u8,
};
use crate::types::dtype::DType;
use crate::types::{ArrayData, ArrayMetadata, NDArrayWrapper, NdArrayHandle};
use parking_lot::RwLock;
use std::sync::Arc;

/// Create a deep copy of an array view.
#[no_mangle]
pub unsafe extern "C" fn ndarray_copy(
    handle: *const NdArrayHandle,
    meta: *const ArrayMetadata,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || meta.is_null() || out_handle.is_null() {
        return crate::helpers::error::ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let meta = &*meta;

        let new_wrapper = match wrapper.dtype {
            DType::Int8 => {
                let view = extract_view_i8(wrapper, meta).expect("Type mismatch");
                NDArrayWrapper {
                    data: ArrayData::Int8(Arc::new(RwLock::new(view.to_owned()))),
                    dtype: DType::Int8,
                }
            }
            DType::Int16 => {
                let view = extract_view_i16(wrapper, meta).expect("Type mismatch");
                NDArrayWrapper {
                    data: ArrayData::Int16(Arc::new(RwLock::new(view.to_owned()))),
                    dtype: DType::Int16,
                }
            }
            DType::Int32 => {
                let view = extract_view_i32(wrapper, meta).expect("Type mismatch");
                NDArrayWrapper {
                    data: ArrayData::Int32(Arc::new(RwLock::new(view.to_owned()))),
                    dtype: DType::Int32,
                }
            }
            DType::Int64 => {
                let view = extract_view_i64(wrapper, meta).expect("Type mismatch");
                NDArrayWrapper {
                    data: ArrayData::Int64(Arc::new(RwLock::new(view.to_owned()))),
                    dtype: DType::Int64,
                }
            }
            DType::Uint8 => {
                let view = extract_view_u8(wrapper, meta).expect("Type mismatch");
                NDArrayWrapper {
                    data: ArrayData::Uint8(Arc::new(RwLock::new(view.to_owned()))),
                    dtype: DType::Uint8,
                }
            }
            DType::Uint16 => {
                let view = extract_view_u16(wrapper, meta).expect("Type mismatch");
                NDArrayWrapper {
                    data: ArrayData::Uint16(Arc::new(RwLock::new(view.to_owned()))),
                    dtype: DType::Uint16,
                }
            }
            DType::Uint32 => {
                let view = extract_view_u32(wrapper, meta).expect("Type mismatch");
                NDArrayWrapper {
                    data: ArrayData::Uint32(Arc::new(RwLock::new(view.to_owned()))),
                    dtype: DType::Uint32,
                }
            }
            DType::Uint64 => {
                let view = extract_view_u64(wrapper, meta).expect("Type mismatch");
                NDArrayWrapper {
                    data: ArrayData::Uint64(Arc::new(RwLock::new(view.to_owned()))),
                    dtype: DType::Uint64,
                }
            }
            DType::Float32 => {
                let view = extract_view_f32(wrapper, meta).expect("Type mismatch");
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(view.to_owned()))),
                    dtype: DType::Float32,
                }
            }
            DType::Float64 => {
                let view = extract_view_f64(wrapper, meta).expect("Type mismatch");
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(view.to_owned()))),
                    dtype: DType::Float64,
                }
            }
            DType::Bool => {
                let view = extract_view_bool(wrapper, meta).expect("Type mismatch");
                NDArrayWrapper {
                    data: ArrayData::Bool(Arc::new(RwLock::new(view.to_owned()))),
                    dtype: DType::Bool,
                }
            }
        };

        *out_handle = NdArrayHandle::from_wrapper(Box::new(new_wrapper));
        crate::helpers::error::SUCCESS
    })
}
