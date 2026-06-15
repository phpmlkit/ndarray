//! Array copy FFI function.

use crate::helpers::view::{
    extract_array_bool, extract_array_c128, extract_array_c64, extract_array_f32,
    extract_array_f64, extract_array_i16, extract_array_i32, extract_array_i64, extract_array_i8,
    extract_array_u16, extract_array_u32, extract_array_u64, extract_array_u8,
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
                let arr = extract_array_i8(wrapper, meta).expect("Type mismatch");
                NDArrayWrapper {
                    data: ArrayData::Int8(Arc::new(RwLock::new(arr))),
                    dtype: DType::Int8,
                }
            }
            DType::Int16 => {
                let arr = extract_array_i16(wrapper, meta).expect("Type mismatch");
                NDArrayWrapper {
                    data: ArrayData::Int16(Arc::new(RwLock::new(arr))),
                    dtype: DType::Int16,
                }
            }
            DType::Int32 => {
                let arr = extract_array_i32(wrapper, meta).expect("Type mismatch");
                NDArrayWrapper {
                    data: ArrayData::Int32(Arc::new(RwLock::new(arr))),
                    dtype: DType::Int32,
                }
            }
            DType::Int64 => {
                let arr = extract_array_i64(wrapper, meta).expect("Type mismatch");
                NDArrayWrapper {
                    data: ArrayData::Int64(Arc::new(RwLock::new(arr))),
                    dtype: DType::Int64,
                }
            }
            DType::Uint8 => {
                let arr = extract_array_u8(wrapper, meta).expect("Type mismatch");
                NDArrayWrapper {
                    data: ArrayData::Uint8(Arc::new(RwLock::new(arr))),
                    dtype: DType::Uint8,
                }
            }
            DType::Uint16 => {
                let arr = extract_array_u16(wrapper, meta).expect("Type mismatch");
                NDArrayWrapper {
                    data: ArrayData::Uint16(Arc::new(RwLock::new(arr))),
                    dtype: DType::Uint16,
                }
            }
            DType::Uint32 => {
                let arr = extract_array_u32(wrapper, meta).expect("Type mismatch");
                NDArrayWrapper {
                    data: ArrayData::Uint32(Arc::new(RwLock::new(arr))),
                    dtype: DType::Uint32,
                }
            }
            DType::Uint64 => {
                let arr = extract_array_u64(wrapper, meta).expect("Type mismatch");
                NDArrayWrapper {
                    data: ArrayData::Uint64(Arc::new(RwLock::new(arr))),
                    dtype: DType::Uint64,
                }
            }
            DType::Float32 => {
                let arr = extract_array_f32(wrapper, meta).expect("Type mismatch");
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(arr))),
                    dtype: DType::Float32,
                }
            }
            DType::Float64 => {
                let arr = extract_array_f64(wrapper, meta).expect("Type mismatch");
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(arr))),
                    dtype: DType::Float64,
                }
            }
            DType::Bool => {
                let arr = extract_array_bool(wrapper, meta).expect("Type mismatch");
                NDArrayWrapper {
                    data: ArrayData::Bool(Arc::new(RwLock::new(arr))),
                    dtype: DType::Bool,
                }
            }
            DType::Complex64 => {
                let arr = extract_array_c64(wrapper, meta).expect("Type mismatch");
                NDArrayWrapper {
                    data: ArrayData::Complex64(Arc::new(RwLock::new(arr))),
                    dtype: DType::Complex64,
                }
            }
            DType::Complex128 => {
                let arr = extract_array_c128(wrapper, meta).expect("Type mismatch");
                NDArrayWrapper {
                    data: ArrayData::Complex128(Arc::new(RwLock::new(arr))),
                    dtype: DType::Complex128,
                }
            }
        };

        *out_handle = NdArrayHandle::from_wrapper(Box::new(new_wrapper));
        crate::helpers::error::SUCCESS
    })
}
