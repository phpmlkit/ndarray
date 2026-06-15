//! Type casting operations for converting arrays between dtypes.
//!
//! Provides astype() functionality to copy arrays with type conversion.

use crate::helpers::error::{set_last_error, ERR_GENERIC, SUCCESS};
use crate::helpers::{
    extract_array_as_bool, extract_array_as_c128, extract_array_as_c64, extract_array_as_f32,
    extract_array_as_f64, extract_array_as_i16, extract_array_as_i32, extract_array_as_i64,
    extract_array_as_i8, extract_array_as_u16, extract_array_as_u32, extract_array_as_u64,
    extract_array_as_u8, extract_array_bool, extract_array_c128, extract_array_c64,
    extract_array_f32, extract_array_f64, extract_array_i16, extract_array_i32, extract_array_i64,
    extract_array_i8, extract_array_u16, extract_array_u32, extract_array_u64, extract_array_u8,
};
use crate::types::dtype::DType;
use crate::types::{ArrayData, ArrayMetadata, NDArrayWrapper, NdArrayHandle};

use parking_lot::RwLock;
use std::sync::Arc;

/// Cast an NDArray to a different dtype.
#[no_mangle]
pub unsafe extern "C" fn ndarray_astype(
    handle: *const NdArrayHandle,
    meta: *const ArrayMetadata,
    target_dtype: i32,
    out: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || out.is_null() || meta.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let meta = &*meta;

        let target = match DType::from_u8(target_dtype as u8) {
            Some(dt) => dt,
            None => {
                set_last_error(format!("Invalid target dtype: {}", target_dtype));
                return ERR_GENERIC;
            }
        };

        let result_wrapper = if wrapper.dtype == target {
            match wrapper.dtype {
                DType::Float64 => {
                    let Some(arr) = extract_array_f64(wrapper, meta) else {
                        set_last_error("Failed to extract Float64 view".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Float64(Arc::new(RwLock::new(arr.to_owned()))),
                        dtype: DType::Float64,
                    }
                }
                DType::Float32 => {
                    let Some(arr) = extract_array_f32(wrapper, meta) else {
                        set_last_error("Failed to extract Float32 view".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Float32(Arc::new(RwLock::new(arr.to_owned()))),
                        dtype: DType::Float32,
                    }
                }
                DType::Int64 => {
                    let Some(arr) = extract_array_i64(wrapper, meta) else {
                        set_last_error("Failed to extract Int64 view".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Int64(Arc::new(RwLock::new(arr.to_owned()))),
                        dtype: DType::Int64,
                    }
                }
                DType::Int32 => {
                    let Some(arr) = extract_array_i32(wrapper, meta) else {
                        set_last_error("Failed to extract Int32 view".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Int32(Arc::new(RwLock::new(arr.to_owned()))),
                        dtype: DType::Int32,
                    }
                }
                DType::Int16 => {
                    let Some(arr) = extract_array_i16(wrapper, meta) else {
                        set_last_error("Failed to extract Int16 view".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Int16(Arc::new(RwLock::new(arr.to_owned()))),
                        dtype: DType::Int16,
                    }
                }
                DType::Int8 => {
                    let Some(arr) = extract_array_i8(wrapper, meta) else {
                        set_last_error("Failed to extract Int8 view".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Int8(Arc::new(RwLock::new(arr.to_owned()))),
                        dtype: DType::Int8,
                    }
                }
                DType::Uint64 => {
                    let Some(arr) = extract_array_u64(wrapper, meta) else {
                        set_last_error("Failed to extract Uint64 view".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Uint64(Arc::new(RwLock::new(arr.to_owned()))),
                        dtype: DType::Uint64,
                    }
                }
                DType::Uint32 => {
                    let Some(arr) = extract_array_u32(wrapper, meta) else {
                        set_last_error("Failed to extract Uint32 view".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Uint32(Arc::new(RwLock::new(arr.to_owned()))),
                        dtype: DType::Uint32,
                    }
                }
                DType::Uint16 => {
                    let Some(arr) = extract_array_u16(wrapper, meta) else {
                        set_last_error("Failed to extract Uint16 view".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Uint16(Arc::new(RwLock::new(arr.to_owned()))),
                        dtype: DType::Uint16,
                    }
                }
                DType::Uint8 => {
                    let Some(arr) = extract_array_u8(wrapper, meta) else {
                        set_last_error("Failed to extract Uint8 view".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Uint8(Arc::new(RwLock::new(arr.to_owned()))),
                        dtype: DType::Uint8,
                    }
                }
                DType::Bool => {
                    let Some(arr) = extract_array_bool(wrapper, meta) else {
                        set_last_error("Failed to extract Bool view".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Bool(Arc::new(RwLock::new(arr.to_owned()))),
                        dtype: DType::Bool,
                    }
                }
                DType::Complex64 => {
                    let Some(arr) = extract_array_c64(wrapper, meta) else {
                        set_last_error("Failed to extract Complex64 view".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Complex64(Arc::new(RwLock::new(arr.to_owned()))),
                        dtype: DType::Complex64,
                    }
                }
                DType::Complex128 => {
                    let Some(arr) = extract_array_c128(wrapper, meta) else {
                        set_last_error("Failed to extract Complex128 view".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Complex128(Arc::new(RwLock::new(arr.to_owned()))),
                        dtype: DType::Complex128,
                    }
                }
            }
        } else {
            match target {
                DType::Float64 => {
                    let Some(arr) = extract_array_as_f64(wrapper, meta) else {
                        set_last_error("Failed to cast array to Float64".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Float64(Arc::new(RwLock::new(arr))),
                        dtype: DType::Float64,
                    }
                }
                DType::Float32 => {
                    let Some(arr) = extract_array_as_f32(wrapper, meta) else {
                        set_last_error("Failed to cast array to Float32".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Float32(Arc::new(RwLock::new(arr))),
                        dtype: DType::Float32,
                    }
                }
                DType::Int64 => {
                    let Some(arr) = extract_array_as_i64(wrapper, meta) else {
                        set_last_error("Failed to cast array to Int64".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Int64(Arc::new(RwLock::new(arr))),
                        dtype: DType::Int64,
                    }
                }
                DType::Int32 => {
                    let Some(arr) = extract_array_as_i32(wrapper, meta) else {
                        set_last_error("Failed to cast array to Int32".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Int32(Arc::new(RwLock::new(arr))),
                        dtype: DType::Int32,
                    }
                }
                DType::Int16 => {
                    let Some(arr) = extract_array_as_i16(wrapper, meta) else {
                        set_last_error("Failed to cast array to Int16".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Int16(Arc::new(RwLock::new(arr))),
                        dtype: DType::Int16,
                    }
                }
                DType::Int8 => {
                    let Some(arr) = extract_array_as_i8(wrapper, meta) else {
                        set_last_error("Failed to cast array to Int8".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Int8(Arc::new(RwLock::new(arr))),
                        dtype: DType::Int8,
                    }
                }
                DType::Uint64 => {
                    let Some(arr) = extract_array_as_u64(wrapper, meta) else {
                        set_last_error("Failed to cast array to Uint64".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Uint64(Arc::new(RwLock::new(arr))),
                        dtype: DType::Uint64,
                    }
                }
                DType::Uint32 => {
                    let Some(arr) = extract_array_as_u32(wrapper, meta) else {
                        set_last_error("Failed to cast array to Uint32".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Uint32(Arc::new(RwLock::new(arr))),
                        dtype: DType::Uint32,
                    }
                }
                DType::Uint16 => {
                    let Some(arr) = extract_array_as_u16(wrapper, meta) else {
                        set_last_error("Failed to cast array to Uint16".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Uint16(Arc::new(RwLock::new(arr))),
                        dtype: DType::Uint16,
                    }
                }
                DType::Uint8 => {
                    let Some(arr) = extract_array_as_u8(wrapper, meta) else {
                        set_last_error("Failed to cast array to Uint8".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Uint8(Arc::new(RwLock::new(arr))),
                        dtype: DType::Uint8,
                    }
                }
                DType::Bool => {
                    let Some(arr) = extract_array_as_bool(wrapper, meta) else {
                        set_last_error("Failed to cast array to Bool".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Bool(Arc::new(RwLock::new(arr))),
                        dtype: DType::Bool,
                    }
                }
                DType::Complex64 => {
                    let Some(arr) = extract_array_as_c64(wrapper, meta) else {
                        set_last_error("Failed to cast array to Complex64".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Complex64(Arc::new(RwLock::new(arr))),
                        dtype: DType::Complex64,
                    }
                }
                DType::Complex128 => {
                    let Some(arr) = extract_array_as_c128(wrapper, meta) else {
                        set_last_error("Failed to cast array to Complex128".to_string());
                        return ERR_GENERIC;
                    };
                    NDArrayWrapper {
                        data: ArrayData::Complex128(Arc::new(RwLock::new(arr))),
                        dtype: DType::Complex128,
                    }
                }
            }
        };

        *out = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
