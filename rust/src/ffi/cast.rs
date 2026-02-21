//! Type casting operations for converting arrays between dtypes.
//!
//! Provides astype() functionality to copy arrays with type conversion.

use crate::core::view_helpers::{
    extract_view_as_bool, extract_view_as_f32, extract_view_as_f64, extract_view_as_i16,
    extract_view_as_i32, extract_view_as_i64, extract_view_as_i8, extract_view_as_u16,
    extract_view_as_u32, extract_view_as_u64, extract_view_as_u8, extract_view_bool,
    extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32, extract_view_i64,
    extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64, extract_view_u8,
};
use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{ERR_GENERIC, SUCCESS};
use crate::ffi::{NdArrayHandle, ViewMetadata};

use parking_lot::RwLock;
use std::sync::Arc;

/// Cast an NDArray to a different dtype.
///
#[no_mangle]
pub unsafe extern "C" fn ndarray_astype(
    handle: *const NdArrayHandle,
    meta: *const ViewMetadata,
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
                crate::error::set_last_error(format!("Invalid target dtype: {}", target_dtype));
                return ERR_GENERIC;
            }
        };

        let result_wrapper = if wrapper.dtype == target {
            copy_same_dtype(wrapper, meta)
        } else {
            cast_to_dtype(wrapper, meta, target)
        };

        *out = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}

/// Copy array data when source and target dtypes are the same.
fn copy_same_dtype(wrapper: &NDArrayWrapper, meta: &ViewMetadata) -> NDArrayWrapper {
    unsafe {
        match &wrapper.data {
            ArrayData::Float64(_) => {
                let view = extract_view_f64(wrapper, meta).unwrap();
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(view.to_owned()))),
                    dtype: DType::Float64,
                }
            }
            ArrayData::Float32(_) => {
                let view = extract_view_f32(wrapper, meta).unwrap();
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(view.to_owned()))),
                    dtype: DType::Float32,
                }
            }
            ArrayData::Int64(_) => {
                let view = extract_view_i64(wrapper, meta).unwrap();
                NDArrayWrapper {
                    data: ArrayData::Int64(Arc::new(RwLock::new(view.to_owned()))),
                    dtype: DType::Int64,
                }
            }
            ArrayData::Int32(_) => {
                let view = extract_view_i32(wrapper, meta).unwrap();
                NDArrayWrapper {
                    data: ArrayData::Int32(Arc::new(RwLock::new(view.to_owned()))),
                    dtype: DType::Int32,
                }
            }
            ArrayData::Int16(_) => {
                let view = extract_view_i16(wrapper, meta).unwrap();
                NDArrayWrapper {
                    data: ArrayData::Int16(Arc::new(RwLock::new(view.to_owned()))),
                    dtype: DType::Int16,
                }
            }
            ArrayData::Int8(_) => {
                let view = extract_view_i8(wrapper, meta).unwrap();
                NDArrayWrapper {
                    data: ArrayData::Int8(Arc::new(RwLock::new(view.to_owned()))),
                    dtype: DType::Int8,
                }
            }
            ArrayData::Uint64(_) => {
                let view = extract_view_u64(wrapper, meta).unwrap();
                NDArrayWrapper {
                    data: ArrayData::Uint64(Arc::new(RwLock::new(view.to_owned()))),
                    dtype: DType::Uint64,
                }
            }
            ArrayData::Uint32(_) => {
                let view = extract_view_u32(wrapper, meta).unwrap();
                NDArrayWrapper {
                    data: ArrayData::Uint32(Arc::new(RwLock::new(view.to_owned()))),
                    dtype: DType::Uint32,
                }
            }
            ArrayData::Uint16(_) => {
                let view = extract_view_u16(wrapper, meta).unwrap();
                NDArrayWrapper {
                    data: ArrayData::Uint16(Arc::new(RwLock::new(view.to_owned()))),
                    dtype: DType::Uint16,
                }
            }
            ArrayData::Uint8(_) => {
                let view = extract_view_u8(wrapper, meta).unwrap();
                NDArrayWrapper {
                    data: ArrayData::Uint8(Arc::new(RwLock::new(view.to_owned()))),
                    dtype: DType::Uint8,
                }
            }
            ArrayData::Bool(_) => {
                let view = extract_view_bool(wrapper, meta).unwrap();
                NDArrayWrapper {
                    data: ArrayData::Bool(Arc::new(RwLock::new(view.to_owned()))),
                    dtype: DType::Bool,
                }
            }
        }
    }
}

/// Cast array data from source dtype to a desired target dtype.
fn cast_to_dtype(wrapper: &NDArrayWrapper, meta: &ViewMetadata, target: DType) -> NDArrayWrapper {
    match target {
        DType::Float64 => NDArrayWrapper {
            data: ArrayData::Float64(Arc::new(RwLock::new(
                extract_view_as_f64(wrapper, &meta).expect("Failed to cast to f64"),
            ))),
            dtype: DType::Float64,
        },
        DType::Float32 => NDArrayWrapper {
            data: ArrayData::Float32(Arc::new(RwLock::new(
                extract_view_as_f32(wrapper, &meta).expect("Failed to cast to f32"),
            ))),
            dtype: DType::Float32,
        },
        DType::Int64 => NDArrayWrapper {
            data: ArrayData::Int64(Arc::new(RwLock::new(
                extract_view_as_i64(wrapper, &meta).expect("Failed to cast to i64"),
            ))),
            dtype: DType::Int64,
        },
        DType::Int32 => NDArrayWrapper {
            data: ArrayData::Int32(Arc::new(RwLock::new(
                extract_view_as_i32(wrapper, &meta).expect("Failed to cast to i32"),
            ))),
            dtype: DType::Int32,
        },
        DType::Int16 => NDArrayWrapper {
            data: ArrayData::Int16(Arc::new(RwLock::new(
                extract_view_as_i16(wrapper, &meta).expect("Failed to cast to i16"),
            ))),
            dtype: DType::Int16,
        },
        DType::Int8 => NDArrayWrapper {
            data: ArrayData::Int8(Arc::new(RwLock::new(
                extract_view_as_i8(wrapper, &meta).expect("Failed to cast to i8"),
            ))),
            dtype: DType::Int8,
        },
        DType::Uint64 => NDArrayWrapper {
            data: ArrayData::Uint64(Arc::new(RwLock::new(
                extract_view_as_u64(wrapper, &meta).expect("Failed to cast to u64"),
            ))),
            dtype: DType::Uint64,
        },
        DType::Uint32 => NDArrayWrapper {
            data: ArrayData::Uint32(Arc::new(RwLock::new(
                extract_view_as_u32(wrapper, &meta).expect("Failed to cast to u32"),
            ))),
            dtype: DType::Uint32,
        },
        DType::Uint16 => NDArrayWrapper {
            data: ArrayData::Uint16(Arc::new(RwLock::new(
                extract_view_as_u16(wrapper, &meta).expect("Failed to cast to u16"),
            ))),
            dtype: DType::Uint16,
        },
        DType::Uint8 => NDArrayWrapper {
            data: ArrayData::Uint8(Arc::new(RwLock::new(
                extract_view_as_u8(wrapper, &meta).expect("Failed to cast to u8"),
            ))),
            dtype: DType::Uint8,
        },
        DType::Bool => NDArrayWrapper {
            data: ArrayData::Bool(Arc::new(RwLock::new(
                extract_view_as_bool(wrapper, &meta).expect("Failed to cast to bool"),
            ))),
            dtype: DType::Bool,
        },
    }
}
