//! Reshape operations.

use crate::core::view_helpers::{
    extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32, extract_view_i64,
    extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64, extract_view_u8,
};
use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{self, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::ffi::NdArrayHandle;
use crate::ffi::ViewMetadata;
use ndarray::IxDyn;
use parking_lot::RwLock;
use std::sync::Arc;

/// Reshape array to new shape.
#[no_mangle]
pub unsafe extern "C" fn ndarray_reshape(
    handle: *const NdArrayHandle,
    meta: *const ViewMetadata,
    new_shape: *const usize,
    new_ndim: usize,
    order: i32,
    out_handle: *mut *mut NdArrayHandle,
) -> i32 {
    if handle.is_null() || meta.is_null() || new_shape.is_null() || out_handle.is_null() {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
        let meta = &*meta;
        let shape_slice = meta.shape_slice();
        let new_shape_slice = std::slice::from_raw_parts(new_shape, new_ndim);

        // Validate total elements match
        let new_size: usize = new_shape_slice.iter().product();
        let old_size: usize = shape_slice.iter().product();

        if new_size != old_size {
            error::set_last_error(format!(
                "Cannot reshape array of size {} into shape {:?} (size {})",
                old_size, new_shape_slice, new_size
            ));
            return ERR_SHAPE;
        }

        let order_enum = match order {
            0 => ndarray::Order::RowMajor,
            1 => ndarray::Order::ColumnMajor,
            _ => {
                error::set_last_error(format!(
                    "Invalid order: {}. Use 0 for RowMajor, 1 for ColumnMajor",
                    order
                ));
                return ERR_GENERIC;
            }
        };

        let result_wrapper = match wrapper.dtype {
            DType::Float64 => {
                let Some(view) = extract_view_f64(wrapper, meta) else {
                    error::set_last_error("Failed to extract f64 view".to_string());
                    return ERR_GENERIC;
                };
                let new_ixdyn = IxDyn(new_shape_slice);
                match view.to_shape((new_ixdyn, order_enum)) {
                    Ok(reshaped) => NDArrayWrapper {
                        data: ArrayData::Float64(Arc::new(RwLock::new(reshaped.to_owned()))),
                        dtype: DType::Float64,
                    },
                    Err(e) => {
                        error::set_last_error(format!("Reshape failed: {}", e));
                        return ERR_SHAPE;
                    }
                }
            }
            DType::Float32 => {
                let Some(view) = extract_view_f32(wrapper, meta) else {
                    error::set_last_error("Failed to extract f32 view".to_string());
                    return ERR_GENERIC;
                };
                let new_ixdyn = IxDyn(new_shape_slice);
                match view.to_shape((new_ixdyn, order_enum)) {
                    Ok(reshaped) => NDArrayWrapper {
                        data: ArrayData::Float32(Arc::new(RwLock::new(reshaped.to_owned()))),
                        dtype: DType::Float32,
                    },
                    Err(e) => {
                        error::set_last_error(format!("Reshape failed: {}", e));
                        return ERR_SHAPE;
                    }
                }
            }
            DType::Int64 => {
                let Some(view) = extract_view_i64(wrapper, meta) else {
                    error::set_last_error("Failed to extract i64 view".to_string());
                    return ERR_GENERIC;
                };
                let new_ixdyn = IxDyn(new_shape_slice);
                match view.to_shape((new_ixdyn, order_enum)) {
                    Ok(reshaped) => NDArrayWrapper {
                        data: ArrayData::Int64(Arc::new(RwLock::new(reshaped.to_owned()))),
                        dtype: DType::Int64,
                    },
                    Err(e) => {
                        error::set_last_error(format!("Reshape failed: {}", e));
                        return ERR_SHAPE;
                    }
                }
            }
            DType::Int32 => {
                let Some(view) = extract_view_i32(wrapper, meta) else {
                    error::set_last_error("Failed to extract i32 view".to_string());
                    return ERR_GENERIC;
                };
                let new_ixdyn = IxDyn(new_shape_slice);
                match view.to_shape((new_ixdyn, order_enum)) {
                    Ok(reshaped) => NDArrayWrapper {
                        data: ArrayData::Int32(Arc::new(RwLock::new(reshaped.to_owned()))),
                        dtype: DType::Int32,
                    },
                    Err(e) => {
                        error::set_last_error(format!("Reshape failed: {}", e));
                        return ERR_SHAPE;
                    }
                }
            }
            DType::Int16 => {
                let Some(view) = extract_view_i16(wrapper, meta) else {
                    error::set_last_error("Failed to extract i16 view".to_string());
                    return ERR_GENERIC;
                };
                let new_ixdyn = IxDyn(new_shape_slice);
                match view.to_shape((new_ixdyn, order_enum)) {
                    Ok(reshaped) => NDArrayWrapper {
                        data: ArrayData::Int16(Arc::new(RwLock::new(reshaped.to_owned()))),
                        dtype: DType::Int16,
                    },
                    Err(e) => {
                        error::set_last_error(format!("Reshape failed: {}", e));
                        return ERR_SHAPE;
                    }
                }
            }
            DType::Int8 => {
                let Some(view) = extract_view_i8(wrapper, meta) else {
                    error::set_last_error("Failed to extract i8 view".to_string());
                    return ERR_GENERIC;
                };
                let new_ixdyn = IxDyn(new_shape_slice);
                match view.to_shape((new_ixdyn, order_enum)) {
                    Ok(reshaped) => NDArrayWrapper {
                        data: ArrayData::Int8(Arc::new(RwLock::new(reshaped.to_owned()))),
                        dtype: DType::Int8,
                    },
                    Err(e) => {
                        error::set_last_error(format!("Reshape failed: {}", e));
                        return ERR_SHAPE;
                    }
                }
            }
            DType::Uint64 => {
                let Some(view) = extract_view_u64(wrapper, meta) else {
                    error::set_last_error("Failed to extract u64 view".to_string());
                    return ERR_GENERIC;
                };
                let new_ixdyn = IxDyn(new_shape_slice);
                match view.to_shape((new_ixdyn, order_enum)) {
                    Ok(reshaped) => NDArrayWrapper {
                        data: ArrayData::Uint64(Arc::new(RwLock::new(reshaped.to_owned()))),
                        dtype: DType::Uint64,
                    },
                    Err(e) => {
                        error::set_last_error(format!("Reshape failed: {}", e));
                        return ERR_SHAPE;
                    }
                }
            }
            DType::Uint32 => {
                let Some(view) = extract_view_u32(wrapper, meta) else {
                    error::set_last_error("Failed to extract u32 view".to_string());
                    return ERR_GENERIC;
                };
                let new_ixdyn = IxDyn(new_shape_slice);
                match view.to_shape((new_ixdyn, order_enum)) {
                    Ok(reshaped) => NDArrayWrapper {
                        data: ArrayData::Uint32(Arc::new(RwLock::new(reshaped.to_owned()))),
                        dtype: DType::Uint32,
                    },
                    Err(e) => {
                        error::set_last_error(format!("Reshape failed: {}", e));
                        return ERR_SHAPE;
                    }
                }
            }
            DType::Uint16 => {
                let Some(view) = extract_view_u16(wrapper, meta) else {
                    error::set_last_error("Failed to extract u16 view".to_string());
                    return ERR_GENERIC;
                };
                let new_ixdyn = IxDyn(new_shape_slice);
                match view.to_shape((new_ixdyn, order_enum)) {
                    Ok(reshaped) => NDArrayWrapper {
                        data: ArrayData::Uint16(Arc::new(RwLock::new(reshaped.to_owned()))),
                        dtype: DType::Uint16,
                    },
                    Err(e) => {
                        error::set_last_error(format!("Reshape failed: {}", e));
                        return ERR_SHAPE;
                    }
                }
            }
            DType::Uint8 => {
                let Some(view) = extract_view_u8(wrapper, meta) else {
                    error::set_last_error("Failed to extract u8 view".to_string());
                    return ERR_GENERIC;
                };
                let new_ixdyn = IxDyn(new_shape_slice);
                match view.to_shape((new_ixdyn, order_enum)) {
                    Ok(reshaped) => NDArrayWrapper {
                        data: ArrayData::Uint8(Arc::new(RwLock::new(reshaped.to_owned()))),
                        dtype: DType::Uint8,
                    },
                    Err(e) => {
                        error::set_last_error(format!("Reshape failed: {}", e));
                        return ERR_SHAPE;
                    }
                }
            }
            DType::Bool => {
                error::set_last_error("reshape() not supported for Bool type".to_string());
                return ERR_GENERIC;
            }
        };

        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
