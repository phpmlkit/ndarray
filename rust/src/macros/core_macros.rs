//! Core macros for NDArray operations.
//!
//! These macros generate repetitive implementations for handling all 11 data types.

/// Dispatch an operation across all ArrayData variants.
#[macro_export]
macro_rules! match_array_data {
    ($data:expr, $arr:ident => $body:expr) => {
        match &$data {
            $crate::types::ArrayData::Int8($arr) => $body,
            $crate::types::ArrayData::Int16($arr) => $body,
            $crate::types::ArrayData::Int32($arr) => $body,
            $crate::types::ArrayData::Int64($arr) => $body,
            $crate::types::ArrayData::Uint8($arr) => $body,
            $crate::types::ArrayData::Uint16($arr) => $body,
            $crate::types::ArrayData::Uint32($arr) => $body,
            $crate::types::ArrayData::Uint64($arr) => $body,
            $crate::types::ArrayData::Float32($arr) => $body,
            $crate::types::ArrayData::Float64($arr) => $body,
            $crate::types::ArrayData::Bool($arr) => $body,
        }
    };
}

/// Generate from_slice_* methods for NDArrayWrapper.
#[macro_export]
macro_rules! impl_from_slice {
    ($($method:ident, $type:ty, $variant:ident, $dtype:ident);* $(;)?) => {
        impl $crate::types::NDArrayWrapper {
            $(
                #[doc = concat!("Create array from ", stringify!($type), " slice.")]
                pub fn $method(data: &[$type], shape: &[usize]) -> Result<Self, String> {
                    let expected_len: usize = shape.iter().product();
                    if data.len() != expected_len {
                        return Err(format!(
                            "Data length {} does not match shape {:?} (expected {})",
                            data.len(), shape, expected_len
                        ));
                    }

                    let arr = ndarray::ArrayD::from_shape_vec(
                        ndarray::IxDyn(shape),
                        data.to_vec()
                    ).map_err(|e| format!("Shape error: {}", e))?;

                    Ok(Self {
                        data: $crate::types::ArrayData::$variant(
                            std::sync::Arc::new(parking_lot::RwLock::new(arr))
                        ),
                        dtype: $crate::types::dtype::DType::$dtype,
                    })
                }
            )*
        }
    };
}

/// Generate get_element_* methods for NDArrayWrapper.
#[macro_export]
macro_rules! impl_get_element {
    ($($method:ident, $type:ty, $variant:ident);* $(;)?) => {
        impl $crate::types::NDArrayWrapper {
            $(
                #[doc = concat!("Get element at flat index as `", stringify!($type), "`.")]
                pub fn $method(&self, flat_index: usize) -> Result<$type, String> {
                    if let $crate::types::ArrayData::$variant(arr) = &self.data {
                        let guard = arr.read();
                        let flat = guard.as_slice_memory_order();
                        match flat {
                            Some(slice) => {
                                if flat_index >= slice.len() {
                                    Err(format!(
                                        "Index {} out of bounds for array with {} elements",
                                        flat_index, slice.len()
                                    ))
                                } else {
                                    Ok(slice[flat_index])
                                }
                            }
                            None => {
                                guard.iter().nth(flat_index).copied().ok_or_else(|| {
                                    format!(
                                        "Index {} out of bounds for array with {} elements",
                                        flat_index, guard.len()
                                    )
                                })
                            }
                        }
                    } else {
                        Err(format!(
                            "Type mismatch: expected {}, got {:?}",
                            stringify!($variant), self.dtype
                        ))
                    }
                }
            )*
        }
    };
}

/// Generate set_element_* methods for NDArrayWrapper.
#[macro_export]
macro_rules! impl_set_element {
    ($($method:ident, $type:ty, $variant:ident);* $(;)?) => {
        impl $crate::types::NDArrayWrapper {
            $(
                #[doc = concat!("Set element at flat index from `", stringify!($type), "`.")]
                pub fn $method(&self, flat_index: usize, value: $type) -> Result<(), String> {
                    if let $crate::types::ArrayData::$variant(arr) = &self.data {
                        let mut guard = arr.write();
                        let flat = guard.as_slice_memory_order_mut();
                        match flat {
                            Some(slice) => {
                                if flat_index >= slice.len() {
                                    Err(format!(
                                        "Index {} out of bounds for array with {} elements",
                                        flat_index, slice.len()
                                    ))
                                } else {
                                    slice[flat_index] = value;
                                    Ok(())
                                }
                            }
                            None => {
                                let len = guard.len();
                                if flat_index >= len {
                                    return Err(format!(
                                        "Index {} out of bounds for array with {} elements",
                                        flat_index, len
                                    ));
                                }
                                if let Some(elem) = guard.iter_mut().nth(flat_index) {
                                    *elem = value;
                                    Ok(())
                                } else {
                                    Err(format!(
                                        "Index {} out of bounds for array with {} elements",
                                        flat_index, len
                                    ))
                                }
                            }
                        }
                    } else {
                        Err(format!(
                            "Type mismatch: expected {}, got {:?}",
                            stringify!($variant), self.dtype
                        ))
                    }
                }
            )*
        }
    };
}
