//! Macros for reducing type dispatch boilerplate.
//!
//! These macros generate repetitive code for handling all 11 data types
//! without manual duplication.
//!
//! Note: FFI functions (extern "C") must be written explicitly because
//! cbindgen cannot expand Rust macros when generating the C header.

/// Dispatch an operation across all ArrayData variants.
///
/// This macro expands a match expression that handles all 11 type variants,
/// binding the inner Arc<RwLock<ArrayD<T>>> to the specified identifier.
///
/// # Example
/// ```ignore
/// match_array_data!(self.data, arr => {
///     arr.read().shape().to_vec()
/// })
/// ```
#[macro_export]
macro_rules! match_array_data {
    ($data:expr, $arr:ident => $body:expr) => {
        match &$data {
            $crate::core::ArrayData::Int8($arr) => $body,
            $crate::core::ArrayData::Int16($arr) => $body,
            $crate::core::ArrayData::Int32($arr) => $body,
            $crate::core::ArrayData::Int64($arr) => $body,
            $crate::core::ArrayData::Uint8($arr) => $body,
            $crate::core::ArrayData::Uint16($arr) => $body,
            $crate::core::ArrayData::Uint32($arr) => $body,
            $crate::core::ArrayData::Uint64($arr) => $body,
            $crate::core::ArrayData::Float32($arr) => $body,
            $crate::core::ArrayData::Float64($arr) => $body,
            $crate::core::ArrayData::Bool($arr) => $body,
        }
    };
}

/// Generate from_slice_* methods for NDArrayWrapper.
///
/// This macro generates type-specific constructors that create an NDArrayWrapper
/// from a slice of the given type.
///
/// # Arguments
/// * `$method` - Method name (e.g., `from_slice_i8`)
/// * `$type` - Rust type (e.g., `i8`)
/// * `$variant` - ArrayData variant (e.g., `Int8`)
/// * `$dtype` - DType variant (e.g., `Int8`)
#[macro_export]
macro_rules! impl_from_slice {
    ($($method:ident, $type:ty, $variant:ident, $dtype:ident);* $(;)?) => {
        impl $crate::core::NDArrayWrapper {
            $(
                #[doc = concat!("Create array from ", stringify!($type), " slice.")]
                pub fn $method(data: &[$type], shape: &[usize]) -> Result<Self, String> {
                    let expected_len: usize = shape.iter().product();
                    if data.len() != expected_len {
                        return Err(format!(
                            "Data length {} does not match shape {:?} (expected {})",
                            data.len(),
                            shape,
                            expected_len
                        ));
                    }

                    let arr = ndarray::ArrayD::from_shape_vec(
                        ndarray::IxDyn(shape),
                        data.to_vec()
                    ).map_err(|e| format!("Shape error: {}", e))?;

                    Ok(Self {
                        data: $crate::core::ArrayData::$variant(
                            std::sync::Arc::new(parking_lot::RwLock::new(arr))
                        ),
                        dtype: $crate::dtype::DType::$dtype,
                    })
                }
            )*
        }
    };
}

/// Generate get_element_* methods for NDArrayWrapper.
///
/// Each method reads a single element at a flat index and returns the value
/// in its native type. Returns an error if the dtype doesn't match or the
/// index is out of bounds.
#[macro_export]
macro_rules! impl_get_element {
    ($($method:ident, $type:ty, $variant:ident);* $(;)?) => {
        impl $crate::core::NDArrayWrapper {
            $(
                #[doc = concat!("Get element at flat index as `", stringify!($type), "`.")]
                pub fn $method(&self, flat_index: usize) -> Result<$type, String> {
                    if let $crate::core::ArrayData::$variant(arr) = &self.data {
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
                                // Non-contiguous array: use iterator
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
///
/// Each method writes a single element at a flat index.
/// Returns an error if the dtype doesn't match or the index is out of bounds.
#[macro_export]
macro_rules! impl_set_element {
    ($($method:ident, $type:ty, $variant:ident);* $(;)?) => {
        impl $crate::core::NDArrayWrapper {
            $(
                #[doc = concat!("Set element at flat index from `", stringify!($type), "`.")]
                pub fn $method(&self, flat_index: usize, value: $type) -> Result<(), String> {
                    if let $crate::core::ArrayData::$variant(arr) = &self.data {
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
                                // Non-contiguous: fall back to indexed iteration
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

/// Generate assign_slice_* methods for NDArrayWrapper.
///
/// Each method assigns elements from a source array view to a destination view.
#[macro_export]
macro_rules! impl_assign_slice {
    ($($method:ident, $type:ty, $variant:ident);* $(;)?) => {
        impl $crate::core::NDArrayWrapper {
            $(
                pub fn $method(
                    &self,
                    dst_offset: usize,
                    dst_shape: &[usize],
                    dst_strides: &[usize],
                    src: &$crate::core::NDArrayWrapper,
                    src_offset: usize,
                    src_shape: &[usize],
                    src_strides: &[usize],
                ) -> Result<(), String> {
                    // 1. Check if self (dst) matches the expected type variant
                    if let $crate::core::ArrayData::$variant(_) = &self.data {
                        // OK
                    } else {
                         return Err(format!(
                            "Type mismatch: expected {}, got {:?}",
                            stringify!($variant), self.dtype
                        ));
                    }


                    // 2. Check if src matches dst type
                    if self.dtype != src.dtype {
                        return Err(format!(
                            "DType mismatch in assign: dst={:?}, src={:?}",
                            self.dtype, src.dtype
                        ));
                    }

                    // Check for self-assignment / aliasing
                    let is_same = self.is_same_array(src);

                    if is_same {
                        // Case 1: Same underlying array (potential aliasing)
                        let temp_data: Vec<$type> = {
                            if let $crate::core::ArrayData::$variant(arr) = &src.data {
                                let guard = arr.read();
                                let raw_ptr = guard.as_ptr();
                                unsafe {
                                    let ptr = raw_ptr.add(src_offset);
                                    let strides_ix = ndarray::IxDyn(src_strides);
                                    let view = ndarray::ArrayView::from_shape_ptr(
                                        ndarray::ShapeBuilder::strides(ndarray::IxDyn(src_shape), strides_ix),
                                        ptr
                                    );
                                    view.iter().cloned().collect()
                                }
                            } else {
                                unreachable!("DType checked above");
                            }
                        };

                        // Write to destination
                        if let $crate::core::ArrayData::$variant(arr) = &self.data {
                            let mut guard = arr.write();
                            let raw_ptr = guard.as_mut_ptr();
                            unsafe {
                                let ptr = raw_ptr.add(dst_offset);
                                let strides_ix = ndarray::IxDyn(dst_strides);
                                let mut view = ndarray::ArrayViewMut::from_shape_ptr(
                                    ndarray::ShapeBuilder::strides(ndarray::IxDyn(dst_shape), strides_ix),
                                    ptr
                                );

                                let temp_view = ndarray::ArrayView::from_shape(ndarray::IxDyn(dst_shape), &temp_data)
                                    .map_err(|e| e.to_string())?;

                                view.assign(&temp_view);
                            }
                        }
                    } else {
                        // Case 2: Different arrays
                        if let ($crate::core::ArrayData::$variant(dst_arr), $crate::core::ArrayData::$variant(src_arr)) = (&self.data, &src.data) {
                            // Lock source first
                            let src_guard = src_arr.read();
                            let src_ptr = src_guard.as_ptr();

                            // Lock dest
                            let mut dst_guard = dst_arr.write();
                            let dst_ptr = dst_guard.as_mut_ptr();

                            unsafe {
                                let s_ptr = src_ptr.add(src_offset);
                                let s_strides = ndarray::IxDyn(src_strides);
                                let src_view = ndarray::ArrayView::from_shape_ptr(
                                    ndarray::ShapeBuilder::strides(ndarray::IxDyn(src_shape), s_strides),
                                    s_ptr
                                );

                                let d_ptr = dst_ptr.add(dst_offset);
                                let d_strides = ndarray::IxDyn(dst_strides);
                                let mut dst_view = ndarray::ArrayViewMut::from_shape_ptr(
                                    ndarray::ShapeBuilder::strides(ndarray::IxDyn(dst_shape), d_strides),
                                    d_ptr
                                );

                                dst_view.assign(&src_view);
                            }
                        } else {
                            unreachable!("DType checked above");
                        }
                    }

                    Ok(())
                }

            )*
        }
    };
}

/// Generate copy_view_* methods for NDArrayWrapper.
///
/// Each method copies a view (defined by offset, shape, strides) to a new NDArrayWrapper.
#[macro_export]
macro_rules! impl_copy_view {
    ($($method:ident, $type:ty, $variant:ident, $dtype:ident);* $(;)?) => {
        impl $crate::core::NDArrayWrapper {
            $(
                pub unsafe fn $method(
                    &self,
                    meta: &$crate::ffi::ViewMetadata,
                ) -> Result<$crate::core::NDArrayWrapper, String> {
                    let shape = meta.shape_slice();
                    let strides = meta.strides_slice();
                    let offset = meta.offset;

                    if let $crate::core::ArrayData::$variant(arr) = &self.data {
                        let guard = arr.read();
                        let raw_ptr = guard.as_ptr();

                        unsafe {
                            let ptr = raw_ptr.add(offset);
                            let strides_ix = ndarray::IxDyn(strides);
                            let view = ndarray::ArrayView::from_shape_ptr(
                                ndarray::ShapeBuilder::strides(ndarray::IxDyn(shape), strides_ix),
                                ptr
                            );

                            // Create a new owned array from the view
                            // to_owned() creates a standard layout array
                            let new_arr = view.to_owned();

                            Ok($crate::core::NDArrayWrapper {
                                data: $crate::core::ArrayData::$variant(
                                    std::sync::Arc::new(parking_lot::RwLock::new(new_arr))
                                ),
                                dtype: $crate::dtype::DType::$dtype,
                            })
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

/// Generate fill_slice_* methods for NDArrayWrapper.
///
/// Each method fills a view (defined by offset, shape, strides) with a scalar value.
#[macro_export]
macro_rules! impl_fill_slice {
    ($($method:ident, $type:ty, $variant:ident);* $(;)?) => {
        impl $crate::core::NDArrayWrapper {
            $(
                pub unsafe fn $method(
                    &self,
                    value: $type,
                    meta: &$crate::ffi::ViewMetadata,
                ) -> Result<(), String> {
                    let shape = meta.shape_slice();
                    let strides = meta.strides_slice();
                    let offset = meta.offset;

                    if let $crate::core::ArrayData::$variant(arr) = &self.data {
                        let mut guard = arr.write();
                        let raw_ptr = guard.as_mut_ptr();

                        // Safety: The caller (PHP) is responsible for ensuring the view
                        // is within bounds of the allocated memory.
                        // We construct a view from the raw pointer offset.

                        unsafe {
                            let ptr = raw_ptr.add(offset);
                            // Convert strides to IxDyn (which expects usize, negative strides are wrapped)
                            let strides_ix = ndarray::IxDyn(strides);

                            // Construct the view
                            // We use from_shape_ptr which is unsafe.
                            let mut view = ndarray::ArrayViewMut::from_shape_ptr(
                                ndarray::ShapeBuilder::strides(ndarray::IxDyn(shape), strides_ix),
                                ptr
                            );

                            view.fill(value);
                        }

                        Ok(())
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

/// Generate to_vec_* methods for NDArrayWrapper.
///
/// This macro generates type-specific accessors that return a copy of the
/// data as a `Vec<T>`.
#[macro_export]
macro_rules! impl_to_vec {
    ($($method:ident, $type:ty, $variant:ident);* $(;)?) => {
        impl $crate::core::NDArrayWrapper {
            $(
                #[doc = concat!("Get array data as `Vec<", stringify!($type), ">` if type matches.")]
                pub fn $method(&self) -> Option<Vec<$type>> {
                    if let $crate::core::ArrayData::$variant(arr) = &self.data {
                        Some(arr.read().iter().cloned().collect())
                    } else {
                        None
                    }
                }
            )*
        }
    };
}
