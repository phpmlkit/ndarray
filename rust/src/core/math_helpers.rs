//! Math operation helpers for type-specific unary and binary operations.
//!
//! This module provides macros and helper functions for performing mathematical
//! operations while preserving native types and avoiding unnecessary conversions.

/// Macro to generate a binary operation match arm for FFI functions.
///
/// Extracts both arrays as the target type, performs the operation,
/// and returns the wrapper and shape.
#[macro_export]
macro_rules! binary_op_arm {
    (
        $a_wrapper:expr, $a_offset:expr, $a_shape:expr, $a_strides:expr,
        $b_wrapper:expr, $b_offset:expr, $b_shape:expr, $b_strides:expr,
        $dtype:path, $extract_fn:ident, $variant:path, $op:tt
    ) => {
        {
            let Some(a_view) = $extract_fn($a_wrapper, $a_offset, $a_shape, $a_strides) else {
                crate::error::set_last_error(format!("Failed to extract a as {}", stringify!($dtype)));
                return crate::error::ERR_GENERIC;
            };
            let Some(b_view) = $extract_fn($b_wrapper, $b_offset, $b_shape, $b_strides) else {
                crate::error::set_last_error(format!("Failed to extract b as {}", stringify!($dtype)));
                return crate::error::ERR_GENERIC;
            };
            let result = a_view $op b_view;
            let shape = result.shape().to_vec();
            (
                crate::core::NDArrayWrapper {
                    data: $variant(::std::sync::Arc::new(::parking_lot::RwLock::new(result))),
                    dtype: $dtype,
                },
                shape,
            )
        }
    };
}

/// Macro to generate a scalar operation match arm for FFI functions.
///
/// Extracts the array as the target type, applies the scalar operation,
/// and returns the wrapper.
#[macro_export]
macro_rules! scalar_op_arm {
    (
        $wrapper:expr, $offset:expr, $shape:expr, $strides:expr,
        $scalar:expr, $dtype:path, $extract_fn:ident, $variant:path, $op:tt
    ) => {
        {
            let Some(view) = $extract_fn($wrapper, $offset, $shape, $strides) else {
                crate::error::set_last_error(format!("Failed to extract view as {}", stringify!($dtype)));
                return crate::error::ERR_GENERIC;
            };
            let result = view.mapv(|x| x $op $scalar);
            crate::core::NDArrayWrapper {
                data: $variant(::std::sync::Arc::new(::parking_lot::RwLock::new(result))),
                dtype: $dtype,
            }
        }
    };
}
