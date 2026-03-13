//! Operational helpers for element-wise operations.
//!
//! This module provides macros and helper functions for mathematical,
//! comparison, and logical operations while preserving native types.

// ============================================================================
// Math Operations
// ============================================================================

/// Macro to generate a binary operation match arm for FFI functions.
#[macro_export]
macro_rules! binary_op_arm {
    (
        $a_wrapper:expr, $a_meta:expr,
        $b_wrapper:expr, $b_meta:expr,
        $dtype:path, $extract_fn:ident, $variant:path, $op:tt
    ) => {
        {
            let Some(a_view) = $extract_fn($a_wrapper, $a_meta) else {
                crate::error::set_last_error(format!("Failed to extract a as {}", stringify!($dtype)));
                return crate::error::ERR_GENERIC;
            };
            let Some(b_view) = $extract_fn($b_wrapper, $b_meta) else {
                crate::error::set_last_error(format!("Failed to extract b as {}", stringify!($dtype)));
                return crate::error::ERR_GENERIC;
            };
            let result = a_view $op b_view;
            crate::core::NDArrayWrapper {
                data: $variant(::std::sync::Arc::new(::parking_lot::RwLock::new(result))),
                dtype: $dtype,
            }
        }
    };
}

/// Macro to generate a scalar operation match arm for FFI functions.
#[macro_export]
macro_rules! scalar_op_arm {
    (
        $wrapper:expr, $meta:expr,
        $scalar:expr, $dtype:path, $extract_fn:ident, $variant:path, $op:tt
    ) => {
        {
            let Some(view) = $extract_fn($wrapper, $meta) else {
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

// ============================================================================
// Comparison Operations
// ============================================================================

/// Comparison functions for use with Zip::map_collect.
pub mod comparison_ops {
    #[inline(always)]
    pub fn eq<A: PartialEq>(a: &A, b: &A) -> u8 {
        (a == b) as u8
    }

    #[inline(always)]
    pub fn ne<A: PartialEq>(a: &A, b: &A) -> u8 {
        (a != b) as u8
    }

    #[inline(always)]
    pub fn gt<A: PartialOrd>(a: &A, b: &A) -> u8 {
        (a > b) as u8
    }

    #[inline(always)]
    pub fn gte<A: PartialOrd>(a: &A, b: &A) -> u8 {
        (a >= b) as u8
    }

    #[inline(always)]
    pub fn lt<A: PartialOrd>(a: &A, b: &A) -> u8 {
        (a < b) as u8
    }

    #[inline(always)]
    pub fn lte<A: PartialOrd>(a: &A, b: &A) -> u8 {
        (a <= b) as u8
    }
}

/// Logical operations for use with Zip::map_collect.
pub mod logical_ops {
    #[inline(always)]
    pub fn and(a: &u8, b: &u8) -> u8 {
        ((*a != 0) && (*b != 0)) as u8
    }

    #[inline(always)]
    pub fn or(a: &u8, b: &u8) -> u8 {
        ((*a != 0) || (*b != 0)) as u8
    }

    #[inline(always)]
    pub fn xor(a: &u8, b: &u8) -> u8 {
        ((*a != 0) ^ (*b != 0)) as u8
    }

    #[inline(always)]
    pub fn not(a: &u8) -> u8 {
        (*a == 0) as u8
    }
}

/// Macro to generate a binary comparison operation match arm.
#[macro_export]
macro_rules! binary_cmp_op_arm {
    (
        $a_wrapper:expr, $a_meta:expr,
        $b_wrapper:expr, $b_meta:expr,
        $dtype:path, $extract_fn:ident, $cmp_op:ident
    ) => {{
        let Some(a_view) = $extract_fn($a_wrapper, $a_meta) else {
            crate::error::set_last_error(format!("Failed to extract a as {}", stringify!($dtype)));
            return crate::error::ERR_GENERIC;
        };
        let Some(b_view) = $extract_fn($b_wrapper, $b_meta) else {
            crate::error::set_last_error(format!("Failed to extract b as {}", stringify!($dtype)));
            return crate::error::ERR_GENERIC;
        };
        let broadcast_shape =
            match crate::core::view_helpers::broadcast_shape(a_view.shape(), b_view.shape()) {
                Some(s) => s,
                None => {
                    crate::error::set_last_error("incompatible shapes for comparison".to_string());
                    return crate::error::ERR_SHAPE;
                }
            };
        let a_bc = match a_view.broadcast(broadcast_shape.as_slice()) {
            Some(v) => v,
            None => {
                crate::error::set_last_error("incompatible shapes for comparison".to_string());
                return crate::error::ERR_SHAPE;
            }
        };
        let b_bc = match b_view.broadcast(broadcast_shape.as_slice()) {
            Some(v) => v,
            None => {
                crate::error::set_last_error("incompatible shapes for comparison".to_string());
                return crate::error::ERR_SHAPE;
            }
        };
        let result = ndarray::Zip::from(&a_bc)
            .and(&b_bc)
            .map_collect($crate::core::op_helpers::comparison_ops::$cmp_op);
        crate::core::NDArrayWrapper {
            data: crate::core::ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                result,
            ))),
            dtype: crate::dtype::DType::Bool,
        }
    }};
}

/// Macro to generate a unary logical operation.
#[macro_export]
macro_rules! unary_logical_op_arm {
    (
        $wrapper:expr, $meta:expr,
        $op:ident
    ) => {{
        let Some(view) = crate::core::view_helpers::extract_view_as_bool($wrapper, $meta) else {
            crate::error::set_last_error("Failed to extract view as bool".to_string());
            return crate::error::ERR_GENERIC;
        };
        let result = view.mapv(|x| $crate::core::op_helpers::logical_ops::$op(&x));
        crate::core::NDArrayWrapper {
            data: crate::core::ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                result.clone(),
            ))),
            dtype: crate::dtype::DType::Bool,
        }
    }};
}

/// Macro to generate a binary logical operation match arm.
#[macro_export]
macro_rules! binary_logical_op_arm {
    (
        $a_wrapper:expr, $a_meta:expr,
        $b_wrapper:expr, $b_meta:expr,
        $op:ident
    ) => {{
        let Some(a_view) = crate::core::view_helpers::extract_view_as_bool($a_wrapper, $a_meta)
        else {
            crate::error::set_last_error("Failed to extract a as bool".to_string());
            return crate::error::ERR_GENERIC;
        };
        let Some(b_view) = crate::core::view_helpers::extract_view_as_bool($b_wrapper, $b_meta)
        else {
            crate::error::set_last_error("Failed to extract b as bool".to_string());
            return crate::error::ERR_GENERIC;
        };
        let broadcast_shape =
            match crate::core::view_helpers::broadcast_shape(a_view.shape(), b_view.shape()) {
                Some(s) => s,
                None => {
                    crate::error::set_last_error(
                        "incompatible shapes for logical operation".to_string(),
                    );
                    return crate::error::ERR_SHAPE;
                }
            };
        let a_bc = match a_view.broadcast(broadcast_shape.as_slice()) {
            Some(v) => v,
            None => {
                crate::error::set_last_error(
                    "incompatible shapes for logical operation".to_string(),
                );
                return crate::error::ERR_SHAPE;
            }
        };
        let b_bc = match b_view.broadcast(broadcast_shape.as_slice()) {
            Some(v) => v,
            None => {
                crate::error::set_last_error(
                    "incompatible shapes for logical operation".to_string(),
                );
                return crate::error::ERR_SHAPE;
            }
        };
        let result = ndarray::Zip::from(&a_bc)
            .and(&b_bc)
            .map_collect($crate::core::op_helpers::logical_ops::$op);
        crate::core::NDArrayWrapper {
            data: crate::core::ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                result,
            ))),
            dtype: crate::dtype::DType::Bool,
        }
    }};
}

/// Macro to generate a scalar comparison operation.
#[macro_export]
macro_rules! scalar_cmp_op_arm {
    (
        $wrapper:expr, $meta:expr,
        $scalar:expr, $cmp_op:tt
    ) => {{
        let Some(view) =
            crate::core::view_helpers::extract_view_as_f64($wrapper, $meta)
        else {
            crate::error::set_last_error("Failed to extract view for scalar comparison".to_string());
            return crate::error::ERR_GENERIC;
        };
        let result = view.mapv(|x| (x $cmp_op $scalar) as u8);
        crate::core::NDArrayWrapper {
            data: crate::core::ArrayData::Bool(::std::sync::Arc::new(
                ::parking_lot::RwLock::new(result),
            )),
            dtype: crate::dtype::DType::Bool,
        }
    }};
}
