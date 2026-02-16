//! Comparison operation helpers for element-wise eq, ne, gt, gte, lt, lte.
//!
//! Uses ndarray's Zip and broadcast for broadcasting support.
//! Output is always Bool (stored as u8).

/// Comparison functions for use with Zip::map_collect.
/// Return u8 (0 or 1) directly to avoid an extra conversion step.
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

/// Macro to generate a binary comparison operation match arm.
///
/// Extracts both arrays, broadcasts to compatible shape, applies the comparison,
/// and returns (NDArrayWrapper with Bool data, shape).
#[macro_export]
macro_rules! binary_cmp_op_arm {
    (
        $a_wrapper:expr, $a_offset:expr, $a_shape:expr, $a_strides:expr,
        $b_wrapper:expr, $b_offset:expr, $b_shape:expr, $b_strides:expr,
        $dtype:path, $extract_fn:ident, $cmp_op:ident
    ) => {{
        let Some(a_view) = $extract_fn($a_wrapper, $a_offset, $a_shape, $a_strides) else {
            crate::error::set_last_error(format!(
                "Failed to extract a as {}",
                stringify!($dtype)
            ));
            return crate::error::ERR_GENERIC;
        };
        let Some(b_view) = $extract_fn($b_wrapper, $b_offset, $b_shape, $b_strides) else {
            crate::error::set_last_error(format!(
                "Failed to extract b as {}",
                stringify!($dtype)
            ));
            return crate::error::ERR_GENERIC;
        };
        let broadcast_shape = match crate::core::view_helpers::broadcast_shape(
            a_view.shape(),
            b_view.shape(),
        ) {
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
            .map_collect($crate::core::comparison_helpers::comparison_ops::$cmp_op);
        let shape = result.shape().to_vec();
        (
            crate::core::NDArrayWrapper {
                data: crate::core::ArrayData::Bool(::std::sync::Arc::new(
                    ::parking_lot::RwLock::new(result),
                )),
                dtype: crate::dtype::DType::Bool,
            },
            shape,
        )
    }};
}

/// Macro to generate a scalar comparison operation.
///
/// Uses extract_view_as_f64 to support all numeric types (converts for comparison).
/// Returns NDArrayWrapper with Bool data.
#[macro_export]
macro_rules! scalar_cmp_op_arm {
    (
        $wrapper:expr, $offset:expr, $shape:expr, $strides:expr,
        $scalar:expr, $cmp_op:tt
    ) => {{
        let Some(view) =
            crate::core::view_helpers::extract_view_as_f64($wrapper, $offset, $shape, $strides)
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
