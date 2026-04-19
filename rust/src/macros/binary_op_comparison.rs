//! Binary operation macro for comparison operations.
//!
//! This module provides the `binary_op_comparison` macro which:
//! - Extracts views for native type pairs (zero-copy for matching types)
//! - Promotes mixed types and converts before operation
//! - Performs element-wise operations with broadcasting
//! - Works with all numeric types and Bool
//! - Always returns Bool array
//!
//! Pass `equality` for `==` / `!=` (includes complex dtypes, NumPy-compatible), or `ordering` for
//! `<`, `<=`, `>`, `>=` (complex dtypes are rejected, matching NumPy).

/// Same-dtype branch for `Complex64` / `Complex128` (`$kind`: `equality` | `ordering`).
#[macro_export]
#[doc(hidden)]
macro_rules! binary_op_comparison_complex_same {
    (equality, $cmp_op:ident, $a:expr, $a_meta:expr, $b:expr, $b_meta:expr, C64) => {{
        let Some(a_view) = extract_view_c64($a, $a_meta) else {
            set_last_error("Failed to extract Complex64 operand a".to_string());
            return ERR_GENERIC;
        };
        let Some(b_view) = extract_view_c64($b, $b_meta) else {
            set_last_error("Failed to extract Complex64 operand b".to_string());
            return ERR_GENERIC;
        };
        let result = crate::broadcast_binary!(a_view, b_view, $cmp_op);
        NDArrayWrapper {
            data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(result))),
            dtype: DType::Bool,
        }
    }};
    (equality, $cmp_op:ident, $a:expr, $a_meta:expr, $b:expr, $b_meta:expr, C128) => {{
        let Some(a_view) = extract_view_c128($a, $a_meta) else {
            set_last_error("Failed to extract Complex128 operand a".to_string());
            return ERR_GENERIC;
        };
        let Some(b_view) = extract_view_c128($b, $b_meta) else {
            set_last_error("Failed to extract Complex128 operand b".to_string());
            return ERR_GENERIC;
        };
        let result = crate::broadcast_binary!(a_view, b_view, $cmp_op);
        NDArrayWrapper {
            data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(result))),
            dtype: DType::Bool,
        }
    }};
    (ordering, $cmp_op:ident, $a:expr, $a_meta:expr, $b:expr, $b_meta:expr, C64) => {{
        set_last_error(
            "ordering comparison (<, <=, >, >=) is not defined for complex dtypes".to_string(),
        );
        return ERR_GENERIC;
    }};
    (ordering, $cmp_op:ident, $a:expr, $a_meta:expr, $b:expr, $b_meta:expr, C128) => {{
        set_last_error(
            "ordering comparison (<, <=, >, >=) is not defined for complex dtypes".to_string(),
        );
        return ERR_GENERIC;
    }};
}

/// Mixed-dtype branch when the promoted dtype is complex (`$kind`: `equality` | `ordering`).
#[macro_export]
#[doc(hidden)]
macro_rules! binary_op_comparison_complex_mixed {
    (equality, $cmp_op:ident, $a:expr, $a_meta:expr, $b:expr, $b_meta:expr, C64) => {{
        let Some(a_arr) = extract_view_as_c64($a, $a_meta) else {
            set_last_error("Failed to extract operand a as Complex64".to_string());
            return ERR_GENERIC;
        };
        let Some(b_arr) = extract_view_as_c64($b, $b_meta) else {
            set_last_error("Failed to extract operand b as Complex64".to_string());
            return ERR_GENERIC;
        };
        let result = crate::broadcast_binary!(a_arr, b_arr, $cmp_op);
        NDArrayWrapper {
            data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(result))),
            dtype: DType::Bool,
        }
    }};
    (equality, $cmp_op:ident, $a:expr, $a_meta:expr, $b:expr, $b_meta:expr, C128) => {{
        let Some(a_arr) = extract_view_as_c128($a, $a_meta) else {
            set_last_error("Failed to extract operand a as Complex128".to_string());
            return ERR_GENERIC;
        };
        let Some(b_arr) = extract_view_as_c128($b, $b_meta) else {
            set_last_error("Failed to extract operand b as Complex128".to_string());
            return ERR_GENERIC;
        };
        let result = crate::broadcast_binary!(a_arr, b_arr, $cmp_op);
        NDArrayWrapper {
            data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(result))),
            dtype: DType::Bool,
        }
    }};
    (ordering, $cmp_op:ident, $a:expr, $a_meta:expr, $b:expr, $b_meta:expr, C64) => {{
        set_last_error(
            "ordering comparison (<, <=, >, >=) is not defined for complex dtypes".to_string(),
        );
        return ERR_GENERIC;
    }};
    (ordering, $cmp_op:ident, $a:expr, $a_meta:expr, $b:expr, $b_meta:expr, C128) => {{
        set_last_error(
            "ordering comparison (<, <=, >, >=) is not defined for complex dtypes".to_string(),
        );
        return ERR_GENERIC;
    }};
}

/// Binary operation macro for comparison operations.
///
/// Usage:
/// ```rust
/// let result = binary_op_comparison!(a_wrapper, a_meta, b_wrapper, b_meta, eq, equality);
/// let result = binary_op_comparison!(a_wrapper, a_meta, b_wrapper, b_meta, lt, ordering);
/// ```
#[macro_export]
macro_rules! binary_op_comparison {
    ($a_wrapper:expr, $a_meta:expr, $b_wrapper:expr, $b_meta:expr, $cmp_op:ident, $kind:ident) => {{
        #[allow(unused_imports)]
        use crate::helpers::{
            extract_view_as_bool, extract_view_as_c128, extract_view_as_c64, extract_view_as_f32,
            extract_view_as_f64, extract_view_as_i16, extract_view_as_i32, extract_view_as_i64,
            extract_view_as_i8, extract_view_as_u16, extract_view_as_u32, extract_view_as_u64,
            extract_view_as_u8, extract_view_bool, extract_view_c128, extract_view_c64,
            extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32,
            extract_view_i64, extract_view_i8, extract_view_u16, extract_view_u32,
            extract_view_u64, extract_view_u8, set_last_error, ERR_GENERIC,
        };
        use crate::types::dtype::DType;
        use crate::types::{ArrayData, NDArrayWrapper};

        let out_dtype = DType::promote($a_wrapper.dtype, $b_wrapper.dtype);

        if $a_wrapper.dtype == $b_wrapper.dtype {
            // Same dtype: use native views (zero-copy fast path)
            match out_dtype {
                DType::Float64 => {
                    let Some(a_view) = extract_view_f64($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract Float64 operand a".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_view) = extract_view_f64($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract Float64 operand b".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_view, b_view, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Float32 => {
                    let Some(a_view) = extract_view_f32($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract Float32 operand a".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_view) = extract_view_f32($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract Float32 operand b".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_view, b_view, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Int64 => {
                    let Some(a_view) = extract_view_i64($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract Int64 operand a".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_view) = extract_view_i64($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract Int64 operand b".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_view, b_view, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Int32 => {
                    let Some(a_view) = extract_view_i32($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract Int32 operand a".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_view) = extract_view_i32($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract Int32 operand b".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_view, b_view, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Int16 => {
                    let Some(a_view) = extract_view_i16($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract Int16 operand a".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_view) = extract_view_i16($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract Int16 operand b".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_view, b_view, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Int8 => {
                    let Some(a_view) = extract_view_i8($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract Int8 operand a".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_view) = extract_view_i8($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract Int8 operand b".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_view, b_view, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Uint64 => {
                    let Some(a_view) = extract_view_u64($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract Uint64 operand a".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_view) = extract_view_u64($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract Uint64 operand b".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_view, b_view, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Uint32 => {
                    let Some(a_view) = extract_view_u32($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract Uint32 operand a".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_view) = extract_view_u32($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract Uint32 operand b".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_view, b_view, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Uint16 => {
                    let Some(a_view) = extract_view_u16($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract Uint16 operand a".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_view) = extract_view_u16($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract Uint16 operand b".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_view, b_view, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Uint8 => {
                    let Some(a_view) = extract_view_u8($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract Uint8 operand a".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_view) = extract_view_u8($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract Uint8 operand b".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_view, b_view, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Bool => {
                    let Some(a_view) = extract_view_bool($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract Bool operand a".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_view) = extract_view_bool($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract Bool operand b".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_view, b_view, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Complex64 => {
                    $crate::binary_op_comparison_complex_same!(
                        $kind, $cmp_op, $a_wrapper, $a_meta, $b_wrapper, $b_meta, C64
                    )
                }
                DType::Complex128 => {
                    $crate::binary_op_comparison_complex_same!(
                        $kind, $cmp_op, $a_wrapper, $a_meta, $b_wrapper, $b_meta, C128
                    )
                }
            }
        } else {
            // Different dtypes: convert to target type
            match out_dtype {
                DType::Float64 => {
                    let Some(a_arr) = extract_view_as_f64($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract operand a as Float64".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_arr) = extract_view_as_f64($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract operand b as Float64".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_arr, b_arr, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Float32 => {
                    let Some(a_arr) = extract_view_as_f32($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract operand a as Float32".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_arr) = extract_view_as_f32($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract operand b as Float32".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_arr, b_arr, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Int64 => {
                    let Some(a_arr) = extract_view_as_i64($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract operand a as Int64".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_arr) = extract_view_as_i64($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract operand b as Int64".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_arr, b_arr, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Int32 => {
                    let Some(a_arr) = extract_view_as_i32($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract operand a as Int32".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_arr) = extract_view_as_i32($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract operand b as Int32".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_arr, b_arr, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Int16 => {
                    let Some(a_arr) = extract_view_as_i16($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract operand a as Int16".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_arr) = extract_view_as_i16($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract operand b as Int16".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_arr, b_arr, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Int8 => {
                    let Some(a_arr) = extract_view_as_i8($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract operand a as Int8".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_arr) = extract_view_as_i8($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract operand b as Int8".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_arr, b_arr, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Uint64 => {
                    let Some(a_arr) = extract_view_as_u64($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract operand a as Uint64".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_arr) = extract_view_as_u64($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract operand b as Uint64".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_arr, b_arr, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Uint32 => {
                    let Some(a_arr) = extract_view_as_u32($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract operand a as Uint32".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_arr) = extract_view_as_u32($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract operand b as Uint32".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_arr, b_arr, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Uint16 => {
                    let Some(a_arr) = extract_view_as_u16($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract operand a as Uint16".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_arr) = extract_view_as_u16($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract operand b as Uint16".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_arr, b_arr, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Uint8 => {
                    let Some(a_arr) = extract_view_as_u8($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract operand a as Uint8".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_arr) = extract_view_as_u8($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract operand b as Uint8".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_arr, b_arr, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Bool => {
                    let Some(a_arr) = extract_view_as_bool($a_wrapper, $a_meta) else {
                        set_last_error("Failed to extract operand a as Bool".to_string());
                        return ERR_GENERIC;
                    };
                    let Some(b_arr) = extract_view_as_bool($b_wrapper, $b_meta) else {
                        set_last_error("Failed to extract operand b as Bool".to_string());
                        return ERR_GENERIC;
                    };
                    let result = crate::broadcast_binary!(a_arr, b_arr, $cmp_op);
                    NDArrayWrapper {
                        data: ArrayData::Bool(::std::sync::Arc::new(::parking_lot::RwLock::new(
                            result,
                        ))),
                        dtype: DType::Bool,
                    }
                }
                DType::Complex64 => {
                    $crate::binary_op_comparison_complex_mixed!(
                        $kind, $cmp_op, $a_wrapper, $a_meta, $b_wrapper, $b_meta, C64
                    )
                }
                DType::Complex128 => {
                    $crate::binary_op_comparison_complex_mixed!(
                        $kind, $cmp_op, $a_wrapper, $a_meta, $b_wrapper, $b_meta, C128
                    )
                }
            }
        }
    }};
}
