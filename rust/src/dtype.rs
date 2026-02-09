//! Data type definitions for NDArray elements.
//!
//! This module defines the DType enum and utilities for type handling
//! across the FFI boundary. Integer values MUST stay in sync with PHP's
//! DType enum in src/DType.php.

use std::fmt;

/// Data type enumeration for NDArray elements.
///
/// Integer values are used for FFI and must match PHP's DType enum exactly.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    // Signed integers
    Int8 = 0,
    Int16 = 1,
    Int32 = 2,
    Int64 = 3,

    // Unsigned integers
    Uint8 = 4,
    Uint16 = 5,
    Uint32 = 6,
    Uint64 = 7,

    // Floating-point
    Float32 = 8,
    Float64 = 9,

    // Boolean
    Bool = 10,
}

impl DType {
    /// Create DType from FFI integer value. Returns None for invalid values.
    #[inline]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(DType::Int8),
            1 => Some(DType::Int16),
            2 => Some(DType::Int32),
            3 => Some(DType::Int64),
            4 => Some(DType::Uint8),
            5 => Some(DType::Uint16),
            6 => Some(DType::Uint32),
            7 => Some(DType::Uint64),
            8 => Some(DType::Float32),
            9 => Some(DType::Float64),
            10 => Some(DType::Bool),
            _ => None,
        }
    }

    /// Convert to FFI integer value.
    #[inline]
    pub fn to_u8(self) -> u8 {
        self as u8
    }

    /// Get the size of each element in bytes.
    #[inline]
    pub const fn item_size(self) -> usize {
        match self {
            DType::Int8 | DType::Uint8 | DType::Bool => 1,
            DType::Int16 | DType::Uint16 => 2,
            DType::Int32 | DType::Uint32 | DType::Float32 => 4,
            DType::Int64 | DType::Uint64 | DType::Float64 => 8,
        }
    }

    /// Check if this is a signed integer type.
    #[inline]
    pub const fn is_signed(self) -> bool {
        matches!(
            self,
            DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64
        )
    }

    /// Check if this is an unsigned integer type.
    #[inline]
    pub const fn is_unsigned(self) -> bool {
        matches!(
            self,
            DType::Uint8 | DType::Uint16 | DType::Uint32 | DType::Uint64
        )
    }

    /// Check if this is an integer type (signed or unsigned).
    #[inline]
    pub const fn is_integer(self) -> bool {
        self.is_signed() || self.is_unsigned()
    }

    /// Check if this is a floating-point type.
    #[inline]
    pub const fn is_float(self) -> bool {
        matches!(self, DType::Float32 | DType::Float64)
    }

    /// Check if this is a boolean type.
    #[inline]
    pub const fn is_bool(self) -> bool {
        matches!(self, DType::Bool)
    }

    /// Check if this is a numeric type (integer or float).
    #[inline]
    pub const fn is_numeric(self) -> bool {
        !self.is_bool()
    }

    /// Get a human-readable name for this dtype.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            DType::Int8 => "int8",
            DType::Int16 => "int16",
            DType::Int32 => "int32",
            DType::Int64 => "int64",
            DType::Uint8 => "uint8",
            DType::Uint16 => "uint16",
            DType::Uint32 => "uint32",
            DType::Uint64 => "uint64",
            DType::Float32 => "float32",
            DType::Float64 => "float64",
            DType::Bool => "bool",
        }
    }

    /// Determine the result dtype when two dtypes are combined.
    /// Follows NumPy type promotion rules.
    pub fn promote(a: DType, b: DType) -> DType {
        if a == b {
            return a;
        }

        if a == DType::Float64 || b == DType::Float64 {
            return DType::Float64;
        }

        if a == DType::Float32 || b == DType::Float32 {
            return DType::Float32;
        }

        if a == DType::Int64 || b == DType::Int64 {
            return DType::Int64;
        }

        if (a == DType::Uint64 && b.is_signed()) || (b == DType::Uint64 && a.is_signed()) {
            return DType::Float64;
        }

        if a.item_size() >= b.item_size() {
            a
        } else {
            b
        }
    }
}

impl Default for DType {
    fn default() -> Self {
        DType::Float64
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Error type for dtype-related operations.
#[derive(Debug, Clone)]
pub struct DTypeError {
    pub message: String,
}

impl DTypeError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn invalid_dtype(value: u8) -> Self {
        Self::new(format!("Invalid dtype value: {}", value))
    }

    pub fn type_mismatch(expected: DType, got: DType) -> Self {
        Self::new(format!(
            "Type mismatch: expected {}, got {}",
            expected.name(),
            got.name()
        ))
    }

    pub fn unsupported_operation(dtype: DType, operation: &str) -> Self {
        Self::new(format!(
            "Unsupported operation '{}' for dtype {}",
            operation,
            dtype.name()
        ))
    }
}

impl fmt::Display for DTypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DTypeError: {}", self.message)
    }
}

impl std::error::Error for DTypeError {}

// ============================================================================
// FFI Functions
// ============================================================================

/// Get the item size for a dtype. Returns 0 for invalid dtype values.
#[no_mangle]
pub extern "C" fn dtype_item_size(dtype: u8) -> usize {
    DType::from_u8(dtype).map_or(0, |d| d.item_size())
}

/// Check if a dtype value is valid.
#[no_mangle]
pub extern "C" fn dtype_is_valid(dtype: u8) -> bool {
    DType::from_u8(dtype).is_some()
}

/// Check if a dtype is a signed integer.
#[no_mangle]
pub extern "C" fn dtype_is_signed(dtype: u8) -> bool {
    DType::from_u8(dtype).map_or(false, |d| d.is_signed())
}

/// Check if a dtype is an unsigned integer.
#[no_mangle]
pub extern "C" fn dtype_is_unsigned(dtype: u8) -> bool {
    DType::from_u8(dtype).map_or(false, |d| d.is_unsigned())
}

/// Check if a dtype is an integer type.
#[no_mangle]
pub extern "C" fn dtype_is_integer(dtype: u8) -> bool {
    DType::from_u8(dtype).map_or(false, |d| d.is_integer())
}

/// Check if a dtype is a floating-point type.
#[no_mangle]
pub extern "C" fn dtype_is_float(dtype: u8) -> bool {
    DType::from_u8(dtype).map_or(false, |d| d.is_float())
}

/// Check if a dtype is a boolean type.
#[no_mangle]
pub extern "C" fn dtype_is_bool(dtype: u8) -> bool {
    DType::from_u8(dtype).map_or(false, |d| d.is_bool())
}

/// Get the promoted dtype when combining two dtypes.
/// Returns 255 (invalid) if either dtype is invalid.
#[no_mangle]
pub extern "C" fn dtype_promote(a: u8, b: u8) -> u8 {
    match (DType::from_u8(a), DType::from_u8(b)) {
        (Some(da), Some(db)) => DType::promote(da, db).to_u8(),
        _ => 255,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_from_u8() {
        assert_eq!(DType::from_u8(0), Some(DType::Int8));
        assert_eq!(DType::from_u8(9), Some(DType::Float64));
        assert_eq!(DType::from_u8(10), Some(DType::Bool));
        assert_eq!(DType::from_u8(11), None);
        assert_eq!(DType::from_u8(255), None);
    }

    #[test]
    fn test_dtype_to_u8() {
        assert_eq!(DType::Int8.to_u8(), 0);
        assert_eq!(DType::Float64.to_u8(), 9);
        assert_eq!(DType::Bool.to_u8(), 10);
    }

    #[test]
    fn test_item_size() {
        assert_eq!(DType::Int8.item_size(), 1);
        assert_eq!(DType::Int16.item_size(), 2);
        assert_eq!(DType::Int32.item_size(), 4);
        assert_eq!(DType::Int64.item_size(), 8);
        assert_eq!(DType::Float32.item_size(), 4);
        assert_eq!(DType::Float64.item_size(), 8);
        assert_eq!(DType::Bool.item_size(), 1);
    }

    #[test]
    fn test_type_checks() {
        assert!(DType::Int32.is_signed());
        assert!(!DType::Uint32.is_signed());
        assert!(DType::Uint32.is_unsigned());
        assert!(DType::Int32.is_integer());
        assert!(DType::Uint32.is_integer());
        assert!(!DType::Float64.is_integer());
        assert!(DType::Float64.is_float());
        assert!(DType::Float32.is_float());
        assert!(DType::Bool.is_bool());
        assert!(!DType::Float64.is_bool());
    }

    #[test]
    fn test_promotion() {
        assert_eq!(
            DType::promote(DType::Float64, DType::Float64),
            DType::Float64
        );
        assert_eq!(DType::promote(DType::Float64, DType::Int32), DType::Float64);
        assert_eq!(DType::promote(DType::Int32, DType::Float64), DType::Float64);
        assert_eq!(DType::promote(DType::Float32, DType::Int32), DType::Float32);
        assert_eq!(DType::promote(DType::Int64, DType::Int32), DType::Int64);
        assert_eq!(DType::promote(DType::Uint64, DType::Int32), DType::Float64);
    }

    #[test]
    fn test_ffi_functions() {
        assert_eq!(dtype_item_size(9), 8);
        assert_eq!(dtype_item_size(0), 1);
        assert_eq!(dtype_item_size(255), 0);

        assert!(dtype_is_valid(0));
        assert!(dtype_is_valid(10));
        assert!(!dtype_is_valid(11));

        assert!(dtype_is_signed(0));
        assert!(!dtype_is_signed(4));
        assert!(dtype_is_float(9));

        assert_eq!(dtype_promote(9, 2), 9);
        assert_eq!(dtype_promote(255, 0), 255);
    }
}
