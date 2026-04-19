//! Helper functions for reduction operations.

use std::ffi::c_void;

use num_complex::{Complex32, Complex64};

use crate::types::dtype::DType;

/// A scalar value produced by a full-array reduction, written to FFI `out_value` / `out_dtype`.
///
/// Callers must ensure `out_value` points to a buffer at least as large as the element type
/// (e.g. 4 bytes for `Float32`, 8 for `Float64`, 16 for `Complex64`, 32 for `Complex128`).
#[derive(Clone, Copy, Debug)]
pub enum ReductionScalar {
    F32(f32),
    F64(f64),
    I64(i64),
    I32(i32),
    I16(i16),
    I8(i8),
    U64(u64),
    U32(u32),
    U16(u16),
    U8(u8),
    C64(Complex32),
    C128(Complex64),
}

impl ReductionScalar {
    #[inline]
    pub const fn dtype(self) -> DType {
        match self {
            Self::F32(_) => DType::Float32,
            Self::F64(_) => DType::Float64,
            Self::I64(_) => DType::Int64,
            Self::I32(_) => DType::Int32,
            Self::I16(_) => DType::Int16,
            Self::I8(_) => DType::Int8,
            Self::U64(_) => DType::Uint64,
            Self::U32(_) => DType::Uint32,
            Self::U16(_) => DType::Uint16,
            Self::U8(_) => DType::Uint8,
            Self::C64(_) => DType::Complex64,
            Self::C128(_) => DType::Complex128,
        }
    }
}

/// Write a reduction scalar to FFI output buffers (`out_value`, `out_dtype`).
#[inline]
pub unsafe fn write_reduction_scalar(
    out_value: *mut c_void,
    out_dtype: *mut u8,
    scalar: ReductionScalar,
) {
    if !out_dtype.is_null() {
        *out_dtype = scalar.dtype() as u8;
    }
    if out_value.is_null() {
        return;
    }
    match scalar {
        ReductionScalar::F64(v) => *(out_value as *mut f64) = v,
        ReductionScalar::F32(v) => *(out_value as *mut f32) = v,
        ReductionScalar::I64(v) => *(out_value as *mut i64) = v,
        ReductionScalar::I32(v) => *(out_value as *mut i32) = v,
        ReductionScalar::I16(v) => *(out_value as *mut i16) = v,
        ReductionScalar::I8(v) => *(out_value as *mut i8) = v,
        ReductionScalar::U64(v) => *(out_value as *mut u64) = v,
        ReductionScalar::U32(v) => *(out_value as *mut u32) = v,
        ReductionScalar::U16(v) => *(out_value as *mut u16) = v,
        ReductionScalar::U8(v) => *(out_value as *mut u8) = v,
        ReductionScalar::C64(v) => *(out_value as *mut Complex32) = v,
        ReductionScalar::C128(v) => *(out_value as *mut Complex64) = v,
    }
}

/// Compute the output shape for axis reduction with keepdims.
pub fn compute_axis_output_shape(shape: &[usize], axis: usize, keepdims: bool) -> Vec<usize> {
    if keepdims {
        shape
            .iter()
            .enumerate()
            .map(|(i, &dim)| if i == axis { 1 } else { dim })
            .collect()
    } else {
        shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, &dim)| dim)
            .collect()
    }
}
