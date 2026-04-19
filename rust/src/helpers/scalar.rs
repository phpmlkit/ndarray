//! Scalar extraction and casting helpers.
//!
//! Provides functions that read a scalar value from a `*const c_void`
//! given its `DType`, and cast it to the target type.

use crate::types::dtype::DType;

/// Read a scalar from a void pointer as f64.
pub unsafe fn get_scalar_as_f64(scalar: *const std::ffi::c_void, scalar_dtype: DType) -> f64 {
    match scalar_dtype {
        DType::Float64 => *(scalar as *const f64),
        DType::Float32 => *(scalar as *const f32) as f64,
        DType::Int64 => *(scalar as *const i64) as f64,
        DType::Int32 => *(scalar as *const i32) as f64,
        DType::Int16 => *(scalar as *const i16) as f64,
        DType::Int8 => *(scalar as *const i8) as f64,
        DType::Uint64 => *(scalar as *const u64) as f64,
        DType::Uint32 => *(scalar as *const u32) as f64,
        DType::Uint16 => *(scalar as *const u16) as f64,
        DType::Uint8 => *(scalar as *const u8) as f64,
        DType::Bool => (*(scalar as *const u8) != 0) as i32 as f64,
        _ => panic!("Invalid scalar dtype for f64 output: {:?}", scalar_dtype),
    }
}

/// Read a scalar from a void pointer as f32.
pub unsafe fn get_scalar_as_f32(scalar: *const std::ffi::c_void, scalar_dtype: DType) -> f32 {
    match scalar_dtype {
        DType::Float32 => *(scalar as *const f32),
        DType::Float64 => *(scalar as *const f64) as f32,
        DType::Int64 => *(scalar as *const i64) as f32,
        DType::Int32 => *(scalar as *const i32) as f32,
        DType::Int16 => *(scalar as *const i16) as f32,
        DType::Int8 => *(scalar as *const i8) as f32,
        DType::Uint64 => *(scalar as *const u64) as f32,
        DType::Uint32 => *(scalar as *const u32) as f32,
        DType::Uint16 => *(scalar as *const u16) as f32,
        DType::Uint8 => *(scalar as *const u8) as f32,
        DType::Bool => (*(scalar as *const u8) != 0) as i32 as f32,
        _ => panic!("Invalid scalar dtype for f32 output: {:?}", scalar_dtype),
    }
}

/// Read a scalar from a void pointer as i64.
pub unsafe fn get_scalar_as_i64(scalar: *const std::ffi::c_void, scalar_dtype: DType) -> i64 {
    match scalar_dtype {
        DType::Int64 => *(scalar as *const i64),
        DType::Int32 => *(scalar as *const i32) as i64,
        DType::Int16 => *(scalar as *const i16) as i64,
        DType::Int8 => *(scalar as *const i8) as i64,
        DType::Uint64 => *(scalar as *const u64) as i64,
        DType::Uint32 => *(scalar as *const u32) as i64,
        DType::Uint16 => *(scalar as *const u16) as i64,
        DType::Uint8 => *(scalar as *const u8) as i64,
        DType::Bool => (*(scalar as *const u8) != 0) as i64,
        _ => panic!("Invalid scalar dtype for i64 output: {:?}", scalar_dtype),
    }
}

/// Read a scalar from a void pointer as i32.
pub unsafe fn get_scalar_as_i32(scalar: *const std::ffi::c_void, scalar_dtype: DType) -> i32 {
    match scalar_dtype {
        DType::Int32 => *(scalar as *const i32),
        DType::Int64 => *(scalar as *const i64) as i32,
        DType::Int16 => *(scalar as *const i16) as i32,
        DType::Int8 => *(scalar as *const i8) as i32,
        DType::Uint32 => *(scalar as *const u32) as i32,
        DType::Uint16 => *(scalar as *const u16) as i32,
        DType::Uint8 => *(scalar as *const u8) as i32,
        DType::Uint64 => *(scalar as *const u64) as i32,
        DType::Bool => (*(scalar as *const u8) != 0) as i32,
        _ => panic!("Invalid scalar dtype for i32 output: {:?}", scalar_dtype),
    }
}

/// Read a scalar from a void pointer as i16.
pub unsafe fn get_scalar_as_i16(scalar: *const std::ffi::c_void, scalar_dtype: DType) -> i16 {
    match scalar_dtype {
        DType::Int16 => *(scalar as *const i16),
        DType::Int8 => *(scalar as *const i8) as i16,
        DType::Int32 => *(scalar as *const i32) as i16,
        DType::Int64 => *(scalar as *const i64) as i16,
        DType::Uint16 => *(scalar as *const u16) as i16,
        DType::Uint8 => *(scalar as *const u8) as i16,
        DType::Uint32 => *(scalar as *const u32) as i16,
        DType::Uint64 => *(scalar as *const u64) as i16,
        DType::Bool => (*(scalar as *const u8) != 0) as i16,
        _ => panic!("Invalid scalar dtype for i16 output: {:?}", scalar_dtype),
    }
}

/// Read a scalar from a void pointer as i8.
pub unsafe fn get_scalar_as_i8(scalar: *const std::ffi::c_void, scalar_dtype: DType) -> i8 {
    match scalar_dtype {
        DType::Int8 => *(scalar as *const i8),
        DType::Int16 => *(scalar as *const i16) as i8,
        DType::Int32 => *(scalar as *const i32) as i8,
        DType::Int64 => *(scalar as *const i64) as i8,
        DType::Uint8 => *(scalar as *const u8) as i8,
        DType::Bool => (*(scalar as *const u8) != 0) as i8,
        _ => panic!("Invalid scalar dtype for i8 output: {:?}", scalar_dtype),
    }
}

/// Read a scalar from a void pointer as u64.
pub unsafe fn get_scalar_as_u64(scalar: *const std::ffi::c_void, scalar_dtype: DType) -> u64 {
    match scalar_dtype {
        DType::Uint64 => *(scalar as *const u64),
        DType::Uint32 => *(scalar as *const u32) as u64,
        DType::Uint16 => *(scalar as *const u16) as u64,
        DType::Uint8 => *(scalar as *const u8) as u64,
        DType::Int64 => *(scalar as *const i64) as u64,
        DType::Int32 => *(scalar as *const i32) as u64,
        DType::Int16 => *(scalar as *const i16) as u64,
        DType::Int8 => *(scalar as *const i8) as u64,
        DType::Bool => (*(scalar as *const u8) != 0) as u64,
        _ => panic!("Invalid scalar dtype for u64 output: {:?}", scalar_dtype),
    }
}

/// Read a scalar from a void pointer as u32.
pub unsafe fn get_scalar_as_u32(scalar: *const std::ffi::c_void, scalar_dtype: DType) -> u32 {
    match scalar_dtype {
        DType::Uint32 => *(scalar as *const u32),
        DType::Uint16 => *(scalar as *const u16) as u32,
        DType::Uint8 => *(scalar as *const u8) as u32,
        DType::Int32 => *(scalar as *const i32) as u32,
        DType::Int16 => *(scalar as *const i16) as u32,
        DType::Int8 => *(scalar as *const i8) as u32,
        DType::Uint64 => *(scalar as *const u64) as u32,
        DType::Int64 => *(scalar as *const i64) as u32,
        DType::Bool => (*(scalar as *const u8) != 0) as u32,
        _ => panic!("Invalid scalar dtype for u32 output: {:?}", scalar_dtype),
    }
}

/// Read a scalar from a void pointer as u16.
pub unsafe fn get_scalar_as_u16(scalar: *const std::ffi::c_void, scalar_dtype: DType) -> u16 {
    match scalar_dtype {
        DType::Uint16 => *(scalar as *const u16),
        DType::Uint8 => *(scalar as *const u8) as u16,
        DType::Int16 => *(scalar as *const i16) as u16,
        DType::Int8 => *(scalar as *const i8) as u16,
        DType::Uint32 => *(scalar as *const u32) as u16,
        DType::Int32 => *(scalar as *const i32) as u16,
        DType::Uint64 => *(scalar as *const u64) as u16,
        DType::Int64 => *(scalar as *const i64) as u16,
        DType::Bool => (*(scalar as *const u8) != 0) as u16,
        _ => panic!("Invalid scalar dtype for u16 output: {:?}", scalar_dtype),
    }
}

/// Read a scalar from a void pointer as u8.
pub unsafe fn get_scalar_as_u8(scalar: *const std::ffi::c_void, scalar_dtype: DType) -> u8 {
    match scalar_dtype {
        DType::Uint8 => *(scalar as *const u8),
        DType::Int8 => *(scalar as *const i8) as u8,
        DType::Bool => *(scalar as *const u8),
        _ => panic!("Invalid scalar dtype for u8 output: {:?}", scalar_dtype),
    }
}

/// Read a scalar from a void pointer as Complex32.
pub unsafe fn get_scalar_as_c64(
    scalar: *const std::ffi::c_void,
    scalar_dtype: DType,
) -> num_complex::Complex32 {
    match scalar_dtype {
        DType::Complex64 => *(scalar as *const num_complex::Complex32),
        DType::Complex128 => {
            let c = *(scalar as *const num_complex::Complex64);
            num_complex::Complex32::new(c.re as f32, c.im as f32)
        }
        DType::Float64 => num_complex::Complex32::new(*(scalar as *const f64) as f32, 0.0),
        DType::Float32 => num_complex::Complex32::new(*(scalar as *const f32), 0.0),
        DType::Int64 => num_complex::Complex32::new(*(scalar as *const i64) as f32, 0.0),
        DType::Int32 => num_complex::Complex32::new(*(scalar as *const i32) as f32, 0.0),
        DType::Int16 => num_complex::Complex32::new(*(scalar as *const i16) as f32, 0.0),
        DType::Int8 => num_complex::Complex32::new(*(scalar as *const i8) as f32, 0.0),
        DType::Uint64 => num_complex::Complex32::new(*(scalar as *const u64) as f32, 0.0),
        DType::Uint32 => num_complex::Complex32::new(*(scalar as *const u32) as f32, 0.0),
        DType::Uint16 => num_complex::Complex32::new(*(scalar as *const u16) as f32, 0.0),
        DType::Uint8 => num_complex::Complex32::new(*(scalar as *const u8) as f32, 0.0),
        DType::Bool => {
            num_complex::Complex32::new((*(scalar as *const u8) != 0) as i32 as f32, 0.0)
        }
    }
}

/// Read a scalar from a void pointer as Complex64.
pub unsafe fn get_scalar_as_c128(
    scalar: *const std::ffi::c_void,
    scalar_dtype: DType,
) -> num_complex::Complex64 {
    match scalar_dtype {
        DType::Complex128 => *(scalar as *const num_complex::Complex64),
        DType::Complex64 => {
            let c = *(scalar as *const num_complex::Complex32);
            num_complex::Complex64::new(c.re as f64, c.im as f64)
        }
        DType::Float64 => num_complex::Complex64::new(*(scalar as *const f64), 0.0),
        DType::Float32 => num_complex::Complex64::new(*(scalar as *const f32) as f64, 0.0),
        DType::Int64 => num_complex::Complex64::new(*(scalar as *const i64) as f64, 0.0),
        DType::Int32 => num_complex::Complex64::new(*(scalar as *const i32) as f64, 0.0),
        DType::Int16 => num_complex::Complex64::new(*(scalar as *const i16) as f64, 0.0),
        DType::Int8 => num_complex::Complex64::new(*(scalar as *const i8) as f64, 0.0),
        DType::Uint64 => num_complex::Complex64::new(*(scalar as *const u64) as f64, 0.0),
        DType::Uint32 => num_complex::Complex64::new(*(scalar as *const u32) as f64, 0.0),
        DType::Uint16 => num_complex::Complex64::new(*(scalar as *const u16) as f64, 0.0),
        DType::Uint8 => num_complex::Complex64::new(*(scalar as *const u8) as f64, 0.0),
        DType::Bool => {
            num_complex::Complex64::new((*(scalar as *const u8) != 0) as i32 as f64, 0.0)
        }
    }
}
