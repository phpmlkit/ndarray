//! NDArray wrapper providing type-safe array storage.
//!
//! This module wraps ndarray's ArrayD with support for multiple data types
//! and provides the core array operations.

use crate::match_array_data;
use crate::types::dtype::DType;
use crate::types::ArrayData;

/// Main wrapper around ndarray with type information.
pub struct NDArrayWrapper {
    pub data: ArrayData,
    pub dtype: DType,
}

impl NDArrayWrapper {
    /// Get the shape of the array.
    pub fn shape(&self) -> Vec<usize> {
        match_array_data!(self.data, arr => {
            arr.read().shape().to_vec()
        })
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// Get the total number of elements.
    pub fn len(&self) -> usize {
        self.shape().iter().product()
    }

    /// Check if the array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if this wrapper points to the same underlying array data as another.
    pub fn is_same_array(&self, other: &Self) -> bool {
        use crate::types::ArrayData::*;
        use std::sync::Arc;

        match (&self.data, &other.data) {
            (Int8(a), Int8(b)) => Arc::ptr_eq(a, b),
            (Int16(a), Int16(b)) => Arc::ptr_eq(a, b),
            (Int32(a), Int32(b)) => Arc::ptr_eq(a, b),
            (Int64(a), Int64(b)) => Arc::ptr_eq(a, b),
            (Uint8(a), Uint8(b)) => Arc::ptr_eq(a, b),
            (Uint16(a), Uint16(b)) => Arc::ptr_eq(a, b),
            (Uint32(a), Uint32(b)) => Arc::ptr_eq(a, b),
            (Uint64(a), Uint64(b)) => Arc::ptr_eq(a, b),
            (Float32(a), Float32(b)) => Arc::ptr_eq(a, b),
            (Float64(a), Float64(b)) => Arc::ptr_eq(a, b),
            (Bool(a), Bool(b)) => Arc::ptr_eq(a, b),
            (Complex64(a), Complex64(b)) => Arc::ptr_eq(a, b),
            (Complex128(a), Complex128(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

// Generate all from_slice_* methods using the macro
crate::impl_from_slice!(
    from_slice_i8, i8, Int8, Int8;
    from_slice_i16, i16, Int16, Int16;
    from_slice_i32, i32, Int32, Int32;
    from_slice_i64, i64, Int64, Int64;
    from_slice_u8, u8, Uint8, Uint8;
    from_slice_u16, u16, Uint16, Uint16;
    from_slice_u32, u32, Uint32, Uint32;
    from_slice_u64, u64, Uint64, Uint64;
    from_slice_f32, f32, Float32, Float32;
    from_slice_f64, f64, Float64, Float64;
    from_slice_bool, u8, Bool, Bool
);

impl NDArrayWrapper {
    /// Create array from flat f32 slice interpreted as Complex64 pairs.
    ///
    /// The input slice must have an even number of elements where each pair
    /// (re, im) forms one complex number.
    pub fn from_slice_complex64(data: &[f32], shape: &[usize]) -> Result<Self, String> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len * 2 {
            return Err(format!(
                "Data length {} does not match shape {:?} for complex64 (expected {} floats)",
                data.len(),
                shape,
                expected_len * 2
            ));
        }

        // Safe because Complex<f32> is #[repr(C)] with layout identical to [f32; 2]
        let complex_data: &[num_complex::Complex<f32>] = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const num_complex::Complex<f32>,
                expected_len,
            )
        };

        let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(shape), complex_data.to_vec())
            .map_err(|e| format!("Shape error: {}", e))?;

        Ok(Self {
            data: crate::types::ArrayData::Complex64(std::sync::Arc::new(
                parking_lot::RwLock::new(arr),
            )),
            dtype: crate::types::dtype::DType::Complex64,
        })
    }

    /// Create array from flat f64 slice interpreted as Complex128 pairs.
    ///
    /// The input slice must have an even number of elements where each pair
    /// (re, im) forms one complex number.
    pub fn from_slice_complex128(data: &[f64], shape: &[usize]) -> Result<Self, String> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len * 2 {
            return Err(format!(
                "Data length {} does not match shape {:?} for complex128 (expected {} doubles)",
                data.len(),
                shape,
                expected_len * 2
            ));
        }

        let complex_data: &[num_complex::Complex<f64>] = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const num_complex::Complex<f64>,
                expected_len,
            )
        };

        let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(shape), complex_data.to_vec())
            .map_err(|e| format!("Shape error: {}", e))?;

        Ok(Self {
            data: crate::types::ArrayData::Complex128(std::sync::Arc::new(
                parking_lot::RwLock::new(arr),
            )),
            dtype: crate::types::dtype::DType::Complex128,
        })
    }
}

// Generate get_element_* methods for single-element access
crate::impl_get_element!(
    get_element_i8, i8, Int8;
    get_element_i16, i16, Int16;
    get_element_i32, i32, Int32;
    get_element_i64, i64, Int64;
    get_element_u8, u8, Uint8;
    get_element_u16, u16, Uint16;
    get_element_u32, u32, Uint32;
    get_element_u64, u64, Uint64;
    get_element_f32, f32, Float32;
    get_element_f64, f64, Float64;
    get_element_bool, u8, Bool;
    get_element_complex64, num_complex::Complex<f32>, Complex64;
    get_element_complex128, num_complex::Complex<f64>, Complex128
);

// Generate set_element_* methods for single-element mutation
crate::impl_set_element!(
    set_element_i8, i8, Int8;
    set_element_i16, i16, Int16;
    set_element_i32, i32, Int32;
    set_element_i64, i64, Int64;
    set_element_u8, u8, Uint8;
    set_element_u16, u16, Uint16;
    set_element_u32, u32, Uint32;
    set_element_u64, u64, Uint64;
    set_element_f32, f32, Float32;
    set_element_f64, f64, Float64;
    set_element_bool, u8, Bool;
    set_element_complex64, num_complex::Complex<f32>, Complex64;
    set_element_complex128, num_complex::Complex<f64>, Complex128
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_slice_f64() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let arr = NDArrayWrapper::from_slice_f64(&data, &shape).unwrap();

        assert_eq!(arr.dtype, DType::Float64);
        assert_eq!(arr.shape(), vec![2, 3]);
        assert_eq!(arr.ndim(), 2);
        assert_eq!(arr.len(), 6);
    }

    #[test]
    fn test_from_slice_i64() {
        let data = vec![1i64, 2, 3, 4];
        let shape = vec![4];
        let arr = NDArrayWrapper::from_slice_i64(&data, &shape).unwrap();

        assert_eq!(arr.dtype, DType::Int64);
        assert_eq!(arr.shape(), vec![4]);
    }

    #[test]
    fn test_shape_mismatch() {
        let data = vec![1.0, 2.0, 3.0];
        let shape = vec![2, 2]; // expects 4 elements
        let result = NDArrayWrapper::from_slice_f64(&data, &shape);

        assert!(result.is_err());
    }
}
