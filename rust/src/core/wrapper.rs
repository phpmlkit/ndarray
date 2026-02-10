//! NDArray wrapper providing type-safe array storage.
//!
//! This module wraps ndarray's ArrayD with support for multiple data types
//! and provides the core array operations.

use crate::core::ArrayData;
use crate::dtype::DType;

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

    /// Get data as f64 slice (for FFI - converts from actual type).
    pub fn to_f64_vec(&self) -> Vec<f64> {
        match_array_data!(self.data, arr => {
            arr.read().iter().map(|&x| x as f64).collect()
        })
    }

    /// Check if this wrapper points to the same underlying array data as another.
    pub fn is_same_array(&self, other: &Self) -> bool {
        use crate::core::ArrayData::*;
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

// Generate to_vec_* methods for type-safe data access
crate::impl_to_vec!(
    to_int8_vec, i8, Int8;
    to_int16_vec, i16, Int16;
    to_int32_vec, i32, Int32;
    to_int64_vec, i64, Int64;
    to_uint8_vec, u8, Uint8;
    to_uint16_vec, u16, Uint16;
    to_uint32_vec, u32, Uint32;
    to_uint64_vec, u64, Uint64;
    to_float32_vec, f32, Float32;
    to_float64_vec, f64, Float64;
    to_bool_vec, u8, Bool
);

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
    get_element_bool, u8, Bool
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
    set_element_bool, u8, Bool
);

// Generate fill_slice_* methods for view filling
crate::impl_fill_slice!(
    fill_slice_i8, i8, Int8;
    fill_slice_i16, i16, Int16;
    fill_slice_i32, i32, Int32;
    fill_slice_i64, i64, Int64;
    fill_slice_u8, u8, Uint8;
    fill_slice_u16, u16, Uint16;
    fill_slice_u32, u32, Uint32;
    fill_slice_u64, u64, Uint64;
    fill_slice_f32, f32, Float32;
    fill_slice_f64, f64, Float64;
    fill_slice_bool, u8, Bool
);

// Generate assign_slice_* methods for array assignment
crate::impl_assign_slice!(
    assign_slice_i8, i8, Int8;
    assign_slice_i16, i16, Int16;
    assign_slice_i32, i32, Int32;
    assign_slice_i64, i64, Int64;
    assign_slice_u8, u8, Uint8;
    assign_slice_u16, u16, Uint16;
    assign_slice_u32, u32, Uint32;
    assign_slice_u64, u64, Uint64;
    assign_slice_f32, f32, Float32;
    assign_slice_f64, f64, Float64;
    assign_slice_bool, u8, Bool
);

// Generate copy_view_* methods for creating owned arrays from views
crate::impl_copy_view!(
    copy_view_i8, i8, Int8, Int8;
    copy_view_i16, i16, Int16, Int16;
    copy_view_i32, i32, Int32, Int32;
    copy_view_i64, i64, Int64, Int64;
    copy_view_u8, u8, Uint8, Uint8;
    copy_view_u16, u16, Uint16, Uint16;
    copy_view_u32, u32, Uint32, Uint32;
    copy_view_u64, u64, Uint64, Uint64;
    copy_view_f32, f32, Float32, Float32;
    copy_view_f64, f64, Float64, Float64;
    copy_view_bool, u8, Bool, Bool
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

    #[test]
    fn test_to_f64_vec() {
        let data = vec![1i32, 2, 3, 4];
        let shape = vec![2, 2];
        let arr = NDArrayWrapper::from_slice_i32(&data, &shape).unwrap();

        let f64_data = arr.to_f64_vec();
        assert_eq!(f64_data, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
