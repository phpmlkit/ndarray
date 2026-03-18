//! Indexing helpers for array operations.
//!
//! This module provides utilities for validating and normalizing
//! axis indices and element indices within arrays.

/// Validate and normalize an axis index for array operations.
///
/// Supports negative indexing (e.g., -1 means last axis) and validates
/// that the axis is within the valid range for the array's dimensions.
///
/// # Arguments
/// * `shape` - The shape of the array
/// * `axis` - The axis index (can be negative)
/// * `allow_insert` - If true, allows axis == ndim (for operations like stack that insert new axes)
///
/// # Returns
/// * `Ok(usize)` - The normalized axis index
/// * `Err(String)` - Error message if axis is out of bounds
///
/// # Examples
/// ```
/// // For a 3D array (ndim=3)
/// normalize_axis(&[2, 3, 4], 0, false);   // Ok(0) - first axis
/// normalize_axis(&[2, 3, 4], -1, false);  // Ok(2) - last axis
/// normalize_axis(&[2, 3, 4], 3, false);   // Err - axis 3 out of bounds
/// normalize_axis(&[2, 3, 4], 3, true);    // Ok(3) - valid for insert operations
/// ```
pub fn normalize_axis(shape: &[usize], axis: i32, allow_insert: bool) -> Result<usize, String> {
    let ndim = shape.len();
    let max_axis = if allow_insert {
        ndim
    } else {
        ndim.saturating_sub(1)
    };

    let axis_usize = if axis < 0 {
        (ndim as i32 + axis) as usize
    } else {
        axis as usize
    };

    if axis_usize > max_axis {
        return Err(format!(
            "Axis {} is out of bounds for array with {} dimensions (valid range: 0 to {})",
            axis, ndim, max_axis
        ));
    }

    Ok(axis_usize)
}

/// Normalize an element index within a specific axis/dimension.
///
/// Supports negative indexing (e.g., -1 means last element) and validates
/// that the index is within the bounds of the given length.
///
/// # Arguments
/// * `idx` - The element index (can be negative)
/// * `len` - The length of the axis/dimension
///
/// # Returns
/// * `Ok(usize)` - The normalized index
/// * `Err(String)` - Error message if index is out of bounds
///
/// # Examples
/// ```
/// normalize_index(2, 5);   // Ok(2) - valid index
/// normalize_index(-1, 5);  // Ok(4) - last element
/// normalize_index(5, 5);   // Err - index 5 out of bounds for length 5
/// ```
#[inline]
pub fn normalize_index(idx: i64, len: usize) -> Result<usize, String> {
    let mut i = idx;
    if i < 0 {
        i += len as i64;
    }
    if i < 0 || i >= len as i64 {
        return Err(format!(
            "Index {} is out of bounds for axis with size {}",
            idx, len
        ));
    }
    Ok(i as usize)
}
