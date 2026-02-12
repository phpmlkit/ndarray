//! Helper functions for reduction operations.

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

/// Validate axis is within bounds.
pub fn validate_axis(shape: &[usize], axis: i32) -> Result<usize, String> {
    let ndim = shape.len();
    if ndim == 0 {
        return Err("Cannot reduce 0-dimensional array along axis".to_string());
    }

    let axis_usize = if axis < 0 {
        (ndim as i32 + axis) as usize
    } else {
        axis as usize
    };

    if axis_usize >= ndim {
        return Err(format!(
            "Axis {} is out of bounds for array with {} dimensions",
            axis, ndim
        ));
    }

    Ok(axis_usize)
}
