//! Helper functions for stacking operations.

/// Resolve axis for concatenate (supports negative indexing).
/// Axis must be in 0..ndim.
pub fn resolve_axis(shape: &[usize], axis: i32) -> Result<usize, String> {
    let ndim = shape.len();
    if ndim == 0 {
        return Err("Cannot concatenate 0-dimensional array".to_string());
    }
    let axis_i = if axis < 0 {
        (ndim as i32 + axis) as usize
    } else {
        axis as usize
    };
    if axis_i >= ndim {
        return Err(format!(
            "Axis {} is out of bounds for array with {} dimensions",
            axis, ndim
        ));
    }
    Ok(axis_i)
}

/// Resolve axis for stack (insert new axis). Axis can be 0..=ndim.
pub fn resolve_axis_for_stack(shape: &[usize], axis: i32) -> Result<usize, String> {
    let ndim = shape.len();
    let axis_i = if axis < 0 {
        (ndim as i32 + axis + 1) as usize
    } else {
        axis as usize
    };
    if axis_i > ndim {
        return Err(format!(
            "Axis {} is out of bounds for stack (max {})",
            axis, ndim
        ));
    }
    Ok(axis_i)
}
