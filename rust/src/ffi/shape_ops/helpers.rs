//! Helper functions for shape operations.

/// Padding mode for ndarray_pad.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadMode {
    Constant = 0,
    Symmetric = 1,
    Reflect = 2,
}

impl PadMode {
    /// Parse PadMode from FFI integer value.
    pub fn from_i32(value: i32) -> Result<Self, String> {
        match value {
            0 => Ok(PadMode::Constant),
            1 => Ok(PadMode::Symmetric),
            2 => Ok(PadMode::Reflect),
            _ => Err(format!("Invalid pad mode: {}", value)),
        }
    }
}
