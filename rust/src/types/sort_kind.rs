//! Sorting algorithm type for array sorting operations.

/// Sorting algorithm selection for sort/argsort/topk operations.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortKind {
    QuickSort = 0,
    MergeSort = 1,
    HeapSort = 2,
    Stable = 3,
}

impl SortKind {
    /// Parse SortKind from FFI integer value.
    pub fn from_i32(value: i32) -> Result<Self, String> {
        match value {
            0 => Ok(SortKind::QuickSort),
            1 => Ok(SortKind::MergeSort),
            2 => Ok(SortKind::HeapSort),
            3 => Ok(SortKind::Stable),
            _ => Err(format!("Invalid sort kind: {}", value)),
        }
    }
}
