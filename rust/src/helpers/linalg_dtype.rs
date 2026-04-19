//! Dtype selection for linear algebra after [`DType::promote`].

use crate::types::dtype::DType;

/// Dtype used for `matmul` / `dot` computation after [`DType::promote`].
///
/// Integer dtypes are computed in `Float64` (values promoted via `extract_view_as_f64`).
/// [`DType::Bool`] is unsupported.
pub fn linalg_computation_dtype(promoted: DType) -> Option<DType> {
    match promoted {
        DType::Bool => None,
        DType::Int8
        | DType::Int16
        | DType::Int32
        | DType::Int64
        | DType::Uint8
        | DType::Uint16
        | DType::Uint32
        | DType::Uint64 => Some(DType::Float64),
        DType::Float32 | DType::Float64 | DType::Complex64 | DType::Complex128 => Some(promoted),
    }
}
