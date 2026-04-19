//! Shared validation and LAPACK layout helpers for linear algebra FFI.

use ndarray::Ix2;
use ndarray_linalg::UPLO;

use crate::helpers::error::{self, ERR_SHAPE, SUCCESS};
use crate::types::ArrayMetadata;

/// Validate that metadata describes a 2D square matrix.
pub fn validate_square_matrix(a_meta_ref: &ArrayMetadata, op_name: &str) -> i32 {
    if a_meta_ref.ndim != 2 {
        error::set_last_error(format!("{} requires a 2D matrix", op_name));
        return ERR_SHAPE;
    }
    let shape = unsafe { a_meta_ref.shape_slice() };
    if shape[0] != shape[1] {
        error::set_last_error(format!("{} requires a square matrix", op_name));
        return ERR_SHAPE;
    }
    SUCCESS
}

/// Convert FFI `uplo` byte to ndarray-linalg [`UPLO`].
///
/// `0` = lower triangle, `1` = upper triangle.
pub fn uplo_from_int(uplo: u8) -> UPLO {
    if uplo != 0 {
        UPLO::Upper
    } else {
        UPLO::Lower
    }
}

/// Adjust [`UPLO`] for C-layout Hermitian matrices before calling LAPACK.
///
/// ndarray-linalg transposes C-layout matrices to Fortran layout before LAPACK;
/// transposing swaps upper/lower, so we flip `uplo` for standard-layout inputs.
pub fn adjust_uplo_for_layout<A, S>(a: &ndarray::ArrayBase<S, Ix2>, uplo: UPLO) -> UPLO
where
    S: ndarray::Data<Elem = A>,
{
    if a.is_standard_layout() {
        match uplo {
            UPLO::Upper => UPLO::Lower,
            UPLO::Lower => UPLO::Upper,
        }
    } else {
        uplo
    }
}
