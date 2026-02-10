use ndarray::{s, Array2, ArrayD, IxDyn};

#[test]
fn test_array_ones_existence() {
    // Check ArrayD::ones
    let shape = IxDyn(&[2, 2]);
    let a: ArrayD<f64> = ArrayD::ones(shape);
    assert_eq!(a[[0, 0]], 1.0);
    assert_eq!(a.sum(), 4.0);

    // Check Array2::ones (ArrayBase convenience)
    let b: Array2<f64> = Array2::ones((2, 2));
    assert_eq!(b[[0, 0]], 1.0);
    assert_eq!(b.sum(), 4.0);
}

#[test]
fn test_slice_mut_diag_behavior() {
    // Create 3x3 zeros
    let mut arr = Array2::<f64>::zeros((3, 3));

    // Case 1: k=1 (super-diagonal)
    // Strategy: Slice [.., 1..] to shift columns left by 1.
    // The main diagonal of this new view corresponds to (i, i+1) in original.
    {
        let mut view = arr.slice_mut(s![.., 1..]);
        // view is 3x2.
        // diag is min(3, 2) = 2 elements.
        // view[(0,0)] -> arr[(0,1)]
        // view[(1,1)] -> arr[(1,2)]
        view.diag_mut().fill(1.0);
    }

    assert_eq!(arr[[0, 1]], 1.0, "Element (0,1) should be 1.0");
    assert_eq!(arr[[1, 2]], 1.0, "Element (1,2) should be 1.0");
    assert_eq!(arr[[0, 0]], 0.0);
    assert_eq!(arr[[1, 1]], 0.0);
    assert_eq!(arr[[2, 2]], 0.0);

    // Verify we didn't touch anything else
    assert_eq!(arr.sum(), 2.0);
}

#[test]
fn test_slice_mut_diag_k_negative() {
    let mut arr = Array2::<f64>::zeros((3, 3));

    // Case 2: k=-1 (sub-diagonal)
    // Strategy: Slice [1.., ..] to shift rows up by 1.
    // The main diagonal of this new view corresponds to (i+1, i) in original.
    {
        let mut view = arr.slice_mut(s![1.., ..]);
        // view is 2x3.
        // diag is min(2, 3) = 2 elements.
        // view[(0,0)] -> arr[(1,0)]
        // view[(1,1)] -> arr[(2,1)]
        view.diag_mut().fill(2.0);
    }

    assert_eq!(arr[[1, 0]], 2.0);
    assert_eq!(arr[[2, 1]], 2.0);
    assert_eq!(arr.sum(), 4.0);
}

#[test]
fn test_slice_mut_diag_out_of_bounds() {
    let mut arr = Array2::<f64>::zeros((3, 3));

    // Case 3: k=5 (out of bounds)
    // "Slice begin 5 is past end of axis of length 3" -> This panics.
    // So we must verify it panics or handle it.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut view = arr.slice_mut(s![.., 5..]);
        view.diag_mut().fill(3.0);
    }));

    assert!(result.is_err(), "Out of bounds slice should panic");
}
