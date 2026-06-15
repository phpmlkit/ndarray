use ndarray::{ArrayD, ArrayViewD, IxDyn};

#[test]
fn test_view_with_custom_strides_operations() {
    // Create a contiguous [3,3] array: [[1,2,3],[4,5,6],[7,8,9]]
    let arr = ArrayD::from_shape_vec(
        IxDyn(&[3usize, 3usize]),
        vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .unwrap();

    println!("Original array:\n{:?}", arr);
    println!(
        "Original shape: {:?}, strides: {:?}",
        arr.shape(),
        arr.strides()
    );

    // Create t() equivalent: shape [3,3] with custom strides [1,3]
    let shape = IxDyn(&[3usize, 3usize]);
    let strides = IxDyn(&[1usize, 3usize]);
    unsafe {
        let view = ArrayViewD::<f64>::from_shape_ptr(shape.strides(strides), arr.as_ptr());
        println!(
            "\nView shape: {:?}, strides: {:?}",
            view.shape(),
            view.strides()
        );
        println!("view[[0,0]]={} view[[0,1]]={}", view[[0, 0]], view[[0, 1]]);
        println!("view[[1,0]]={} view[[1,1]]={}", view[[1, 0]], view[[1, 1]]);

        // Test to_owned()
        let owned = view.to_owned();
        println!(
            "\nto_owned() shape: {:?} strides: {:?}",
            owned.shape(),
            owned.strides()
        );
        println!("owned[[0,0]]={} [0,1]={}", owned[[0, 0]], owned[[0, 1]]);
        println!("owned[[1,0]]={} [1,0]={}", owned[[1, 0]], owned[[1, 1]]);

        // Check if data is correct
        assert!(
            (owned[[0, 0]] - 1.0).abs() < 1e-10,
            "to_owned[[0,0]] should be 1.0 but got {}",
            owned[[0, 0]]
        );
        assert!(
            (owned[[0, 1]] - 4.0).abs() < 1e-10,
            "to_owned[[0,1]] should be 4.0 but got {}",
            owned[[0, 1]]
        );
        assert!(
            (owned[[1, 0]] - 2.0).abs() < 1e-10,
            "to_owned[[1,0]] should be 2.0 but got {}",
            owned[[1, 0]]
        );
        println!("to_owned() assertion PASSED");
    }
}
