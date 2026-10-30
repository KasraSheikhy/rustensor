use rustensor::{broadcast_shape, Error, Tensor};

fn tensor(data: &[f64], shape: &[usize]) -> Tensor {
    Tensor::from_vec(data.to_vec(), shape.to_vec()).unwrap()
}

fn assert_tensor(actual: &Tensor, shape: &[usize], data: &[f64]) {
    assert_eq!(actual.shape(), shape);
    assert_eq!(actual.data(), data);
}

#[test]
fn construction_preserves_shape_and_row_major_data() {
    let value = tensor(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    assert_tensor(&value, &[2, 3], &[1., 2., 3., 4., 5., 6.]);
}

#[test]
fn construction_rejects_mismatched_element_counts() {
    for (data, shape) in [
        (vec![1., 2.], vec![2, 2]),
        (vec![], vec![]),
        (vec![1., 2.], vec![]),
        (vec![1.], vec![0]),
    ] {
        assert_eq!(Tensor::from_vec(data, shape), Err(Error::ShapeMismatch));
    }
}

#[test]
fn scalar_has_rank_zero_and_one_element() {
    let value = Tensor::scalar(7.);
    assert_tensor(&value, &[], &[7.]);
    assert_eq!(value.get(&[]), Ok(7.));
    assert_eq!(value, tensor(&[7.], &[]));
    assert_eq!(value.get(&[0]), Err(Error::InvalidIndex));
}

#[test]
fn zeros_supports_matrices_scalars_and_empty_tensors() {
    assert_tensor(&Tensor::zeros(vec![2, 3]).unwrap(), &[2, 3], &[0.; 6]);
    assert_tensor(&Tensor::zeros(vec![]).unwrap(), &[], &[0.]);
    for shape in [vec![0], vec![2, 0, 3]] {
        assert_tensor(&Tensor::zeros(shape.clone()).unwrap(), &shape, &[]);
    }
}

#[test]
fn construction_rejects_shape_overflow_before_allocation() {
    assert_eq!(
        Tensor::from_vec(vec![], vec![usize::MAX, 2]),
        Err(Error::ShapeOverflow)
    );
    assert_eq!(
        Tensor::zeros(vec![usize::MAX, 2]),
        Err(Error::ShapeOverflow)
    );
}

#[test]
fn zero_dimension_makes_even_a_large_shape_empty() {
    for shape in [vec![0, usize::MAX, 2], vec![usize::MAX, 2, 0]] {
        let value = Tensor::zeros(shape.clone()).unwrap();
        assert_tensor(&value, &shape, &[]);
        assert_eq!(value.get(&[0, 0, 0]), Err(Error::InvalidIndex));
    }
}

#[test]
fn indexing_uses_row_major_coordinates_in_three_dimensions() {
    let value = tensor(
        &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
        &[2, 2, 3],
    );
    assert_eq!(value.get(&[0, 0, 0]), Ok(0.));
    assert_eq!(value.get(&[0, 1, 2]), Ok(5.));
    assert_eq!(value.get(&[1, 0, 1]), Ok(7.));
    assert_eq!(value.get(&[1, 1, 2]), Ok(11.));
}

#[test]
fn indexing_rejects_wrong_rank_and_out_of_bounds_coordinates() {
    let value = Tensor::zeros(vec![2, 3]).unwrap();
    for index in [
        vec![],
        vec![0],
        vec![0, 0, 0],
        vec![2, 0],
        vec![0, 3],
        vec![usize::MAX, 0],
    ] {
        assert_eq!(value.get(&index), Err(Error::InvalidIndex));
    }
}

#[test]
fn reshape_preserves_data_and_reuses_the_allocation() {
    let value = tensor(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let pointer = value.data().as_ptr();
    let reshaped = value.reshape(vec![3, 2]).unwrap();
    assert_tensor(&reshaped, &[3, 2], &[1., 2., 3., 4., 5., 6.]);
    assert_eq!(reshaped.data().as_ptr(), pointer);
    assert_eq!(reshaped.get(&[2, 1]), Ok(6.));
}

#[test]
fn reshape_handles_scalars_and_empty_shapes() {
    let scalar = Tensor::scalar(4.).reshape(vec![1, 1]).unwrap();
    assert_tensor(&scalar, &[1, 1], &[4.]);
    assert_tensor(&scalar.reshape(vec![]).unwrap(), &[], &[4.]);
    let empty = Tensor::zeros(vec![0, 3]).unwrap();
    assert_tensor(&empty.reshape(vec![2, 0, 5]).unwrap(), &[2, 0, 5], &[]);
}

#[test]
fn reshape_rejects_mismatched_and_overflowing_shapes() {
    assert_eq!(
        Tensor::scalar(1.).reshape(vec![2]),
        Err(Error::ShapeMismatch)
    );
    assert_eq!(
        Tensor::scalar(1.).reshape(vec![usize::MAX, 2]),
        Err(Error::ShapeOverflow)
    );
    assert_eq!(
        Tensor::zeros(vec![0]).unwrap().reshape(vec![]),
        Err(Error::ShapeMismatch)
    );
}

#[test]
fn map_preserves_shape_and_leaves_input_unchanged() {
    let value = tensor(&[-2., 0., 3., 4.], &[2, 2]);
    assert_tensor(&value.map(|x| x * x + 1.), &[2, 2], &[5., 1., 10., 17.]);
    assert_tensor(&value, &[2, 2], &[-2., 0., 3., 4.]);
    assert_tensor(&Tensor::scalar(3.).map(|x| -x), &[], &[-3.]);
    let empty = Tensor::zeros(vec![2, 0]).unwrap();
    assert_tensor(
        &empty.map(|_| panic!("empty map called its closure")),
        &[2, 0],
        &[],
    );
}

#[test]
fn broadcasting_resolves_trailing_singleton_and_missing_dimensions() {
    let cases = [
        (vec![], vec![], vec![]),
        (vec![], vec![2, 3], vec![2, 3]),
        (vec![2, 3], vec![2, 3], vec![2, 3]),
        (vec![3], vec![2, 3], vec![2, 3]),
        (vec![2, 1], vec![1, 3], vec![2, 3]),
        (vec![2, 1, 3], vec![4, 1], vec![2, 4, 3]),
        (vec![0, 3], vec![1, 3], vec![0, 3]),
        (vec![0], vec![0], vec![0]),
    ];
    for (left, right, expected) in cases {
        assert_eq!(broadcast_shape(&left, &right), Ok(expected.clone()));
        assert_eq!(broadcast_shape(&right, &left), Ok(expected));
    }
}

#[test]
fn broadcasting_rejects_incompatible_dimensions_even_for_empty_outputs() {
    for (left, right) in [
        (vec![2], vec![3]),
        (vec![2, 3], vec![2]),
        (vec![0], vec![2]),
        (vec![0, 2], vec![0, 3]),
    ] {
        assert_eq!(broadcast_shape(&left, &right), Err(Error::ShapeMismatch));
        assert_eq!(broadcast_shape(&right, &left), Err(Error::ShapeMismatch));
    }
}

#[test]
fn broadcasting_checks_the_output_element_count_for_overflow() {
    assert_eq!(
        broadcast_shape(&[usize::MAX, 1], &[1, 2]),
        Err(Error::ShapeOverflow)
    );
    assert_eq!(
        broadcast_shape(&[usize::MAX, 2, 0], &[]),
        Ok(vec![usize::MAX, 2, 0])
    );
}

#[test]
fn addition_and_multiplication_work_elementwise_without_mutating_inputs() {
    let left = tensor(&[1., -2., 3., 4.], &[2, 2]);
    let right = tensor(&[5., 6., -7., 8.], &[2, 2]);
    assert_tensor(&left.add(&right).unwrap(), &[2, 2], &[6., 4., -4., 12.]);
    assert_tensor(&left.mul(&right).unwrap(), &[2, 2], &[5., -12., -21., 32.]);
    assert_tensor(&left, &[2, 2], &[1., -2., 3., 4.]);
    assert_tensor(&right, &[2, 2], &[5., 6., -7., 8.]);
}

#[test]
fn scalar_arithmetic_broadcasts_on_either_side() {
    let scalar = Tensor::scalar(2.);
    let matrix = tensor(&[1., 2., 3., 4.], &[2, 2]);
    for value in [scalar.add(&matrix).unwrap(), matrix.add(&scalar).unwrap()] {
        assert_tensor(&value, &[2, 2], &[3., 4., 5., 6.]);
    }
    for value in [scalar.mul(&matrix).unwrap(), matrix.mul(&scalar).unwrap()] {
        assert_tensor(&value, &[2, 2], &[2., 4., 6., 8.]);
    }
    assert_tensor(&scalar.add(&Tensor::scalar(3.)).unwrap(), &[], &[5.]);
    assert_tensor(&scalar.mul(&Tensor::scalar(3.)).unwrap(), &[], &[6.]);
}

#[test]
fn vector_broadcasting_repeats_values_across_matrix_rows() {
    let matrix = tensor(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let vector = tensor(&[10., 20., 30.], &[3]);
    assert_tensor(
        &matrix.add(&vector).unwrap(),
        &[2, 3],
        &[11., 22., 33., 14., 25., 36.],
    );
    assert_tensor(
        &vector.mul(&matrix).unwrap(),
        &[2, 3],
        &[10., 40., 90., 40., 100., 180.],
    );
}

#[test]
fn singleton_axes_broadcast_both_operands() {
    let column = tensor(&[1., 2.], &[2, 1]);
    let row = tensor(&[10., 20., 30.], &[1, 3]);
    assert_tensor(
        &column.add(&row).unwrap(),
        &[2, 3],
        &[11., 21., 31., 12., 22., 32.],
    );
    assert_tensor(
        &column.mul(&row).unwrap(),
        &[2, 3],
        &[10., 20., 30., 20., 40., 60.],
    );
}

#[test]
fn broadcasting_maps_coordinates_across_three_axes() {
    let left = tensor(&[1., 2., 3., 4.], &[2, 1, 2]);
    let right = tensor(&[10., 20., 30.], &[3, 1]);
    assert_tensor(
        &left.add(&right).unwrap(),
        &[2, 3, 2],
        &[11., 12., 21., 22., 31., 32., 13., 14., 23., 24., 33., 34.],
    );
    assert_tensor(
        &right.mul(&left).unwrap(),
        &[2, 3, 2],
        &[10., 20., 20., 40., 30., 60., 30., 40., 60., 80., 90., 120.],
    );
}

#[test]
fn binary_operations_reject_incompatible_shapes() {
    let left = Tensor::zeros(vec![2, 3]).unwrap();
    let right = Tensor::zeros(vec![2]).unwrap();
    assert_eq!(left.add(&right), Err(Error::ShapeMismatch));
    assert_eq!(left.mul(&right), Err(Error::ShapeMismatch));
}

#[test]
fn binary_operations_handle_empty_outputs_without_reading_storage() {
    let empty = Tensor::zeros(vec![2, 0, 3]).unwrap();
    let row = tensor(&[1., 2., 3.], &[3]);
    assert_tensor(&empty.add(&row).unwrap(), &[2, 0, 3], &[]);
    assert_tensor(&row.mul(&empty).unwrap(), &[2, 0, 3], &[]);
    let huge_empty = Tensor::zeros(vec![usize::MAX, 2, 0]).unwrap();
    assert_tensor(
        &huge_empty.add(&Tensor::scalar(1.)).unwrap(),
        &[usize::MAX, 2, 0],
        &[],
    );
}

#[test]
fn sum_handles_values_scalars_and_empty_tensors() {
    assert_eq!(tensor(&[1., -2., 3., 4.], &[2, 2]).sum(), 6.);
    assert_eq!(Tensor::scalar(7.).sum(), 7.);
    assert_eq!(Tensor::zeros(vec![0, 3]).unwrap().sum(), 0.);
}

#[test]
fn matmul_computes_a_rectangular_product() {
    let left = tensor(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let right = tensor(&[7., 8., 9., 10., 11., 12.], &[3, 2]);
    assert_tensor(
        &left.matmul(&right).unwrap(),
        &[2, 2],
        &[58., 64., 139., 154.],
    );
    assert_tensor(&left, &[2, 3], &[1., 2., 3., 4., 5., 6.]);
    assert_tensor(&right, &[3, 2], &[7., 8., 9., 10., 11., 12.]);
}

#[test]
fn matmul_handles_identity_and_single_element_matrices() {
    let value = tensor(&[1., -2., 3., 4.], &[2, 2]);
    let identity = tensor(&[1., 0., 0., 1.], &[2, 2]);
    assert_eq!(value.matmul(&identity).unwrap(), value);
    assert_eq!(identity.matmul(&value).unwrap(), value);
    assert_tensor(
        &tensor(&[3.], &[1, 1])
            .matmul(&tensor(&[4.], &[1, 1]))
            .unwrap(),
        &[1, 1],
        &[12.],
    );
}

#[test]
fn matmul_rejects_wrong_ranks_and_inner_dimensions() {
    for (left_shape, right_shape) in [
        (vec![], vec![1, 1]),
        (vec![1, 1], vec![]),
        (vec![2], vec![2, 1]),
        (vec![1, 2], vec![2]),
        (vec![1, 2, 3], vec![3, 2]),
        (vec![2, 3], vec![1, 3, 2]),
        (vec![2, 3], vec![2, 4]),
        (vec![0, 3], vec![2, 0]),
    ] {
        let left = Tensor::zeros(left_shape).unwrap();
        let right = Tensor::zeros(right_shape).unwrap();
        assert_eq!(left.matmul(&right), Err(Error::InvalidMatmul));
    }
}

#[test]
fn matmul_handles_zero_rows_columns_and_contracted_dimensions() {
    for (left_shape, right_shape, expected_shape, expected_data) in [
        (vec![0, 3], vec![3, 2], vec![0, 2], vec![]),
        (vec![2, 3], vec![3, 0], vec![2, 0], vec![]),
        (vec![2, 0], vec![0, 3], vec![2, 3], vec![0.; 6]),
        (vec![0, 0], vec![0, 0], vec![0, 0], vec![]),
    ] {
        let left = Tensor::zeros(left_shape).unwrap();
        let right = Tensor::zeros(right_shape).unwrap();
        assert_tensor(
            &left.matmul(&right).unwrap(),
            &expected_shape,
            &expected_data,
        );
    }
}

#[test]
fn matmul_rejects_an_overflowing_output_shape() {
    let left = Tensor::zeros(vec![usize::MAX, 0]).unwrap();
    let right = Tensor::zeros(vec![0, 2]).unwrap();
    assert_eq!(left.matmul(&right), Err(Error::ShapeOverflow));
}
