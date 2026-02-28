<?php

declare(strict_types=1);

require_once 'vendor/autoload.php';

use PhpMlKit\NDArray\NDArray;

$matrix = NDArray::array([[1, 2, 3], [4, 5, 6]]);
$col = $matrix->slice([':', '0']);  // Shape [2, 1]

echo $col.\PHP_EOL;

$a = NDArray::array([[1, 2, 3], [4, 5, 6]]);
// Note: slice(['1', '2']) returns shape [1, 1] instead of [1]
$single = $a->slice(['1', '2']);

echo $single.\PHP_EOL;

// python3 -c "
// import numpy as np
//
// arr = np.array([[[1, 2], [3, 4]],[[5, 6], [7, 8]]])
// print('Original shape:', arr.shape)

// view = arr[:, :, 0]
// result = np.product(view, axis=0)
// print('result shape:', result.shape)
// print('result values:', result)
// "
