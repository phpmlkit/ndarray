<?php

require 'vendor/autoload.php';

use PhpMlKit\NDArray\NDArray;

$matrix = NDArray::array([[1, 2], [3, 4]]);
$flat = $matrix->flat();

echo $flat[0];   // 1
echo $flat[3];   // 4
echo $flat[-1];  // 4 (last element)
