<?php

declare(strict_types=1);

require_once 'vendor/autoload.php';

use PhpMlKit\NDArray\NDArray;

// Create a simple array
$vector = NDArray::array([1, 2, 3, 4, 5]);
echo "Vector: {$vector}\n";
echo "Sum: {$vector->sum()}\n\n";

// Create a 2D array
$matrix = NDArray::zeros([3, 3]);
echo "Zero matrix: {$matrix}\n\n";

echo "✓ NDArray is working correctly!\n";
