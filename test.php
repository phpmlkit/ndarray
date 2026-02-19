<?php

require_once __DIR__ . '/vendor/autoload.php';

use NDArray\DType;
use NDArray\NDArray;

$starttime = microtime(true);

$a = NDArray::random([2000, 2000], DType::Float32);
$b = $a->transpose()
    ->add($a)
    ->log()
    // ->mean(0)
    ->softmax()
    ->astype(DType::Int32)
    ->toArray();

dump($b);

$endtime = microtime(true);
$elapsed_ms = ($endtime - $starttime) * 1000;
echo 'Time taken: ' . $elapsed_ms . ' ms' . PHP_EOL;
