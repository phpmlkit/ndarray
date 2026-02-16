<?php

require_once __DIR__ . '/vendor/autoload.php';

use NDArray\DType;
use NDArray\NDArray;

$starttime = microtime(true);

$data = range(1, 1000000);
$a = NDArray::array($data, dtype: DType::Float32);
$b = $a->reshape([1000, 1000])
    ->transpose()
    ->add($a->reshape([1000, 1000]))
    ->log()
    ->mean(0)
    ->astype(DType::Int32)
    ->toArray();

dump($b);

$endtime = microtime(true);
$elapsed_ms = ($endtime - $starttime) * 1000;
echo 'Time taken: ' . $elapsed_ms . ' ms' . PHP_EOL;
