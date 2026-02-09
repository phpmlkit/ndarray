<?php

$libPath = __DIR__ . '/lib/libndarray_php.dylib';
$headerPath = __DIR__ . '/lib/ndarray_php.h';
$header = file_get_contents($headerPath);

$ffi = FFI::cdef($header, $libPath);

echo "Calling hello_from_rust()...\n";
$result = $ffi->hello_from_rust();
echo "Result: $result\n\n";

echo "Calling add_numbers(10, 32)...\n";
$sum = $ffi->add_numbers(10, 32);
echo "Result: $sum\n\n";

if ($result === 42 && $sum === 42) {
    echo "✅ FFI is working perfectly!\n";
} else {
    echo "❌ Something went wrong\n";
}
