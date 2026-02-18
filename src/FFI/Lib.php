<?php

declare(strict_types=1);

namespace NDArray\FFI;

use FFI;
use FFI\CData;
use NDArray\Exceptions\NDArrayException;
use NDArray\Exceptions\ShapeException;
use NDArray\Exceptions\DTypeException;
use NDArray\Exceptions\AllocationException;
use NDArray\Exceptions\PanicException;
use NDArray\Exceptions\IndexException;
use NDArray\Exceptions\MathException;

/**
 * Singleton wrapper for the FFI library.
 */
final class Lib
{
    /** Maximum number of dimensions for output shape buffer (caller-allocated). */
    public const MAX_NDIM = 32;

    /** @var (FFI&Bindings)|null */
    private static ?FFI $ffi = null;
    
    private static ?string $libraryPath = null;

    /**
     * Prevent instantiation.
     */
    private function __construct()
    {
    }

    /**
     * Initialize FFI with the Rust library.
     */
    public static function init(?string $libraryPath = null): void
    {
        if (self::$ffi !== null) {
            return;
        }

        self::$libraryPath = $libraryPath ?? self::findLibrary();

        $headerPath = dirname(__DIR__, 2) . '/lib/ndarray_php.h';

        if (!file_exists($headerPath)) {
            throw new NDArrayException(
                "Header file not found: $headerPath. Did you build the Rust library?"
            );
        }

        if (!file_exists(self::$libraryPath)) {
            throw new NDArrayException(
                "Library not found: " . self::$libraryPath
            );
        }

        /** @var FFI&Bindings $ffi */
        $ffi = FFI::cdef(
            file_get_contents($headerPath),
            self::$libraryPath
        );
        self::$ffi = $ffi;
    }

    /**
     * Get FFI instance.
     *
     * @return FFI&Bindings
     */
    public static function get(): FFI
    {
        if (self::$ffi === null) {
            self::init();
        }

        return self::$ffi;
    }

    /**
     * Reset FFI instance (mainly for testing).
     */
    public static function reset(): void
    {
        self::$ffi = null;
        self::$libraryPath = null;
    }

    /**
     * Find the library file based on platform.
     */
    private static function findLibrary(): string
    {
        $baseDir = dirname(__DIR__, 2) . '/lib/';

        return match (PHP_OS_FAMILY) {
            'Windows' => $baseDir . 'ndarray_php.dll',
            'Darwin' => $baseDir . 'libndarray_php.dylib',
            default => $baseDir . 'libndarray_php.so',
        };
    }

    /**
     * Get the last error message from Rust.
     *
     * @return string
     */
    public static function getLastError(): string
    {
        $ffi = self::get();
        $buffer = $ffi->new("char[1024]");
        $len = $ffi->ndarray_get_last_error($buffer, 1024);
        
        if ($len === 0) {
            return "Unknown error";
        }
        
        return FFI::string($buffer, $len);
    }

    /**
     * Check the status code returned by an FFI function.
     *
     * @param int $code
     * @throws NDArrayException
     */
    public static function checkStatus(int $code): void
    {
        if ($code === 0) {
            return;
        }

        $message = self::getLastError();

        match ($code) {
            1 => throw new NDArrayException($message), // Generic
            2 => throw new ShapeException($message), // Shape
            3 => throw new DTypeException($message), // DType
            4 => throw new AllocationException($message), // Alloc
            5 => throw new PanicException($message), // Panic
            6 => throw new IndexException($message), // Index
            7 => throw new MathException($message), // Math
            default => throw new NDArrayException($message),
        };
    }

    /**
     * Create a C array from a PHP array with the given type.
     *
     * @param string $type C type (e.g., 'double', 'int64_t', 'size_t')
     * @param array $data PHP array of values
     * @return CData The allocated C array
     */
    public static function createCArray(string $type, array $data): CData
    {
        $ffi = self::get();
        $count = count($data);

        if ($count === 0) {
            // Return a valid but empty array
            return $ffi->new("{$type}[1]");
        }

        $cArray = $ffi->new("{$type}[{$count}]");

        foreach ($data as $i => $value) {
            $cArray[$i] = $value;
        }

        return $cArray;
    }

    /**
     * Create a C array for shape data (always size_t).
     *
     * @param array<int> $shape Shape dimensions
     * @return CData
     */
    public static function createShapeArray(array $shape): CData
    {
        return self::createCArray('size_t', $shape);
    }

    /**
     * Extract shape array from FFI output pointer or caller-allocated buffer.
     *
     * @param CData $shapePtr Pointer to size_t array from FFI (or buffer)
     * @param int $ndim Number of dimensions
     * @return array<int> Shape dimensions
     */
    public static function extractShapeFromPointer(CData $shapePtr, int $ndim): array
    {
        $shape = [];
        for ($i = 0; $i < $ndim; $i++) {
            $shape[] = (int) $shapePtr[$i];
        }
        return $shape;
    }

    /**
     * Allocate caller-provided output metadata buffers for Rust FFI (dtype, ndim, shape).
     *
     * @return array{0: CData, 1: CData, 2: CData} [out_dtype (uint8_t), out_ndim (size_t), out_shape (size_t[MAX_NDIM])]
     */
    public static function createOutputMetadataBuffers(): array
    {
        $ffi = self::get();
        return [
            $ffi->new('uint8_t'),
            $ffi->new('size_t'),
            $ffi->new('size_t[' . self::MAX_NDIM . ']'),
        ];
    }

    /**
     * Create a C box of the given type.
     *
     * @param string $type C type (e.g., 'char', 'size_t', 'struct NdArrayHandle*')
     * @return CData&Box
     */
    public static function createBox(string $type): CData
    {
        return self::get()->new($type);
    }

    /**
     * Cast a C pointer to the specified type.
     *
     * @param string $type Target type
     * @param CData $ptr Source pointer
     * @return CData
     */
    public static function cast(string $type, CData $ptr): CData
    {
        return self::get()->cast($type, $ptr);
    }

    /**
     * Get the address of a CData value.
     *
     * @param CData $value
     * @return CData
     */
    public static function addr(CData $value): CData
    {
        return FFI::addr($value);
    }
}
