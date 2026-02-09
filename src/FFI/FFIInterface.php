<?php

declare(strict_types=1);

namespace NDArray\FFI;

use FFI;
use FFI\CData;
use RuntimeException;

/**
 * Singleton manager for FFI access to the Rust library.
 */
final class FFIInterface
{
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
            throw new RuntimeException(
                "Header file not found: $headerPath. Did you build the Rust library?"
            );
        }

        if (!file_exists(self::$libraryPath)) {
            throw new RuntimeException(
                "Library not found: " . self::$libraryPath
            );
        }

        self::$ffi = FFI::cdef(
            file_get_contents($headerPath),
            self::$libraryPath
        );
    }

    /**
     * Get FFI instance.
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
