<?php

declare(strict_types=1);

namespace NDArray;

use FFI;
use FFI\CData;
use NDArray\Exceptions\ShapeException;
use NDArray\FFI\FFIInterface;

/**
 * N-dimensional array class backed by Rust.
 */
class NDArray
{
    /** Opaque pointer to Rust NDArrayWrapper */
    private CData $handle;

    /** Shape of the array */
    private array $shape;

    /** Data type */
    private DType $dtype;

    /** Number of dimensions */
    private int $ndim;

    /** Total number of elements */
    private int $size;

    /**
     * Private constructor - use factory methods.
     */
    private function __construct(CData $handle, array $shape, DType $dtype)
    {
        $this->handle = $handle;
        $this->shape = $shape;
        $this->dtype = $dtype;
        $this->ndim = count($shape);
        $this->size = (int) array_product($shape);
    }

    /**
     * Destructor - cleanup Rust memory.
     */
    public function __destruct()
    {
        if (isset($this->handle)) {
            FFIInterface::get()->ndarray_free($this->handle);
        }
    }

    // =========================================================================
    // Factory Methods
    // =========================================================================

    /**
     * Create array from PHP array.
     *
     * @param array $data Nested PHP array
     * @param DType|null $dtype Data type (auto-inferred if null)
     * @return self
     */
    public static function array(array $data, ?DType $dtype = null): self
    {
        $shape = self::inferShape($data);
        $flatData = self::flattenArray($data);

        if (empty($flatData)) {
            throw new ShapeException('Cannot create array from empty data');
        }

        $dtype ??= DType::fromArray($data);

        $ffi = FFIInterface::get();
        $len = count($flatData);

        $handle = self::createTyped($ffi, $dtype, $flatData, $shape, $len);

        if ($handle === null) {
            throw new ShapeException('Failed to create array: shape/data mismatch');
        }

        return new self($handle, $shape, $dtype);
    }

    // =========================================================================
    // Properties
    // =========================================================================

    /**
     * Get shape.
     *
     * @return array<int>
     */
    public function shape(): array
    {
        return $this->shape;
    }

    /**
     * Get number of dimensions.
     */
    public function ndim(): int
    {
        return $this->ndim;
    }

    /**
     * Get total number of elements.
     */
    public function size(): int
    {
        return $this->size;
    }

    /**
     * Get data type.
     */
    public function dtype(): DType
    {
        return $this->dtype;
    }

    /**
     * Get item size in bytes.
     */
    public function itemsize(): int
    {
        return $this->dtype->itemSize();
    }

    /**
     * Get total bytes consumed.
     */
    public function nbytes(): int
    {
        return $this->size * $this->itemsize();
    }

    /**
     * Get the internal handle (for advanced FFI usage).
     *
     * @internal
     */
    public function getHandle(): CData
    {
        return $this->handle;
    }

    // =========================================================================
    // Conversion Methods
    // =========================================================================

    /**
     * Convert to PHP array.
     *
     * @return array
     */
    public function toArray(): array
    {
        $ffi = FFIInterface::get();

        $outPtr = $ffi->new("char*");
        $outLen = $ffi->new("uintptr_t");

        $success = $ffi->ndarray_to_json(
            $this->handle,
            FFIInterface::addr($outPtr),
            FFIInterface::addr($outLen),
            17 // default max precision
        );

        if (!$success) {
            throw new \RuntimeException("Failed to serialize array to JSON");
        }

        $json = FFI::string($outPtr, $outLen->cdata);

        $ffi->ndarray_free_string($outPtr);

        return json_decode($json, true, 512, JSON_THROW_ON_ERROR);
    }

    // =========================================================================
    // Private Helper Methods
    // =========================================================================

    /**
     * Infer shape from nested PHP array.
     *
     * @return array<int>
     */
    private static function inferShape(array $data): array
    {
        $shape = [];
        $current = $data;

        while (is_array($current)) {
            $count = count($current);
            if ($count === 0) {
                break;
            }
            $shape[] = $count;
            $current = reset($current);
        }

        return $shape;
    }

    /**
     * Flatten nested PHP array.
     *
     * @return array<mixed>
     */
    private static function flattenArray(array $data): array
    {
        $result = [];

        array_walk_recursive($data, function ($value) use (&$result) {
            $result[] = $value;
        });

        return $result;
    }

    // =========================================================================
    // Type-specific creation
    // =========================================================================

    /**
     * Create an NDArray handle using the appropriate FFI function for the dtype.
     *
     * @param \FFI $ffi FFI instance
     * @param DType $dtype Data type
     * @param array $data Flat array of values
     * @param array<int> $shape Array shape
     * @param int $len Number of elements
     * @return CData|null Opaque handle or null on failure
     */
    private static function createTyped(\FFI $ffi, DType $dtype, array $data, array $shape, int $len): ?CData
    {
        // Convert bools to u8 (0 or 1) for FFI
        if ($dtype === DType::Bool) {
            $data = array_map(fn($v) => $v ? 1 : 0, $data);
        }

        $cData = FFIInterface::createCArray($dtype->ffiType(), $data);
        $cShape = FFIInterface::createShapeArray($shape);

        // Dynamic dispatch to the appropriate FFI function
        $funcName = 'ndarray_create_' . $dtype->name();

        return $ffi->$funcName($cData, $len, $cShape, count($shape));
    }
}
