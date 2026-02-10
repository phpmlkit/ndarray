<?php

declare(strict_types=1);

namespace NDArray;

use FFI;
use FFI\CData;
use NDArray\Exceptions\ShapeException;
use NDArray\FFI\Box;
use NDArray\FFI\Handle;
use NDArray\FFI\Lib;

/**
 * N-dimensional array class.
 */
class NDArray
{
    /** Number of dimensions */
    private int $ndim;

    /** Total number of elements */
    private int $size;

    /**
     * Private constructor - use factory methods.
     *
     * @param Handle $handle Opaque pointer to Rust NDArrayWrapper
     * @param array $shape Shape of the array
     * @param DType $dtype Data type
     */
    private function __construct(private CData $handle, private array $shape, private DType $dtype)
    {
        $this->ndim = count($shape);
        $this->size = (int) array_product($shape);
    }

    /**
     * Destructor - cleanup Rust memory.
     */
    public function __destruct()
    {
        if (isset($this->handle)) {
            Lib::get()->ndarray_free($this->handle);
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

        $ffi = Lib::get();
        $len = count($flatData);

        $handle = self::createTyped($ffi, $dtype, $flatData, $shape, $len);

        return new self($handle, $shape, $dtype);
    }

    /**
     * Create an array filled with zeros.
     *
     * @param array<int> $shape Array shape
     * @param DType|null $dtype Data type (default: Float64)
     * @return self
     */
    public static function zeros(array $shape, ?DType $dtype = null): self
    {
        $dtype ??= DType::Float64;

        $ffi = Lib::get();
        $cShape = Lib::createShapeArray($shape);
        $outHandle = $ffi->new("struct NdArrayHandle*");

        $status = $ffi->ndarray_zeros(
            $cShape,
            count($shape),
            $dtype->value,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new self($outHandle, $shape, $dtype);
    }

    /**
     * Create an array filled with ones.
     *
     * @param array<int> $shape Array shape
     * @param DType|null $dtype Data type (default: Float64)
     * @return self
     */
    public static function ones(array $shape, ?DType $dtype = null): self
    {
        $dtype ??= DType::Float64;

        $ffi = Lib::get();
        $cShape = Lib::createShapeArray($shape);
        $outHandle = $ffi->new("struct NdArrayHandle*");

        $status = $ffi->ndarray_ones(
            $cShape,
            count($shape),
            $dtype->value,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new self($outHandle, $shape, $dtype);
    }

    /**
     * Create an array filled with a specific value.
     *
     * @param array<int> $shape Array shape
     * @param mixed $fillValue Value to fill array with
     * @param DType|null $dtype Data type (default: inferred from value)
     * @return self
     */
    public static function full(array $shape, mixed $fillValue, ?DType $dtype = null): self
    {
        if ($dtype === null) {
            // Simple inference based on value type
            if (is_int($fillValue)) {
                $dtype = DType::Int64;
            } elseif (is_float($fillValue)) {
                $dtype = DType::Float64;
            } elseif (is_bool($fillValue)) {
                $dtype = DType::Bool;
            } else {
                throw new \InvalidArgumentException("Cannot infer dtype from fill value");
            }
        }

        $ffi = Lib::get();
        $cShape = Lib::createShapeArray($shape);

        // Create scalar value for FFI
        if ($dtype === DType::Bool) {
            $val = $fillValue ? 1 : 0;
        } else {
            $val = $fillValue;
        }
        $cValue = Lib::createCArray($dtype->ffiType(), [$val]);

        $outHandle = $ffi->new("struct NdArrayHandle*");

        $status = $ffi->ndarray_full(
            $cShape,
            count($shape),
            $cValue, // Pointer to value
            $dtype->value,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new self($outHandle, $shape, $dtype);
    }

    /**
     * Create a 2D identity matrix.
     *
     * @param int $N Number of rows
     * @param int|null $M Number of columns (default: N)
     * @param int $k Diagonal index (0: main, >0: upper, <0: lower)
     * @param DType|null $dtype Data type (default: Float64)
     * @return self
     */
    public static function eye(int $N, ?int $M = null, int $k = 0, ?DType $dtype = null): self
    {
        $M ??= $N;
        $dtype ??= DType::Float64;

        $ffi = Lib::get();
        $outHandle = $ffi->new("struct NdArrayHandle*");

        $status = $ffi->ndarray_eye(
            $N,
            $M,
            $k,
            $dtype->value,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new self($outHandle, [$N, $M], $dtype);
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
        $ffi = Lib::get();

        $outPtr = $ffi->new("char*");
        /** @var Box */
        $outLen = $ffi->new("size_t");

        $status = $ffi->ndarray_to_json(
            $this->handle,
            Lib::addr($outPtr),
            Lib::addr($outLen),
            17 // default max precision
        );

        Lib::checkStatus($status);

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
     * @param FFI $ffi FFI instance
     * @param DType $dtype Data type
     * @param array $data Flat array of values
     * @param array<int> $shape Array shape
     * @param int $len Number of elements
     * @return CData Opaque handle
     */
    private static function createTyped(FFI $ffi, DType $dtype, array $data, array $shape, int $len): CData
    {
        if ($dtype === DType::Bool) {
            $data = array_map(fn($v) => $v ? 1 : 0, $data);
        }

        $cData = Lib::createCArray($dtype->ffiType(), $data);
        $cShape = Lib::createShapeArray($shape);

        $outHandle = $ffi->new("struct NdArrayHandle*");

        $funcName = "ndarray_create_{$dtype->name()}";

        $status = $ffi->$funcName(
            $cData,
            $len,
            $cShape,
            count($shape),
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return $outHandle;
    }
}
