<?php

declare(strict_types=1);

namespace NDArray\Traits;

use FFI;
use FFI\CData;
use NDArray\DType;
use NDArray\Exceptions\ShapeException;
use NDArray\FFI\Bindings;
use NDArray\FFI\Lib;

/**
 * Static factory methods for creating NDArray instances.
 *
 * Provides array(), empty(), zeros(), ones(), full(), and eye().
 */
trait CreatesArrays
{
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
     * Create an empty array for the given shape and dtype.
     *
     * This is intended for preallocation APIs where shape metadata matters,
     * including zero-size shapes such as [3, 0, 1]. The current implementation
     * uses the same safe allocation path as zeros().
     *
     * @param array<int> $shape Array shape
     * @param DType|null $dtype Data type (default: Float64)
     * @return self
     */
    public static function empty(array $shape, ?DType $dtype = null): self
    {
        if (!in_array(0, $shape, true)) {
            throw new ShapeException("empty() requires a zero-size shape (at least one dimension must be 0)");
        }

        return self::zeros($shape, $dtype);
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
            $cValue,
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

    /**
     * Create evenly spaced values within a given interval.
     *
     * @param int|float $start Start of interval (inclusive)
     * @param int|float|null $stop End of interval (exclusive)
     * @param int|float $step Spacing between values
     * @param DType|null $dtype Data type
     * @return self
     */
    public static function arange(int|float $start, int|float|null $stop = null, int|float $step = 1, ?DType $dtype = null): self
    {
        if ($stop === null) {
            $stop = $start;
            $start = 0;
        }

        $dtype ??= DType::fromValue($start);

        if ($step == 0) {
            throw new \InvalidArgumentException("Step cannot be zero");
        }

        if ($dtype === DType::Bool) {
            throw new \InvalidArgumentException("Bool dtype not supported for arange");
        }

        $ffi = Lib::get();
        $outHandle = $ffi->new("struct NdArrayHandle*");

        $status = $ffi->ndarray_arange(
            (float) $start,
            (float) $stop,
            (float) $step,
            $dtype->value,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        // Calculate the shape
        $num = 0;
        if ($step > 0 && $start < $stop) {
            $num = (int) ceil(($stop - $start) / $step);
        } elseif ($step < 0 && $start > $stop) {
            $num = (int) ceil(($start - $stop) / -$step);
        }

        return new self($outHandle, [$num], $dtype);
    }

    /**
     * Create evenly spaced numbers over a specified interval.
     *
     * @param float $start The starting value of the sequence
     * @param float $stop The end value of the sequence
     * @param int $num Number of samples to generate
     * @param bool $endpoint If true, stop is the last sample
     * @param DType $dtype Data type (default: Float64)
     * @return self
     */
    public static function linspace(
        float $start,
        float $stop,
        int $num = 50,
        bool $endpoint = true,
        DType $dtype = DType::Float64
    ): self {
        if ($num <= 0) {
            throw new ShapeException("Number of samples, $num, must be positive");
        }

        if ($dtype !== DType::Float32 && $dtype !== DType::Float64) {
            throw new \InvalidArgumentException("linspace only supports Float32 and Float64 dtypes");
        }

        $ffi = Lib::get();
        $outHandle = $ffi->new("struct NdArrayHandle*");

        $status = $ffi->ndarray_linspace(
            $start,
            $stop,
            $num,
            $endpoint,
            $dtype->value,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new self($outHandle, [$num], $dtype);
    }

    /**
     * Create numbers spaced evenly on a log scale.
     *
     * @param float $start The starting exponent (base**start is the first value)
     * @param float $stop The end exponent (base**stop is the final value)
     * @param int $num Number of samples to generate
     * @param float $base The base of the log space
     * @param DType $dtype Data type (default: Float64)
     * @return self
     */
    public static function logspace(
        float $start,
        float $stop,
        int $num = 50,
        float $base = 10.0,
        DType $dtype = DType::Float64
    ): self {
        if ($num <= 0) {
            throw new ShapeException("Number of samples, $num, must be positive");
        }

        if ($dtype !== DType::Float32 && $dtype !== DType::Float64) {
            throw new \InvalidArgumentException("logspace only supports Float32 and Float64 dtypes");
        }

        $ffi = Lib::get();
        $outHandle = $ffi->new("struct NdArrayHandle*");

        $status = $ffi->ndarray_logspace(
            $start,
            $stop,
            $num,
            $base,
            $dtype->value,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new self($outHandle, [$num], $dtype);
    }

    /**
     * Create numbers spaced geometrically from start to stop.
     *
     * @param float $start The starting value of the sequence
     * @param float $stop The end value of the sequence
     * @param int $num Number of samples to generate
     * @param DType $dtype Data type (default: Float64)
     * @return self
     */
    public static function geomspace(
        float $start,
        float $stop,
        int $num = 50,
        DType $dtype = DType::Float64
    ): self {
        if ($num <= 0) {
            throw new ShapeException("Number of samples, $num, must be positive");
        }

        if ($dtype !== DType::Float32 && $dtype !== DType::Float64) {
            throw new \InvalidArgumentException("geomspace only supports Float32 and Float64 dtypes");
        }

        if ($start == 0.0 || $stop == 0.0) {
            throw new \InvalidArgumentException("geomspace does not support zero values");
        }

        if (($start > 0.0) !== ($stop > 0.0)) {
            throw new \InvalidArgumentException("geomspace requires start and stop to have the same sign");
        }

        $ffi = Lib::get();
        $outHandle = $ffi->new("struct NdArrayHandle*");

        $status = $ffi->ndarray_geomspace(
            $start,
            $stop,
            $num,
            $dtype->value,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new self($outHandle, [$num], $dtype);
    }

    /**
     * Create random samples from a uniform distribution over [0, 1).
     *
     * @param array<int> $shape Output shape
     * @param DType|null $dtype Float dtype (default: Float64)
     * @param int|null $seed Optional seed for deterministic output
     * @return self
     */
    public static function random(array $shape, ?DType $dtype = null, ?int $seed = null): self
    {
        $dtype ??= DType::Float64;
        self::assertFloatDtype($dtype, 'random');

        $ffi = Lib::get();
        $outHandle = $ffi->new("struct NdArrayHandle*");
        $status = $ffi->ndarray_random(
            Lib::createShapeArray($shape),
            count($shape),
            $dtype->value,
            $seed !== null,
            $seed ?? 0,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);
        return new self($outHandle, $shape, $dtype);
    }

    /**
     * Create random integer samples from [low, high).
     *
     * @param int $low Inclusive lower bound
     * @param int $high Exclusive upper bound
     * @param array<int> $shape Output shape
     * @param DType|null $dtype Integer dtype (default: Int64)
     * @param int|null $seed Optional seed for deterministic output
     * @return self
     */
    public static function randomInt(int $low, int $high, array $shape, ?DType $dtype = null, ?int $seed = null): self
    {
        $dtype ??= DType::Int64;
        if (!$dtype->isInteger()) {
            throw new \InvalidArgumentException('randomInt only supports integer dtypes');
        }
        if ($high <= $low) {
            throw new \InvalidArgumentException("randomInt requires high > low, got [$low, $high)");
        }

        $ffi = Lib::get();
        $outHandle = $ffi->new("struct NdArrayHandle*");
        $status = $ffi->ndarray_random_int(
            $low,
            $high,
            Lib::createShapeArray($shape),
            count($shape),
            $dtype->value,
            $seed !== null,
            $seed ?? 0,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);
        return new self($outHandle, $shape, $dtype);
    }

    /**
     * Create random samples from a standard normal distribution N(0, 1).
     *
     * @param array<int> $shape Output shape
     * @param DType|null $dtype Float dtype (default: Float64)
     * @param int|null $seed Optional seed for deterministic output
     * @return self
     */
    public static function randn(array $shape, ?DType $dtype = null, ?int $seed = null): self
    {
        $dtype ??= DType::Float64;
        self::assertFloatDtype($dtype, 'randn');

        $ffi = Lib::get();
        $outHandle = $ffi->new("struct NdArrayHandle*");
        $status = $ffi->ndarray_randn(
            Lib::createShapeArray($shape),
            count($shape),
            $dtype->value,
            $seed !== null,
            $seed ?? 0,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);
        return new self($outHandle, $shape, $dtype);
    }

    /**
     * Create random samples from a normal distribution N(mean, std).
     *
     * @param float $mean Mean of the distribution
     * @param float $std Standard deviation (must be > 0)
     * @param array<int> $shape Output shape
     * @param DType|null $dtype Float dtype (default: Float64)
     * @param int|null $seed Optional seed for deterministic output
     * @return self
     */
    public static function normal(float $mean, float $std, array $shape, ?DType $dtype = null, ?int $seed = null): self
    {
        $dtype ??= DType::Float64;
        self::assertFloatDtype($dtype, 'normal');
        if (!($std > 0.0)) {
            throw new \InvalidArgumentException("normal requires std > 0, got $std");
        }

        $ffi = Lib::get();
        $outHandle = $ffi->new("struct NdArrayHandle*");
        $status = $ffi->ndarray_normal(
            $mean,
            $std,
            Lib::createShapeArray($shape),
            count($shape),
            $dtype->value,
            $seed !== null,
            $seed ?? 0,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);
        return new self($outHandle, $shape, $dtype);
    }

    /**
     * Create random samples from a uniform distribution over [low, high).
     *
     * @param float $low Inclusive lower bound
     * @param float $high Exclusive upper bound
     * @param array<int> $shape Output shape
     * @param DType|null $dtype Float dtype (default: Float64)
     * @param int|null $seed Optional seed for deterministic output
     * @return self
     */
    public static function uniform(float $low, float $high, array $shape, ?DType $dtype = null, ?int $seed = null): self
    {
        $dtype ??= DType::Float64;
        self::assertFloatDtype($dtype, 'uniform');
        if ($high <= $low) {
            throw new \InvalidArgumentException("uniform requires high > low, got [$low, $high)");
        }

        $ffi = Lib::get();
        $outHandle = $ffi->new("struct NdArrayHandle*");
        $status = $ffi->ndarray_uniform(
            $low,
            $high,
            Lib::createShapeArray($shape),
            count($shape),
            $dtype->value,
            $seed !== null,
            $seed ?? 0,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);
        return new self($outHandle, $shape, $dtype);
    }

    // =========================================================================
    // Private Helpers
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

    /**
     * Create an NDArray handle using the appropriate FFI function for the dtype.
     *
     * @param FFI&Bindings $ffi FFI instance
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

        $status = $ffi->ndarray_create(
            $cData,
            $len,
            $cShape,
            count($shape),
            $dtype->value,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return $outHandle;
    }

    /**
     * Ensure method only accepts floating-point dtypes.
     */
    private static function assertFloatDtype(DType $dtype, string $method): void
    {
        if (!$dtype->isFloat()) {
            throw new \InvalidArgumentException("$method only supports Float32 and Float64 dtypes");
        }
    }
}
