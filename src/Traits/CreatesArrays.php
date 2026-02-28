<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Traits;

use FFI\CData;
use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\Exceptions\DTypeException;
use PhpMlKit\NDArray\Exceptions\ShapeException;
use PhpMlKit\NDArray\FFI\Bindings;
use PhpMlKit\NDArray\FFI\Lib;

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
     * @param array<mixed> $data  Nested PHP array
     * @param null|DType   $dtype Data type (auto-inferred if null)
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
        $len = \count($flatData);

        $handle = self::createTyped($ffi, $dtype, $flatData, $shape, $len);

        return new self($handle, $shape, $dtype);
    }

    /**
     * Create an array filled with zeros.
     *
     * @param array<int> $shape Array shape
     * @param DType      $dtype Data type (default: Float64)
     */
    public static function zeros(array $shape, DType $dtype = DType::Float64): self
    {
        $ffi = Lib::get();
        $cShape = Lib::createShapeArray($shape);
        $outHandle = $ffi->new('struct NdArrayHandle*');

        $status = $ffi->ndarray_zeros(
            $cShape,
            \count($shape),
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
     * @param DType      $dtype Data type (default: Float64)
     */
    public static function ones(array $shape, DType $dtype = DType::Float64): self
    {
        $ffi = Lib::get();
        $cShape = Lib::createShapeArray($shape);
        $outHandle = $ffi->new('struct NdArrayHandle*');

        $status = $ffi->ndarray_ones(
            $cShape,
            \count($shape),
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
     * @param DType      $dtype Data type (default: Float64)
     */
    public static function empty(array $shape, DType $dtype = DType::Float64): self
    {
        if (!\in_array(0, $shape, true)) {
            throw new ShapeException('empty() requires a zero-size shape (at least one dimension must be 0)');
        }

        return self::zeros($shape, $dtype);
    }

    /**
     * Create an array filled with a specific value.
     *
     * @param array<int>     $shape Array shape
     * @param bool|float|int $value Value to fill array with
     * @param null|DType     $dtype Data type (default: inferred from value)
     */
    public static function full(array $shape, bool|float|int $value, ?DType $dtype = null): self
    {
        if (null === $dtype) {
            if (\is_int($value)) {
                $dtype = DType::Int64;
            } elseif (\is_float($value)) {
                $dtype = DType::Float64;
            } elseif (\is_bool($value)) {
                $dtype = DType::Bool;
            } else {
                throw new \InvalidArgumentException('Cannot infer dtype from fill value');
            }
        }

        $ffi = Lib::get();

        $cShape = Lib::createShapeArray($shape);
        $cValue = $dtype->createCValue($value);

        $outHandle = $ffi->new('struct NdArrayHandle*');

        $status = $ffi->ndarray_full(
            $cShape,
            \count($shape),
            Lib::addr($cValue),
            $dtype->value,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new self($outHandle, $shape, $dtype);
    }

    /**
     * Create an array from an external C buffer pointer.
     *
     * This method copies data from an external FFI buffer into a new NDArray. The source buffer
     * remains owned by the caller - this method creates an independent copy.
     *
     * @param CData      $buffer Pointer to raw C data (e.g., float*, int16_t*)
     * @param array<int> $shape  Array shape
     * @param DType      $dtype  Data type of the buffer
     *
     * @throws ShapeException If buffer size doesn't match shape
     */
    public static function fromBuffer(CData $buffer, array $shape, DType $dtype): self
    {
        $expectedSize = (int) array_product($shape);

        if ($expectedSize <= 0) {
            throw new ShapeException('Shape must have positive size');
        }

        $ffi = Lib::get();
        $cShape = Lib::createShapeArray($shape);
        $outHandle = $ffi->new('struct NdArrayHandle*');

        $status = $ffi->ndarray_create(
            $buffer,
            $expectedSize,
            $cShape,
            \count($shape),
            $dtype->value,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new self($outHandle, $shape, $dtype);
    }

    /**
     * Create an array of zeros with the same shape as the input array.
     *
     * @param self       $array Input array defining the output shape
     * @param null|DType $dtype Data type (default: same as input array)
     */
    public static function zerosLike(self $array, ?DType $dtype = null): self
    {
        return self::zeros($array->shape(), $dtype ?? $array->dtype());
    }

    /**
     * Create an array of ones with the same shape as the input array.
     *
     * @param self       $array Input array defining the output shape
     * @param null|DType $dtype Data type (default: same as input array)
     */
    public static function onesLike(self $array, ?DType $dtype = null): self
    {
        return self::ones($array->shape(), $dtype ?? $array->dtype());
    }

    /**
     * Create an array filled with a specific value, with the same shape as the input array.
     *
     * @param self           $array Input array defining the output shape
     * @param bool|float|int $value Value to fill array with
     * @param null|DType     $dtype Data type (default: inferred from value or same as input)
     */
    public static function fullLike(self $array, bool|float|int $value, ?DType $dtype = null): self
    {
        $dtype ??= $array->dtype();

        return self::full($array->shape(), $value, $dtype);
    }

    /**
     * Create a 2D identity matrix.
     *
     * @param int      $N     Number of rows
     * @param null|int $M     Number of columns (default: N)
     * @param int      $k     Diagonal index (0: main, >0: upper, <0: lower)
     * @param DType    $dtype Data type (default: Float64)
     */
    public static function eye(int $N, ?int $M = null, int $k = 0, DType $dtype = DType::Float64): self
    {
        $M ??= $N;
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

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
     * @param float|int      $start Start of interval (inclusive)
     * @param null|float|int $stop  End of interval (exclusive)
     * @param float|int      $step  Spacing between values
     * @param null|DType     $dtype Data type
     */
    public static function arange(float|int $start, float|int|null $stop = null, float|int $step = 1, ?DType $dtype = null): self
    {
        if (null === $stop) {
            $stop = $start;
            $start = 0;
        }

        $dtype ??= DType::fromValue($start);

        if (0 == $step) {
            throw new \InvalidArgumentException('Step cannot be zero');
        }

        if (DType::Bool === $dtype) {
            throw new \InvalidArgumentException('Bool dtype not supported for arange');
        }

        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

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
     * @param float $start    The starting value of the sequence
     * @param float $stop     The end value of the sequence
     * @param int   $num      Number of samples to generate
     * @param bool  $endpoint If true, stop is the last sample
     * @param DType $dtype    Data type (default: Float64)
     */
    public static function linspace(
        float $start,
        float $stop,
        int $num = 50,
        bool $endpoint = true,
        DType $dtype = DType::Float64
    ): self {
        if ($num <= 0) {
            throw new ShapeException("Number of samples, {$num}, must be positive");
        }

        if (DType::Float32 !== $dtype && DType::Float64 !== $dtype) {
            throw new \InvalidArgumentException('linspace only supports Float32 and Float64 dtypes');
        }

        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

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
     * @param float $stop  The end exponent (base**stop is the final value)
     * @param int   $num   Number of samples to generate
     * @param float $base  The base of the log space
     * @param DType $dtype Data type (default: Float64)
     */
    public static function logspace(
        float $start,
        float $stop,
        int $num = 50,
        float $base = 10.0,
        DType $dtype = DType::Float64
    ): self {
        if ($num <= 0) {
            throw new ShapeException("Number of samples, {$num}, must be positive");
        }

        if (DType::Float32 !== $dtype && DType::Float64 !== $dtype) {
            throw new \InvalidArgumentException('logspace only supports Float32 and Float64 dtypes');
        }

        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

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
     * @param float $stop  The end value of the sequence
     * @param int   $num   Number of samples to generate
     * @param DType $dtype Data type (default: Float64)
     */
    public static function geomspace(
        float $start,
        float $stop,
        int $num = 50,
        DType $dtype = DType::Float64
    ): self {
        if ($num <= 0) {
            throw new ShapeException("Number of samples, {$num}, must be positive");
        }

        if (DType::Float32 !== $dtype && DType::Float64 !== $dtype) {
            throw new \InvalidArgumentException('geomspace only supports Float32 and Float64 dtypes');
        }

        if (0.0 == $start || 0.0 == $stop) {
            throw new \InvalidArgumentException('geomspace does not support zero values');
        }

        if (($start > 0.0) !== ($stop > 0.0)) {
            throw new \InvalidArgumentException('geomspace requires start and stop to have the same sign');
        }

        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

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
     * @param null|DType $dtype Float dtype (default: Float64)
     * @param null|int   $seed  Optional seed for deterministic output
     */
    public static function random(array $shape, ?DType $dtype = null, ?int $seed = null): self
    {
        $dtype ??= DType::Float64;
        self::assertFloatDtype($dtype, 'random');

        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');
        $status = $ffi->ndarray_random(
            Lib::createShapeArray($shape),
            \count($shape),
            $dtype->value,
            null !== $seed,
            $seed ?? 0,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new self($outHandle, $shape, $dtype);
    }

    /**
     * Create random integer samples from [low, high).
     *
     * @param int        $low   Inclusive lower bound
     * @param int        $high  Exclusive upper bound
     * @param array<int> $shape Output shape
     * @param null|DType $dtype Integer dtype (default: Int64)
     * @param null|int   $seed  Optional seed for deterministic output
     */
    public static function randomInt(int $low, int $high, array $shape, ?DType $dtype = null, ?int $seed = null): self
    {
        $dtype ??= DType::Int64;
        if (!$dtype->isInteger()) {
            throw new DTypeException('randomInt only supports integer dtypes');
        }
        if ($high <= $low) {
            throw new ShapeException("randomInt requires high > low, got [{$low}, {$high})");
        }

        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');
        $status = $ffi->ndarray_random_int(
            $low,
            $high,
            Lib::createShapeArray($shape),
            \count($shape),
            $dtype->value,
            null !== $seed,
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
     * @param null|DType $dtype Float dtype (default: Float64)
     * @param null|int   $seed  Optional seed for deterministic output
     */
    public static function randn(array $shape, ?DType $dtype = null, ?int $seed = null): self
    {
        $dtype ??= DType::Float64;
        self::assertFloatDtype($dtype, 'randn');

        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');
        $status = $ffi->ndarray_randn(
            Lib::createShapeArray($shape),
            \count($shape),
            $dtype->value,
            null !== $seed,
            $seed ?? 0,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new self($outHandle, $shape, $dtype);
    }

    /**
     * Create random samples from a normal distribution N(mean, std).
     *
     * @param float      $mean  Mean of the distribution
     * @param float      $std   Standard deviation (must be > 0)
     * @param array<int> $shape Output shape
     * @param null|DType $dtype Float dtype (default: Float64)
     * @param null|int   $seed  Optional seed for deterministic output
     */
    public static function normal(float $mean, float $std, array $shape, ?DType $dtype = null, ?int $seed = null): self
    {
        $dtype ??= DType::Float64;
        self::assertFloatDtype($dtype, 'normal');
        if (!($std > 0.0)) {
            throw new \InvalidArgumentException("normal requires std > 0, got {$std}");
        }

        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');
        $status = $ffi->ndarray_normal(
            $mean,
            $std,
            Lib::createShapeArray($shape),
            \count($shape),
            $dtype->value,
            null !== $seed,
            $seed ?? 0,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new self($outHandle, $shape, $dtype);
    }

    /**
     * Create random samples from a uniform distribution over [low, high).
     *
     * @param float      $low   Inclusive lower bound
     * @param float      $high  Exclusive upper bound
     * @param array<int> $shape Output shape
     * @param null|DType $dtype Float dtype (default: Float64)
     * @param null|int   $seed  Optional seed for deterministic output
     */
    public static function uniform(float $low, float $high, array $shape, ?DType $dtype = null, ?int $seed = null): self
    {
        $dtype ??= DType::Float64;
        self::assertFloatDtype($dtype, 'uniform');
        if ($high <= $low) {
            throw new \InvalidArgumentException("uniform requires high > low, got [{$low}, {$high})");
        }

        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');
        $status = $ffi->ndarray_uniform(
            $low,
            $high,
            Lib::createShapeArray($shape),
            \count($shape),
            $dtype->value,
            null !== $seed,
            $seed ?? 0,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new self($outHandle, $shape, $dtype);
    }

    /**
     * Tile an array by repeating it along each axis.
     *
     * Static convenience method that accepts either a PHP array or NDArray.
     *
     * @param array<int>|self     $a    Input array or NDArray
     * @param array<int>|int|self $reps The number of repetitions of A along each axis
     *
     * @return self The tiled output array
     */
    public static function tileArray(array|self $a, array|int|self $reps): self
    {
        if (\is_array($a)) {
            $a = self::array($a);
        }

        return $a->tile($reps);
    }

    /**
     * Repeat elements of an array.
     *
     * Static convenience method that accepts either a PHP array or NDArray.
     *
     * @param array<mixed>|self   $a       Input array or NDArray
     * @param array<int>|int|self $repeats The number of repetitions for each element
     * @param null|int            $axis    The axis along which to repeat values. By default, use the flattened input array
     *
     * @return self Output array which has the same shape as a, except along the given axis
     */
    public static function repeatArray(array|self $a, array|int|self $repeats, ?int $axis = null): self
    {
        if (\is_array($a)) {
            $a = self::array($a);
        }

        return $a->repeat($repeats, $axis);
    }

    // =========================================================================
    // Private Helpers
    // =========================================================================

    /**
     * Infer shape from nested PHP array.
     *
     * @param array<mixed> $data Nested PHP array
     *
     * @return array<int>
     */
    private static function inferShape(array $data): array
    {
        $shape = [];
        $current = $data;

        while (\is_array($current)) {
            $count = \count($current);
            if (0 === $count) {
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
     * @param array<mixed> $data Nested PHP array
     *
     * @return array<mixed>
     */
    private static function flattenArray(array $data): array
    {
        $result = [];

        array_walk_recursive($data, static function ($value) use (&$result) {
            $result[] = $value;
        });

        return $result;
    }

    /**
     * Create an NDArray handle using the appropriate FFI function for the dtype.
     *
     * @param Bindings&\FFI $ffi   FFI instance
     * @param DType         $dtype Data type
     * @param array<mixed>  $data  Flat array of values
     * @param array<int>    $shape Array shape
     * @param int           $len   Number of elements
     *
     * @return CData Opaque handle
     */
    private static function createTyped(\FFI $ffi, DType $dtype, array $data, array $shape, int $len): CData
    {
        $data = $dtype->prepareArrayValues($data);

        $cData = $dtype->createCArray($len, $data);
        $cShape = Lib::createShapeArray($shape);

        $outHandle = $ffi->new('struct NdArrayHandle*');

        $status = $ffi->ndarray_create(
            $cData,
            $len,
            $cShape,
            \count($shape),
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
            throw new DTypeException("{$method} only supports Float32 and Float64 dtypes");
        }
    }
}
