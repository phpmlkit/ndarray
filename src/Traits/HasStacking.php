<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Traits;

use PhpMlKit\NDArray\Exceptions\ShapeException;
use PhpMlKit\NDArray\FFI\Lib;
use PhpMlKit\NDArray\NDArray;

/**
 * Joining and splitting operations.
 *
 * Provides concatenate, stack, vstack, hstack, split, vsplit, hsplit.
 */
trait HasStacking
{
    /**
     * Join arrays along an existing axis.
     *
     * All arrays must have the same shape except for the dimension along axis.
     *
     * @param array<NDArray> $arrays Arrays to concatenate
     * @param int            $axis   Axis along which to join (default 0)
     */
    public static function concatenate(array $arrays, int $axis = 0): NDArray
    {
        if (empty($arrays)) {
            throw new ShapeException('concatenate requires at least one array');
        }

        $numArrays = \count($arrays);
        $ndim = $arrays[0]->ndim();

        $axisResolved = $axis < 0 ? $ndim + $axis : $axis;
        if ($axisResolved < 0 || $axisResolved >= $ndim) {
            throw new ShapeException("Axis {$axis} out of bounds for array with {$ndim} dimensions");
        }

        foreach ($arrays as $i => $arr) {
            if ($arr->ndim() !== $ndim) {
                throw new ShapeException(
                    "concatenate requires all arrays to have the same number of dimensions (array {$i} has {$arr->ndim()}, expected {$ndim})"
                );
            }
        }

        $ffi = Lib::get();
        $metaWrappers = array_map(static fn (NDArray $a) => $a->viewMetadata()->toCData(), $arrays);
        $cHandles = $ffi->new("struct NdArrayHandle*[{$numArrays}]");
        $cMetas = $ffi->new("struct ViewMetadata*[{$numArrays}]");
        for ($i = 0; $i < $numArrays; ++$i) {
            $cHandles[$i] = $arrays[$i]->handle;
            $cMetas[$i] = Lib::addr($metaWrappers[$i]);
        }

        $outHandle = $ffi->new('struct NdArrayHandle*');
        $outNdimBuf = $ffi->new('size_t');
        $outShapeBuf = Lib::createCArray('size_t', array_fill(0, Lib::MAX_NDIM, 0));

        $status = $ffi->ndarray_concatenate(
            $cHandles,
            $cMetas,
            $numArrays,
            $axisResolved,
            Lib::addr($outHandle),
            Lib::addr($outNdimBuf),
            $outShapeBuf,
            Lib::MAX_NDIM
        );

        Lib::checkStatus($status);

        $outNdim = (int) $outNdimBuf->cdata;
        $outShape = Lib::extractShapeFromPointer($outShapeBuf, $outNdim);

        return new self($outHandle, $outShape, $arrays[0]->dtype);
    }

    /**
     * Stack arrays along a new axis.
     *
     * All arrays must have identical shapes.
     *
     * @param array<NDArray> $arrays Arrays to stack
     * @param int            $axis   Axis in the result at which the arrays are stacked
     */
    public static function stack(array $arrays, int $axis = 0): NDArray
    {
        if (empty($arrays)) {
            throw new ShapeException('stack requires at least one array');
        }

        $numArrays = \count($arrays);
        $ndim = $arrays[0]->ndim();

        $axisResolved = $axis < 0 ? $ndim + $axis + 1 : $axis;
        if ($axisResolved < 0 || $axisResolved > $ndim) {
            throw new ShapeException("Axis {$axis} out of bounds for stack");
        }

        foreach ($arrays as $i => $arr) {
            if ($arr->ndim() !== $ndim) {
                throw new ShapeException(
                    "stack requires all arrays to have the same number of dimensions (array {$i} has {$arr->ndim()}, expected {$ndim})"
                );
            }
        }

        $ffi = Lib::get();
        $metaWrappers = array_map(static fn (NDArray $a) => $a->viewMetadata()->toCData(), $arrays);
        $cHandles = $ffi->new("struct NdArrayHandle*[{$numArrays}]");
        $cMetas = $ffi->new("struct ViewMetadata*[{$numArrays}]");
        for ($i = 0; $i < $numArrays; ++$i) {
            $cHandles[$i] = $arrays[$i]->handle;
            $cMetas[$i] = Lib::addr($metaWrappers[$i]);
        }

        $outHandle = $ffi->new('struct NdArrayHandle*');
        $outNdimBuf = $ffi->new('size_t');
        $outShapeBuf = Lib::createCArray('size_t', array_fill(0, Lib::MAX_NDIM, 0));

        $status = $ffi->ndarray_stack(
            $cHandles,
            $cMetas,
            $numArrays,
            $axisResolved,
            Lib::addr($outHandle),
            Lib::addr($outNdimBuf),
            $outShapeBuf,
            Lib::MAX_NDIM
        );

        Lib::checkStatus($status);

        $outNdim = (int) $outNdimBuf->cdata;
        $outShape = Lib::extractShapeFromPointer($outShapeBuf, $outNdim);

        return new self($outHandle, $outShape, $arrays[0]->dtype);
    }

    /**
     * Stack arrays vertically (along axis 0).
     *
     * Equivalent to concatenate(arrays, axis=0).
     *
     * @param array<NDArray> $arrays Arrays to stack
     */
    public static function vstack(array $arrays): NDArray
    {
        return self::concatenate($arrays, 0);
    }

    /**
     * Stack arrays horizontally (along axis 1).
     *
     * Equivalent to concatenate(arrays, axis=1).
     *
     * @param array<NDArray> $arrays Arrays to stack
     */
    public static function hstack(array $arrays): NDArray
    {
        return self::concatenate($arrays, 1);
    }

    /**
     * Split array along axis.
     *
     * If $indicesOrSections is an integer N, split into N equal parts (axis length must be divisible by N).
     * If it is an array of indices, split at those positions.
     *
     * @param array<int>|int $indicesOrSections Number of equal parts, or array of split indices
     * @param int            $axis              Axis along which to split
     *
     * @return array<NDArray> List of sub-arrays (views)
     */
    public function split(array|int $indicesOrSections, int $axis = 0): array
    {
        $ffi = Lib::get();
        $ndim = $this->ndim();

        $axisResolved = $axis < 0 ? $ndim + $axis : $axis;
        if ($axisResolved < 0 || $axisResolved >= $ndim) {
            throw new ShapeException("Axis {$axis} out of bounds for array with {$ndim} dimensions");
        }

        $axisLen = $this->shape()[$axisResolved];

        $indices = \is_int($indicesOrSections)
            ? self::indicesForEqualSplit($axisLen, $indicesOrSections)
            : $indicesOrSections;

        if (empty($indices)) {
            return [$this];
        }

        $numParts = \count($indices) + 1;
        $cIndices = Lib::createCArray('size_t', $indices);
        $cOutOffsets = $ffi->new("size_t[{$numParts}]");
        $cOutShapes = $ffi->new('size_t['.($numParts * $ndim).']');
        $cOutStrides = $ffi->new('size_t['.($numParts * $ndim).']');

        $meta = $this->viewMetadata()->toCData();
        $status = $ffi->ndarray_split(
            $this->handle,
            Lib::addr($meta),
            $axisResolved,
            $cIndices,
            \count($indices),
            $cOutOffsets,
            $cOutShapes,
            $cOutStrides
        );

        Lib::checkStatus($status);

        $base = $this->base ?? $this;
        $result = [];
        for ($i = 0; $i < $numParts; ++$i) {
            $partShape = [];
            $partStrides = [];
            for ($d = 0; $d < $ndim; ++$d) {
                $partShape[] = (int) $cOutShapes[$i * $ndim + $d];
                $partStrides[] = (int) $cOutStrides[$i * $ndim + $d];
            }
            $partOffset = (int) $cOutOffsets[$i];
            $result[] = new self(
                $this->handle,
                $partShape,
                $this->dtype,
                $partStrides,
                $partOffset,
                $base
            );
        }

        return $result;
    }

    /**
     * Split array vertically (along axis 0).
     *
     * @param array<int>|int $indicesOrSections Number of equal parts or split indices
     *
     * @return array<NDArray>
     */
    public function vsplit(array|int $indicesOrSections): array
    {
        return $this->split($indicesOrSections, 0);
    }

    /**
     * Split array horizontally (along axis 1).
     *
     * @param array<int>|int $indicesOrSections Number of equal parts or split indices
     *
     * @return array<NDArray>
     */
    public function hsplit(array|int $indicesOrSections): array
    {
        return $this->split($indicesOrSections, 1);
    }

    /**
     * Compute split indices for equal N-way split.
     *
     * @return array<int>
     */
    private static function indicesForEqualSplit(int $axisLen, int $n): array
    {
        if ($n < 1) {
            throw new ShapeException('Number of sections must be >= 1');
        }
        if (0 !== $axisLen % $n) {
            throw new ShapeException(
                "Array split does not result in an equal division (axis length {$axisLen} not divisible by {$n})"
            );
        }
        $chunk = (int) ($axisLen / $n);
        $indices = [];
        for ($i = 1; $i < $n; ++$i) {
            $indices[] = $i * $chunk;
        }

        return $indices;
    }
}
