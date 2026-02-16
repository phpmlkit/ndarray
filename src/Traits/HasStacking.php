<?php

declare(strict_types=1);

namespace NDArray\Traits;

use NDArray\Exceptions\ShapeException;
use NDArray\FFI\Lib;
use NDArray\NDArray;

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
     * @param int $axis Axis along which to join (default 0)
     * @return NDArray
     */
    public static function concatenate(array $arrays, int $axis = 0): NDArray
    {
        if (empty($arrays)) {
            throw new ShapeException('concatenate requires at least one array');
        }

        $numArrays = count($arrays);
        $ndim = $arrays[0]->ndim;

        $axisResolved = $axis < 0 ? $ndim + $axis : $axis;
        if ($axisResolved < 0 || $axisResolved >= $ndim) {
            throw new ShapeException("Axis $axis out of bounds for array with $ndim dimensions");
        }

        foreach ($arrays as $i => $arr) {
            if ($arr->ndim !== $ndim) {
                throw new ShapeException(
                    "concatenate requires all arrays to have the same number of dimensions (array $i has {$arr->ndim}, expected $ndim)"
                );
            }
        }

        $ffi = Lib::get();

        $cHandles = $ffi->new("struct NdArrayHandle*[$numArrays]");
        $cOffsets = $ffi->new("size_t[$numArrays]");
        $cShapes = $ffi->new("size_t[" . ($numArrays * $ndim) . "]");
        $cStrides = $ffi->new("size_t[" . ($numArrays * $ndim) . "]");

        for ($i = 0; $i < $numArrays; $i++) {
            $arr = $arrays[$i];
            $cHandles[$i] = $arr->handle;
            $cOffsets[$i] = $arr->offset;
            for ($d = 0; $d < $ndim; $d++) {
                $cShapes[$i * $ndim + $d] = $arr->shape[$d];
                $cStrides[$i * $ndim + $d] = $arr->strides[$d];
            }
        }

        $outHandle = $ffi->new('struct NdArrayHandle*');
        $status = $ffi->ndarray_concatenate(
            $cHandles,
            $cOffsets,
            $cShapes,
            $cStrides,
            $numArrays,
            $ndim,
            $axisResolved,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        $newShape = $arrays[0]->shape;
        $newShape[$axisResolved] = 0;
        foreach ($arrays as $arr) {
            $newShape[$axisResolved] += $arr->shape[$axisResolved];
        }

        return new self($outHandle, $newShape, $arrays[0]->dtype);
    }

    /**
     * Stack arrays along a new axis.
     *
     * All arrays must have identical shapes.
     *
     * @param array<NDArray> $arrays Arrays to stack
     * @param int $axis Axis in the result at which the arrays are stacked
     * @return NDArray
     */
    public static function stack(array $arrays, int $axis = 0): NDArray
    {
        if (empty($arrays)) {
            throw new ShapeException('stack requires at least one array');
        }

        $numArrays = count($arrays);
        $ndim = $arrays[0]->ndim;

        $axisResolved = $axis < 0 ? $ndim + $axis + 1 : $axis;
        if ($axisResolved < 0 || $axisResolved > $ndim) {
            throw new ShapeException("Axis $axis out of bounds for stack");
        }

        foreach ($arrays as $i => $arr) {
            if ($arr->ndim !== $ndim) {
                throw new ShapeException(
                    "stack requires all arrays to have the same number of dimensions (array $i has {$arr->ndim}, expected $ndim)"
                );
            }
        }

        $ffi = Lib::get();

        $cHandles = $ffi->new("struct NdArrayHandle*[$numArrays]");
        $cOffsets = $ffi->new("size_t[$numArrays]");
        $cShapes = $ffi->new("size_t[" . ($numArrays * $ndim) . "]");
        $cStrides = $ffi->new("size_t[" . ($numArrays * $ndim) . "]");

        for ($i = 0; $i < $numArrays; $i++) {
            $arr = $arrays[$i];
            $cHandles[$i] = $arr->handle;
            $cOffsets[$i] = $arr->offset;
            for ($d = 0; $d < $ndim; $d++) {
                $cShapes[$i * $ndim + $d] = $arr->shape[$d];
                $cStrides[$i * $ndim + $d] = $arr->strides[$d];
            }
        }

        $outHandle = $ffi->new('struct NdArrayHandle*');
        $status = $ffi->ndarray_stack(
            $cHandles,
            $cOffsets,
            $cShapes,
            $cStrides,
            $numArrays,
            $ndim,
            $axisResolved,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        $newShape = $arrays[0]->shape;
        array_splice($newShape, $axisResolved, 0, [$numArrays]);

        return new self($outHandle, $newShape, $arrays[0]->dtype);
    }

    /**
     * Stack arrays vertically (along axis 0).
     *
     * Equivalent to concatenate(arrays, axis=0).
     *
     * @param array<NDArray> $arrays Arrays to stack
     * @return NDArray
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
     * @return NDArray
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
     * @param int|array<int> $indicesOrSections Number of equal parts, or array of split indices
     * @param int $axis Axis along which to split
     * @return array<NDArray> List of sub-arrays (views)
     */
    public function split(int|array $indicesOrSections, int $axis = 0): array
    {
        $ffi = Lib::get();
        $ndim = $this->ndim;

        $axisResolved = $axis < 0 ? $ndim + $axis : $axis;
        if ($axisResolved < 0 || $axisResolved >= $ndim) {
            throw new ShapeException("Axis $axis out of bounds for array with $ndim dimensions");
        }

        $axisLen = $this->shape[$axisResolved];

        $indices = is_int($indicesOrSections)
            ? self::indicesForEqualSplit($axisLen, $indicesOrSections)
            : $indicesOrSections;

        if (empty($indices)) {
            return [$this];
        }

        $numParts = count($indices) + 1;
        $cIndices = Lib::createCArray('size_t', $indices);
        $cOutOffsets = $ffi->new("size_t[$numParts]");
        $cOutShapes = $ffi->new("size_t[" . ($numParts * $ndim) . "]");
        $cOutStrides = $ffi->new("size_t[" . ($numParts * $ndim) . "]");

        $status = $ffi->ndarray_split(
            $this->handle,
            $this->offset,
            Lib::createShapeArray($this->shape),
            Lib::createCArray('size_t', $this->strides),
            $ndim,
            $axisResolved,
            $cIndices,
            count($indices),
            $cOutOffsets,
            $cOutShapes,
            $cOutStrides
        );

        Lib::checkStatus($status);

        $base = $this->base ?? $this;
        $result = [];
        for ($i = 0; $i < $numParts; $i++) {
            $partShape = [];
            $partStrides = [];
            for ($d = 0; $d < $ndim; $d++) {
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
     * @param int|array<int> $indicesOrSections Number of equal parts or split indices
     * @return array<NDArray>
     */
    public function vsplit(int|array $indicesOrSections): array
    {
        return $this->split($indicesOrSections, 0);
    }

    /**
     * Split array horizontally (along axis 1).
     *
     * @param int|array<int> $indicesOrSections Number of equal parts or split indices
     * @return array<NDArray>
     */
    public function hsplit(int|array $indicesOrSections): array
    {
        return $this->split($indicesOrSections, 1);
    }

    /**
     * Compute split indices for equal N-way split.
     */
    private static function indicesForEqualSplit(int $axisLen, int $n): array
    {
        if ($n < 1) {
            throw new ShapeException("Number of sections must be >= 1");
        }
        if ($axisLen % $n !== 0) {
            throw new ShapeException(
                "Array split does not result in an equal division (axis length $axisLen not divisible by $n)"
            );
        }
        $chunk = (int) ($axisLen / $n);
        $indices = [];
        for ($i = 1; $i < $n; $i++) {
            $indices[] = $i * $chunk;
        }
        return $indices;
    }
}
