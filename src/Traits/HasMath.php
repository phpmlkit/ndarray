<?php

declare(strict_types=1);

namespace NDArray\Traits;

use NDArray\DType;
use NDArray\NDArray;
use NDArray\FFI\Lib;

/**
 * Mathematical operations trait for NDArray.
 *
 * Provides element-wise arithmetic operations with broadcasting support
 * for both array-array and array-scalar operations.
 */
trait HasMath
{
    /**
     * Add another array or scalar to this array.
     *
     * @param NDArray|float|int $other Array or scalar to add
     * @return NDArray New array with result
     */
    public function add(NDArray|float|int $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_add', $other);
        }
        return $this->scalarOp('ndarray_add_scalar', (float) $other);
    }

    /**
     * Subtract another array or scalar from this array.
     *
     * @param NDArray|float|int $other Array or scalar to subtract
     * @return NDArray New array with result
     */
    public function subtract(NDArray|float|int $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_sub', $other);
        }
        return $this->scalarOp('ndarray_sub_scalar', (float) $other);
    }

    /**
     * Multiply this array by another array or scalar.
     *
     * @param NDArray|float|int $other Array or scalar to multiply by
     * @return NDArray New array with result
     */
    public function multiply(NDArray|float|int $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_mul', $other);
        }
        return $this->scalarOp('ndarray_mul_scalar', (float) $other);
    }

    /**
     * Divide this array by another array or scalar.
     *
     * @param NDArray|float|int $other Array or scalar to divide by
     * @return NDArray New array with result
     */
    public function divide(NDArray|float|int $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_div', $other);
        }
        return $this->scalarOp('ndarray_div_scalar', (float) $other);
    }

    /**
     * Negate all elements of this array.
     *
     * @return NDArray
     */
    public function negate(): NDArray
    {
        return $this->multiply(-1);
    }

    /**
     * Compute the absolute value of all elements.
     *
     * @return NDArray
     */
    public function abs(): NDArray
    {
        $data = $this->toArray();
        $result = $this->absRecursive($data);
        return NDArray::array($result, $this->dtype);
    }

    /**
     * Helper to compute absolute value recursively for nested arrays.
     *
     * @param array $data
     * @return array
     */
    private function absRecursive(array $data): array
    {
        $result = [];
        foreach ($data as $key => $value) {
            if (is_array($value)) {
                $result[$key] = $this->absRecursive($value);
            } else {
                $result[$key] = abs($value);
            }
        }
        return $result;
    }

    /**
     * Perform a binary operation with another array.
     *
     * @param string $funcName FFI function name
     * @param NDArray $other Other array
     * @return NDArray
     */
    private function binaryOp(string $funcName, NDArray $other): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new("struct NdArrayHandle*");

        // Get view metadata for both arrays
        $aShape = Lib::createShapeArray($this->shape);
        $aStrides = Lib::createShapeArray($this->strides);
        $bShape = Lib::createShapeArray($other->shape);
        $bStrides = Lib::createShapeArray($other->strides);

        // Use the maximum ndim for broadcasting
        $ndim = max(count($this->shape), count($other->shape));

        $status = $ffi->$funcName(
            $this->handle,
            $this->offset,
            $aShape,
            $aStrides,
            $other->handle,
            $other->offset,
            $bShape,
            $bStrides,
            $ndim,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        // Determine output shape based on broadcasting rules
        $outShape = $this->broadcastShapes($this->shape, $other->shape);

        // Determine output dtype via type promotion
        $outDtype = DType::promote($this->dtype, $other->dtype);

        return new NDArray($outHandle, $outShape, $outDtype);
    }

    /**
     * Perform a scalar operation.
     *
     * @param string $funcName FFI function name
     * @param float $scalar Scalar value
     * @return NDArray
     */
    private function scalarOp(string $funcName, float $scalar): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new("struct NdArrayHandle*");

        $aShape = Lib::createShapeArray($this->shape);
        $aStrides = Lib::createShapeArray($this->strides);

        $status = $ffi->$funcName(
            $this->handle,
            $this->offset,
            $aShape,
            $aStrides,
            count($this->shape),
            $scalar,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new NDArray($outHandle, $this->shape, $this->dtype);
    }

    /**
     * Compute the broadcasted shape of two arrays.
     *
     * @param array<int> $shapeA
     * @param array<int> $shapeB
     * @return array<int>
     */
    private function broadcastShapes(array $shapeA, array $shapeB): array
    {
        $ndimA = count($shapeA);
        $ndimB = count($shapeB);
        $maxNdim = max($ndimA, $ndimB);

        // Pad shorter shape with 1s on the left
        $paddedA = array_merge(array_fill(0, $maxNdim - $ndimA, 1), $shapeA);
        $paddedB = array_merge(array_fill(0, $maxNdim - $ndimB, 1), $shapeB);

        $result = [];
        for ($i = 0; $i < $maxNdim; $i++) {
            $dimA = $paddedA[$i];
            $dimB = $paddedB[$i];

            if ($dimA === 1) {
                $result[] = $dimB;
            } elseif ($dimB === 1) {
                $result[] = $dimA;
            } elseif ($dimA === $dimB) {
                $result[] = $dimA;
            } else {
                throw new \NDArray\Exceptions\ShapeException(
                    "Cannot broadcast shapes " . json_encode($shapeA) . " and " . json_encode($shapeB)
                );
            }
        }

        return $result;
    }

    // =========================================================================
    // Static Methods
    // =========================================================================

    /**
     * Add two arrays element-wise.
     *
     * @param NDArray $a
     * @param NDArray|float|int $b
     * @return NDArray
     */
    public static function addArrays(NDArray $a, NDArray|float|int $b): NDArray
    {
        return $a->add($b);
    }

    /**
     * Subtract two arrays element-wise.
     *
     * @param NDArray $a
     * @param NDArray|float|int $b
     * @return NDArray
     */
    public static function subtractArrays(NDArray $a, NDArray|float|int $b): NDArray
    {
        return $a->subtract($b);
    }

    /**
     * Multiply two arrays element-wise.
     *
     * @param NDArray $a
     * @param NDArray|float|int $b
     * @return NDArray
     */
    public static function multiplyArrays(NDArray $a, NDArray|float|int $b): NDArray
    {
        return $a->multiply($b);
    }

    /**
     * Divide two arrays element-wise.
     *
     * @param NDArray $a
     * @param NDArray|float|int $b
     * @return NDArray
     */
    public static function divideArrays(NDArray $a, NDArray|float|int $b): NDArray
    {
        return $a->divide($b);
    }
}
