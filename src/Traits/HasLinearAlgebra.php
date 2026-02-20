<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Traits;

use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\NDArray;
use PhpMlKit\NDArray\FFI\Lib;

/**
 * Linear algebra operations trait.
 *
 * Provides matrix operations like dot product, matrix multiplication,
 * trace, and diagonal extraction.
 */
trait HasLinearAlgebra
{
    /**
     * Compute vector or matrix norm.
     *
     * Supported orders:
     * - 1
     * - 2
     * - INF
     * - -INF
     * - 'fro' (matrix only, axis=null)
     *
     * @param int|float|string|null $ord Norm order
     * @param int|null $axis Reduction axis. If null, reduces all elements
     * @param bool $keepdims Keep reduced axis with size 1 (axis mode only)
     * @return NDArray|float
     */
    public function norm(int|float|string|null $ord = null, ?int $axis = null, bool $keepdims = false): NDArray|float
    {
        $ffi = Lib::get();
        $ordCode = $this->normalizeNormOrder($ord, $axis);
        $ndim = count($this->shape);

        if ($ordCode === 5 && $ndim !== 2) {
            throw new \InvalidArgumentException("Norm order '{$ord}' requires a 2D matrix");
        }
        if ($ordCode === 5 && $axis !== null) {
            throw new \InvalidArgumentException("Norm order '{$ord}' is only supported when axis is null");
        }

        $shape = Lib::createShapeArray($this->shape);
        $strides = Lib::createShapeArray($this->strides);

        if ($axis === null) {
            $outValue = Lib::createBox('double');
            $outDtype = $ffi->new('uint8_t');
            $status = $ffi->ndarray_norm(
                $this->handle,
                $this->offset,
                $shape,
                $strides,
                $ndim,
                $ordCode,
                \FFI::addr($outValue),
                Lib::addr($outDtype)
            );

            Lib::checkStatus($status);
            return (float) $outValue->cdata;
        }

        $outHandle = $ffi->new('struct NdArrayHandle*');
        $status = $ffi->ndarray_norm_axis(
            $this->handle,
            $this->offset,
            $shape,
            $strides,
            $ndim,
            $axis,
            $keepdims,
            $ordCode,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);
        $outShape = $this->computeNormOutputShape($axis, $keepdims);
        return new NDArray($outHandle, $outShape, DType::Float64);
    }

    /**
     * Compute dot product of two arrays.
     *
     * For 1D arrays: returns scalar
     * For 2D arrays: returns matrix multiplication
     *
     * @param NDArray $other The other array
     * @return NDArray
     */
    public function dot(NDArray $other): NDArray
    {
        return $this->binaryOp('ndarray_dot', $other);
    }

    /**
     * Compute matrix multiplication.
     *
     * Requires both arrays to be at least 2D.
     *
     * @param NDArray $other The other array
     * @return NDArray
     */
    public function matmul(NDArray $other): NDArray
    {
        return $this->binaryOp('ndarray_matmul', $other);
    }

    /**
     * Extract diagonal elements from a 2D array.
     *
     * Returns a 1D array containing the diagonal.
     *
     * @return NDArray
     */
    public function diagonal(): NDArray
    {
        return $this->unaryOp('ndarray_diagonal');
    }

    /**
     * Compute trace (sum of diagonal elements).
     *
     * Returns a scalar array.
     *
     * @return NDArray
     */
    public function trace(): NDArray
    {
        return $this->unaryOp('ndarray_trace');
    }

    /**
     * Compute output shape for dot product.
     *
     * @param array $aShape Shape of first array
     * @param array $bShape Shape of second array
     * @return array
     */
    private function computeDotShape(array $aShape, array $bShape): array
    {
        // 1D @ 1D -> scalar (0D)
        if (count($aShape) === 1 && count($bShape) === 1) {
            return [];
        }
        
        // 2D @ 1D -> 1D (M)
        if (count($aShape) === 2 && count($bShape) === 1) {
            return [$aShape[0]];
        }
        
        // 1D @ 2D -> 1D (N)
        if (count($aShape) === 1 && count($bShape) === 2) {
            return [$bShape[1]];
        }
        
        // 2D @ 2D -> 2D (M x N)
        if (count($aShape) === 2 && count($bShape) === 2) {
            return [$aShape[0], $bShape[1]];
        }
        
        // Default to empty for other cases
        return [];
    }

    /**
     * Compute output shape for matrix multiplication.
     *
     * @param array $aShape Shape of first array
     * @param array $bShape Shape of second array
     * @return array
     */
    private function computeMatmulShape(array $aShape, array $bShape): array
    {
        // Simple 2D @ 2D -> (M x K)
        if (count($aShape) === 2 && count($bShape) === 2) {
            return [$aShape[0], $bShape[1]];
        }
        
        // For other cases, try to infer
        return [$aShape[0], $bShape[count($bShape) - 1]];
    }

    /**
     * Normalize norm order into FFI code.
     */
    private function normalizeNormOrder(int|float|string|null $ord, ?int $axis): int
    {
        if ($ord === null) {
            if ($axis === null && count($this->shape) === 2) {
                return 5; // fro
            }
            return 2;
        }

        if (is_int($ord)) {
            return match ($ord) {
                1 => 1,
                2 => 2,
                default => throw new \InvalidArgumentException("Unsupported norm order: {$ord}"),
            };
        }

        if (is_float($ord)) {
            if ($ord === INF) {
                return 3;
            }
            if ($ord === -INF) {
                return 4;
            }
            if ($ord === 1.0) {
                return 1;
            }
            if ($ord === 2.0) {
                return 2;
            }
            throw new \InvalidArgumentException("Unsupported norm order: {$ord}");
        }

        $normalized = strtolower(trim($ord));
        return match ($normalized) {
            '1' => 1,
            '2' => 2,
            'inf', '+inf' => 3,
            '-inf' => 4,
            'fro' => 5,
            default => throw new \InvalidArgumentException("Unsupported norm order: {$ord}"),
        };
    }

    /**
     * Compute norm output shape for axis reduction.
     *
     * @return array<int>
     */
    private function computeNormOutputShape(int $axis, bool $keepdims): array
    {
        $shape = $this->shape;
        $ndim = count($shape);
        $axisNorm = $axis < 0 ? $ndim + $axis : $axis;

        if ($keepdims) {
            $shape[$axisNorm] = 1;
            return $shape;
        }

        $out = array_values(array_filter(
            $shape,
            static fn (int $idx): bool => $idx !== $axisNorm,
            ARRAY_FILTER_USE_KEY
        ));
        return $out;
    }
}
