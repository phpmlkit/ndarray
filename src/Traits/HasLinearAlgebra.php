<?php

declare(strict_types=1);

namespace NDArray\Traits;

use NDArray\NDArray;
use NDArray\FFI\Lib;

/**
 * Linear algebra operations trait.
 *
 * Provides matrix operations like dot product, matrix multiplication,
 * trace, and diagonal extraction.
 */
trait HasLinearAlgebra
{
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
        $ffi = Lib::get();
        $out = $ffi->new('void*');

        $ndimA = count($this->shape);
        $ndimB = count($other->shape);

        $shapeA = $ffi->new('size_t[' . $ndimA . ']', false);
        $shapeB = $ffi->new('size_t[' . $ndimB . ']', false);
        $stridesA = $ffi->new('size_t[' . $ndimA . ']', false);
        $stridesB = $ffi->new('size_t[' . $ndimB . ']', false);

        foreach ($this->shape as $i => $dim) {
            $shapeA[$i] = $dim;
        }
        foreach ($this->strides as $i => $stride) {
            $stridesA[$i] = $stride;
        }
        foreach ($other->shape as $i => $dim) {
            $shapeB[$i] = $dim;
        }
        foreach ($other->strides as $i => $stride) {
            $stridesB[$i] = $stride;
        }

        $result = $ffi->ndarray_dot(
            $this->handle,
            $this->offset,
            $shapeA,
            $stridesA,
            $ndimA,
            $other->handle,
            $other->offset,
            $shapeB,
            $stridesB,
            $ndimB,
            \FFI::addr($out)
        );

        if ($result !== 0) {
            throw new \RuntimeException('Dot product failed: ' . Lib::getLastError());
        }

        // Determine output shape based on input shapes
        $aShape = $this->shape();
        $bShape = $other->shape();
        $outShape = $this->computeDotShape($aShape, $bShape);

        return new NDArray(
            $out,
            $outShape,
            $this->dtype,
            [],
            0,
            null
        );
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
        $ffi = Lib::get();
        $out = $ffi->new('void*');

        $ndimA = count($this->shape);
        $ndimB = count($other->shape);

        $shapeA = $ffi->new('size_t[' . $ndimA . ']', false);
        $shapeB = $ffi->new('size_t[' . $ndimB . ']', false);
        $stridesA = $ffi->new('size_t[' . $ndimA . ']', false);
        $stridesB = $ffi->new('size_t[' . $ndimB . ']', false);

        foreach ($this->shape as $i => $dim) {
            $shapeA[$i] = $dim;
        }
        foreach ($this->strides as $i => $stride) {
            $stridesA[$i] = $stride;
        }
        foreach ($other->shape as $i => $dim) {
            $shapeB[$i] = $dim;
        }
        foreach ($other->strides as $i => $stride) {
            $stridesB[$i] = $stride;
        }

        $result = $ffi->ndarray_matmul(
            $this->handle,
            $this->offset,
            $shapeA,
            $stridesA,
            $ndimA,
            $other->handle,
            $other->offset,
            $shapeB,
            $stridesB,
            $ndimB,
            \FFI::addr($out)
        );

        if ($result !== 0) {
            throw new \RuntimeException('Matrix multiplication failed: ' . Lib::getLastError());
        }

        // Determine output shape based on input shapes
        $aShape = $this->shape();
        $bShape = $other->shape();
        $outShape = $this->computeMatmulShape($aShape, $bShape);

        return new NDArray(
            $out,
            $outShape,
            $this->dtype,
            [],
            0,
            null
        );
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
        $ffi = Lib::get();
        $out = $ffi->new('void*');

        $ndim = count($this->shape);
        $shape = $ffi->new('size_t[' . $ndim . ']', false);
        $strides = $ffi->new('size_t[' . $ndim . ']', false);

        foreach ($this->shape as $i => $dim) {
            $shape[$i] = $dim;
        }
        foreach ($this->strides as $i => $stride) {
            $strides[$i] = $stride;
        }

        $result = $ffi->ndarray_diagonal(
            $this->handle,
            $this->offset,
            $shape,
            $strides,
            $ndim,
            \FFI::addr($out)
        );

        if ($result !== 0) {
            throw new \RuntimeException('Diagonal extraction failed: ' . Lib::getLastError());
        }

        // Output is 1D with length = min(M, N)
        $shapeArr = $this->shape();
        $diagLen = min($shapeArr[0], $shapeArr[1] ?? $shapeArr[0]);

        return new NDArray(
            $out,
            [$diagLen],
            $this->dtype,
            [],
            0,
            null
        );
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
        $ffi = Lib::get();
        $out = $ffi->new('void*');

        $ndim = count($this->shape);
        $shape = $ffi->new('size_t[' . $ndim . ']', false);
        $strides = $ffi->new('size_t[' . $ndim . ']', false);

        foreach ($this->shape as $i => $dim) {
            $shape[$i] = $dim;
        }
        foreach ($this->strides as $i => $stride) {
            $strides[$i] = $stride;
        }

        $result = $ffi->ndarray_trace(
            $this->handle,
            $this->offset,
            $shape,
            $strides,
            $ndim,
            \FFI::addr($out)
        );

        if ($result !== 0) {
            throw new \RuntimeException('Trace computation failed: ' . Lib::getLastError());
        }

        // Output is scalar (0D)
        return new NDArray(
            $out,
            [],
            $this->dtype,
            [],
            0,
            null
        );
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
}
