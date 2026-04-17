<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Traits;

use PhpMlKit\NDArray\ArrayMetadata;
use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\FFI\Lib;
use PhpMlKit\NDArray\NDArray;

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
     * @param null|float|int|string $ord      Norm order
     * @param null|int              $axis     Reduction axis. If null, reduces all elements
     * @param bool                  $keepdims Keep reduced axis with size 1 (axis mode only)
     */
    public function norm(float|int|string|null $ord = null, ?int $axis = null, bool $keepdims = false): float|NDArray
    {
        $ffi = Lib::get();
        $ordCode = $this->normalizeNormOrder($ord, $axis);
        $ndim = $this->ndim();

        if (5 === $ordCode && 2 !== $ndim) {
            throw new \InvalidArgumentException("Norm order '{$ord}' requires a 2D matrix");
        }
        if (5 === $ordCode && null !== $axis) {
            throw new \InvalidArgumentException("Norm order '{$ord}' is only supported when axis is null");
        }

        $meta = $this->meta()->toCData();
        if (null === $axis) {
            $outValue = $ffi->new('double');
            $outDtype = $ffi->new('uint8_t');
            $status = $ffi->ndarray_norm(
                $this->handle,
                Lib::addr($meta),
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
            Lib::addr($meta),
            $axis,
            $keepdims,
            $ordCode,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);
        $outShape = $this->computeNormOutputShape($axis, $keepdims);

        return new NDArray($outHandle, new ArrayMetadata($outShape), DType::Float64);
    }

    /**
     * Compute dot product of two arrays.
     *
     * For 1D arrays: returns scalar
     * For 2D arrays: returns matrix multiplication
     *
     * @param NDArray $other The other array
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
     * @param int $offset Diagonal offset. 0 = main diagonal, positive = upper diagonal, negative = lower diagonal
     */
    public function diagonal(int $offset = 0): NDArray
    {
        return $this->unaryOp('ndarray_diagonal', $offset);
    }

    /**
     * Extract a diagonal or construct a diagonal array.
     *
     * If the input is 1D: returns a 2D array with the input as the diagonal.
     * If the input is 2D: returns a 1D array containing the diagonal.
     *
     * @param int $offset Diagonal offset. 0 = main diagonal, positive = upper diagonal, negative = lower diagonal
     */
    public function diag(int $offset = 0): NDArray
    {
        return 1 === $this->ndim()
            ? $this->unaryOp('ndarray_from_diag', $offset)
            : $this->unaryOp('ndarray_diagonal', $offset);
    }

    /**
     * Compute trace (sum of diagonal elements).
     *
     * Returns a scalar array.
     */
    public function trace(): NDArray
    {
        return $this->unaryOp('ndarray_trace');
    }

    /**
     * Solve a linear system A * x = b.
     *
     * A must be a 2D square matrix. b can be 1D or 2D.
     *
     * @param NDArray $b Right-hand side array
     */
    public function solve(NDArray $b): NDArray
    {
        return $this->binaryOp('ndarray_solve', $b);
    }

    /**
     * Compute the inverse of a square matrix.
     *
     * Requires a 2D square matrix.
     */
    public function inv(): NDArray
    {
        return $this->unaryOp('ndarray_inv');
    }

    /**
     * Compute the determinant of a square matrix.
     *
     * Requires a 2D square matrix.
     */
    public function det(): float
    {
        return $this->scalarReductionOp('ndarray_det');
    }

    /**
     * Compute Singular Value Decomposition (SVD).
     *
     * Decomposes matrix A into U * S * V^T where:
     * - U is an orthogonal matrix (left singular vectors)
     * - S is a diagonal matrix of singular values (returned as 1D array)
     * - V^T is an orthogonal matrix (right singular vectors transposed)
     *
     * @param bool $computeUv If true, compute U and V^T matrices. If false, only compute singular values.
     *
     * @return ($computeUv is true ? array{0: NDArray, 1: NDArray, 2: NDArray} : NDArray)
     */
    public function svd(bool $computeUv = true): array|NDArray
    {
        $ffi = Lib::get();
        $meta = $this->meta()->toCData();
        $calcUv = $computeUv ? 1 : 0;
        $maxNdim = 8;

        $outHandleS = $ffi->new('struct NdArrayHandle*');
        $outDtypeS = $ffi->new('uint8_t');
        $outNdimS = $ffi->new('size_t');
        $outShapeS = $ffi->new("size_t[{$maxNdim}]");

        $outHandleU = $computeUv ? $ffi->new('struct NdArrayHandle*') : null;
        $outDtypeU = $computeUv ? $ffi->new('uint8_t') : null;
        $outNdimU = $computeUv ? $ffi->new('size_t') : null;
        $outShapeU = $computeUv ? $ffi->new("size_t[{$maxNdim}]") : null;

        $outHandleVt = $computeUv ? $ffi->new('struct NdArrayHandle*') : null;
        $outDtypeVt = $computeUv ? $ffi->new('uint8_t') : null;
        $outNdimVt = $computeUv ? $ffi->new('size_t') : null;
        $outShapeVt = $computeUv ? $ffi->new("size_t[{$maxNdim}]") : null;

        $status = $ffi->ndarray_svd(
            $this->handle,
            Lib::addr($meta),
            $calcUv,
            $calcUv,
            $computeUv ? Lib::addr($outHandleU) : null,
            $computeUv ? Lib::addr($outDtypeU) : null,
            $computeUv ? Lib::addr($outNdimU) : null,
            $computeUv ? $outShapeU : null,
            $maxNdim,
            Lib::addr($outHandleS),
            Lib::addr($outDtypeS),
            Lib::addr($outNdimS),
            $outShapeS,
            $computeUv ? Lib::addr($outHandleVt) : null,
            $computeUv ? Lib::addr($outDtypeVt) : null,
            $computeUv ? Lib::addr($outNdimVt) : null,
            $computeUv ? $outShapeVt : null,
        );

        Lib::checkStatus($status);

        $sShape = Lib::extractShapeFromPointer($outShapeS, $outNdimS->cdata);
        $s = new NDArray($outHandleS, new ArrayMetadata($sShape), $this->dtype);

        if (!$computeUv) {
            return $s;
        }

        $uShape = Lib::extractShapeFromPointer($outShapeU, $outNdimU->cdata);
        $vtShape = Lib::extractShapeFromPointer($outShapeVt, $outNdimVt->cdata);

        $u = new NDArray($outHandleU, new ArrayMetadata($uShape), $this->dtype);
        $vt = new NDArray($outHandleVt, new ArrayMetadata($vtShape), $this->dtype);

        return [$u, $s, $vt];
    }

    /**
     * Compute QR decomposition.
     *
     * Decomposes matrix A into Q * R where Q is orthogonal and R is upper triangular.
     *
     * @return array{0: NDArray, 1: NDArray} [Q, R]
     */
    public function qr(): array
    {
        $ffi = Lib::get();
        $meta = $this->meta()->toCData();
        $maxNdim = 8;

        $outHandleQ = $ffi->new('struct NdArrayHandle*');
        $outDtypeQ = $ffi->new('uint8_t');
        $outNdimQ = $ffi->new('size_t');
        $outShapeQ = $ffi->new("size_t[{$maxNdim}]");

        $outHandleR = $ffi->new('struct NdArrayHandle*');
        $outDtypeR = $ffi->new('uint8_t');
        $outNdimR = $ffi->new('size_t');
        $outShapeR = $ffi->new("size_t[{$maxNdim}]");

        $status = $ffi->ndarray_qr(
            $this->handle,
            Lib::addr($meta),
            Lib::addr($outHandleQ),
            Lib::addr($outDtypeQ),
            Lib::addr($outNdimQ),
            $outShapeQ,
            $maxNdim,
            Lib::addr($outHandleR),
            Lib::addr($outDtypeR),
            Lib::addr($outNdimR),
            $outShapeR,
        );

        Lib::checkStatus($status);

        $qShape = Lib::extractShapeFromPointer($outShapeQ, $outNdimQ->cdata);
        $rShape = Lib::extractShapeFromPointer($outShapeR, $outNdimR->cdata);

        $q = new NDArray($outHandleQ, new ArrayMetadata($qShape), $this->dtype);
        $r = new NDArray($outHandleR, new ArrayMetadata($rShape), $this->dtype);

        return [$q, $r];
    }

    /**
     * Compute Cholesky decomposition.
     *
     * For a Hermitian positive-definite matrix A:
     * - upper=false (default): returns L such that A = L * L^H
     * - upper=true: returns U such that A = U^H * U
     */
    public function cholesky(bool $upper = false): NDArray
    {
        return $this->unaryOp('ndarray_cholesky', $upper ? 1 : 0);
    }

    /**
     * Solve a least-squares problem min ||Ax - b||_2.
     *
     * Returns [x, residuals, rank, s] where:
     * - x is the least-squares solution
     * - residuals is the sum of residuals (or null if not applicable)
     * - rank is the effective rank of A
     * - s is the singular values of A
     *
     * @return array{0: NDArray, 1: null|NDArray, 2: int, 3: NDArray}
     */
    public function lstsq(NDArray $b): array
    {
        $ffi = Lib::get();
        $aMeta = $this->meta()->toCData();
        $bMeta = $b->meta()->toCData();
        $maxNdim = 8;

        $outSolution = $ffi->new('struct NdArrayHandle*');
        $outDtypeSol = $ffi->new('uint8_t');
        $outNdimSol = $ffi->new('size_t');
        $outShapeSol = $ffi->new("size_t[{$maxNdim}]");

        $outResiduals = $ffi->new('struct NdArrayHandle*');
        $outDtypeRes = $ffi->new('uint8_t');
        $outNdimRes = $ffi->new('size_t');
        $outShapeRes = $ffi->new("size_t[{$maxNdim}]");

        $outRank = $ffi->new('int32_t');

        $outS = $ffi->new('struct NdArrayHandle*');
        $outDtypeS = $ffi->new('uint8_t');
        $outNdimS = $ffi->new('size_t');
        $outShapeS = $ffi->new("size_t[{$maxNdim}]");

        $status = $ffi->ndarray_lstsq(
            $this->handle,
            Lib::addr($aMeta),
            $b->handle,
            Lib::addr($bMeta),
            Lib::addr($outSolution),
            Lib::addr($outResiduals),
            Lib::addr($outRank),
            Lib::addr($outS),
            Lib::addr($outDtypeSol),
            Lib::addr($outNdimSol),
            $outShapeSol,
            Lib::addr($outDtypeRes),
            Lib::addr($outNdimRes),
            $outShapeRes,
            Lib::addr($outDtypeS),
            Lib::addr($outNdimS),
            $outShapeS,
            $maxNdim,
        );

        Lib::checkStatus($status);

        $solShape = Lib::extractShapeFromPointer($outShapeSol, $outNdimSol->cdata);
        $solution = new NDArray($outSolution, new ArrayMetadata($solShape), $this->dtype);

        $residuals = null;
        if (!\FFI::isNull($outResiduals)) {
            $resShape = Lib::extractShapeFromPointer($outShapeRes, $outNdimRes->cdata);
            $residuals = new NDArray($outResiduals, new ArrayMetadata($resShape), $this->dtype);
        }

        $sShape = Lib::extractShapeFromPointer($outShapeS, $outNdimS->cdata);
        $s = new NDArray($outS, new ArrayMetadata($sShape), $this->dtype);

        return [$solution, $residuals, (int) $outRank->cdata, $s];
    }

    /**
     * Alias for lstsq().
     *
     * @return array{0: NDArray, 1: null|NDArray, 2: int, 3: NDArray}
     */
    public function leastSquares(NDArray $b): array
    {
        return $this->lstsq($b);
    }

    /**
     * Compute the Moore-Penrose pseudo-inverse of a matrix.
     *
     * @param null|float $rcond Cutoff for small singular values. Default uses machine precision.
     */
    public function pinv(?float $rcond = null): NDArray
    {
        $ffi = Lib::get();
        $meta = $this->meta()->toCData();
        $maxNdim = 8;

        $outHandle = $ffi->new('struct NdArrayHandle*');
        $outDtype = $ffi->new('uint8_t');
        $outNdim = $ffi->new('size_t');
        $outShape = $ffi->new("size_t[{$maxNdim}]");

        $rcondPtr = null !== $rcond ? $ffi->new('double') : null;
        if (null !== $rcondPtr) {
            $rcondPtr->cdata = $rcond;
        }

        $status = $ffi->ndarray_pinv(
            $this->handle,
            Lib::addr($meta),
            null !== $rcondPtr ? Lib::addr($rcondPtr) : null,
            Lib::addr($outHandle),
            Lib::addr($outDtype),
            Lib::addr($outNdim),
            $outShape,
            $maxNdim,
        );

        Lib::checkStatus($status);

        $shape = Lib::extractShapeFromPointer($outShape, $outNdim->cdata);

        return new NDArray($outHandle, new ArrayMetadata($shape), $this->dtype);
    }

    /**
     * Compute the 2-norm condition number of a matrix.
     */
    public function cond(): float
    {
        return $this->scalarReductionOp('ndarray_cond');
    }

    /**
     * Compute the rank of a matrix using SVD.
     *
     * @param null|float $tol threshold below which SVD values are considered zero
     */
    public function rank(?float $tol = null): int
    {
        $ffi = Lib::get();
        $meta = $this->meta()->toCData();

        $outRank = $ffi->new('int32_t');
        $tolPtr = null !== $tol ? $ffi->new('double') : null;
        if (null !== $tolPtr) {
            $tolPtr->cdata = $tol;
        }

        $status = $ffi->ndarray_rank(
            $this->handle,
            Lib::addr($meta),
            null !== $tolPtr ? Lib::addr($tolPtr) : null,
            Lib::addr($outRank),
        );

        Lib::checkStatus($status);

        return (int) $outRank->cdata;
    }

    /**
     * Normalize norm order into FFI code.
     */
    private function normalizeNormOrder(float|int|string|null $ord, ?int $axis): int
    {
        if (null === $ord) {
            if (null === $axis && 2 === $this->ndim()) {
                return 5; // fro
            }

            return 2;
        }

        if (\is_int($ord)) {
            return match ($ord) {
                1 => 1,
                2 => 2,
                default => throw new \InvalidArgumentException("Unsupported norm order: {$ord}"),
            };
        }

        if (\is_float($ord)) {
            if (\INF === $ord) {
                return 3;
            }
            if ($ord === -\INF) {
                return 4;
            }
            if (1.0 === $ord) {
                return 1;
            }
            if (2.0 === $ord) {
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
        $shape = $this->shape();
        $ndim = \count($shape);
        $axisNorm = $axis < 0 ? $ndim + $axis : $axis;

        if ($keepdims) {
            $shape[$axisNorm] = 1;

            return $shape;
        }

        $out = array_values(array_filter(
            $shape,
            static fn (int $idx): bool => $idx !== $axisNorm,
            \ARRAY_FILTER_USE_KEY
        ));

        return $out;
    }
}
