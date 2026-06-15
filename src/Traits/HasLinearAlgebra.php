<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Traits;

use PhpMlKit\NDArray\ArrayMetadata;
use PhpMlKit\NDArray\Complex;
use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\Exceptions\NDArrayException;
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
     *
     * @return ($axis is null ? float : NDArray)
     */
    public function norm(float|int|string|null $ord = null, ?int $axis = null, bool $keepdims = false): float|NDArray
    {
        $lib = Lib::get();
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
            $outValue = $lib->new('double');
            $outDtype = $lib->new('uint8_t');
            $status = $lib->ndarray_norm(
                $this->handle,
                Lib::addr($meta),
                $ordCode,
                \FFI::addr($outValue),
                Lib::addr($outDtype)
            );

            $lib->checkStatus($status);

            return (float) $outValue->cdata;
        }

        $outHandle = $lib->new('struct NdArrayHandle*');

        $status = $lib->ndarray_norm_axis(
            $this->handle,
            Lib::addr($meta),
            $axis,
            $keepdims,
            $ordCode,
            Lib::addr($outHandle)
        );

        $lib->checkStatus($status);
        $outShape = $this->computeNormOutputShape($axis, $keepdims);

        return new NDArray($outHandle, new ArrayMetadata($outShape), DType::Float64);
    }

    /**
     * Generalized dot product for 1D and 2D operands.
     *
     * Operand dtypes are promoted to a common type before the operation. Shape rules:
     * - **1D × 1D**: inner product → scalar
     * - **2D × 2D**: matrix product → 2D array
     * - **1D × 2D** or **2D × 1D**: vector–matrix product → 1D array
     *
     * @param NDArray $other The other array
     *
     * @return Complex|float|int|NDArray scalar when the result is 0-D, otherwise an NDArray
     */
    public function dot(NDArray $other): Complex|float|int|NDArray
    {
        $result = $this->binaryOp('ndarray_dot', $other);

        return 0 === $result->ndim() ? $result->toScalar() : $result;
    }

    /**
     * Matrix multiplication (`@`) for 1D and 2D operands.
     *
     * Operand dtypes are promoted to a common type before the operation. Supported shapes:
     * - **2D × 2D**: matrix × matrix → 2D array
     * - **2D × 1D** or **1D × 2D**: matrix × vector → 1D array
     * - **1D × 1D**: inner product → scalar (0-D result unpacked to a PHP scalar or `Complex`)
     *
     * Operands with more than two dimensions are not supported.
     *
     * @param NDArray $other The other array
     *
     * @return Complex|float|int|NDArray scalar when the result is 0-D, otherwise an NDArray
     */
    public function matmul(NDArray $other): Complex|float|int|NDArray
    {
        $result = $this->binaryOp('ndarray_matmul', $other);

        return 0 === $result->ndim() ? $result->toScalar() : $result;
    }

    /**
     * Einstein summation with deterministic accumulation order.
     *
     * Evaluates the subscript expression using a fixed loop order.
     * Unlike matmul() which delegates to BLAS (tiled, non-deterministic accumulation),
     * einsum uses plain nested loops producing identical results on every call.
     *
     * Supports both two-operand patterns (`ij,jk->ik`, `i,i->`) and
     * single-operand patterns (`ii->` for trace, `ij->ji` for transpose,
     * `ij->i` for sum, etc.). When `->` is omitted, output labels are
     * inferred from labels appearing exactly once across all operands.
     *
     * @param string       $subscripts Einstein summation subscript
     * @param null|NDArray $other      The second operand (null for single-operand patterns)
     *
     * @return NDArray Result of the contraction
     */
    public function einsum(string $subscripts, ?NDArray $other = null): NDArray
    {
        $lib = Lib::get();
        $aMeta = $this->meta()->toCData();

        $outHandle = $lib->new('struct NdArrayHandle*');
        $outDtypeBuf = $lib->new('uint8_t');
        $outNdimBuf = $lib->new('size_t');
        $outShapeBuf = $lib->new(\sprintf('size_t[%d]', Lib::MAX_NDIM));

        $subscriptsBytes = $subscripts."\0";
        $subscriptsPtr = $lib->new('char['.\strlen($subscriptsBytes).']');
        \FFI::memcpy($subscriptsPtr, $subscriptsBytes, \strlen($subscriptsBytes));

        $bMetaCData = null !== $other ? $other->meta()->toCData() : null;

        $status = $lib->ndarray_einsum(
            $this->handle,
            Lib::addr($aMeta),
            null !== $other ? $other->handle : null,
            null !== $bMetaCData ? Lib::addr($bMetaCData) : null,
            $subscriptsPtr,
            Lib::addr($outHandle),
            Lib::addr($outDtypeBuf),
            Lib::addr($outNdimBuf),
            $outShapeBuf,
            Lib::MAX_NDIM,
        );

        $lib->checkStatus($status);

        $dtype = DType::tryFrom((int) $outDtypeBuf->cdata);
        if (null === $dtype) {
            throw new NDArrayException('Invalid dtype returned from einsum');
        }

        $ndim = (int) $outNdimBuf->cdata;
        $outShape = $lib->readSizeTArray($outShapeBuf, $ndim);

        return new NDArray($outHandle, new ArrayMetadata($outShape), $dtype);
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
     * @return Complex|float|int scalar value
     */
    public function trace(): Complex|float|int
    {
        return $this->unaryOp('ndarray_trace')->toScalar();
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
    public function det(): Complex|float|int
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
        $lib = Lib::get();
        $meta = $this->meta()->toCData();
        $calcUv = $computeUv ? 1 : 0;
        $maxNdim = 8;

        $outHandleS = $lib->new('struct NdArrayHandle*');
        $outDtypeS = $lib->new('uint8_t');
        $outNdimS = $lib->new('size_t');
        $outShapeS = $lib->new("size_t[{$maxNdim}]");

        $outHandleU = $computeUv ? $lib->new('struct NdArrayHandle*') : null;
        $outDtypeU = $computeUv ? $lib->new('uint8_t') : null;
        $outNdimU = $computeUv ? $lib->new('size_t') : null;
        $outShapeU = $computeUv ? $lib->new("size_t[{$maxNdim}]") : null;

        $outHandleVt = $computeUv ? $lib->new('struct NdArrayHandle*') : null;
        $outDtypeVt = $computeUv ? $lib->new('uint8_t') : null;
        $outNdimVt = $computeUv ? $lib->new('size_t') : null;
        $outShapeVt = $computeUv ? $lib->new("size_t[{$maxNdim}]") : null;

        $status = $lib->ndarray_svd(
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

        $lib->checkStatus($status);

        $sShape = $lib->readSizeTArray($outShapeS, $outNdimS->cdata);
        $sDtype = DType::from($outDtypeS->cdata);
        $s = new NDArray($outHandleS, new ArrayMetadata($sShape), $sDtype);

        if (!$computeUv) {
            return $s;
        }

        $uShape = $lib->readSizeTArray($outShapeU, $outNdimU->cdata);
        $vtShape = $lib->readSizeTArray($outShapeVt, $outNdimVt->cdata);

        $uDtype = DType::from($outDtypeU->cdata);
        $vtDtype = DType::from($outDtypeVt->cdata);

        $u = new NDArray($outHandleU, new ArrayMetadata($uShape), $uDtype);
        $vt = new NDArray($outHandleVt, new ArrayMetadata($vtShape), $vtDtype);

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
        $lib = Lib::get();
        $meta = $this->meta()->toCData();
        $maxNdim = 8;

        $outHandleQ = $lib->new('struct NdArrayHandle*');
        $outDtypeQ = $lib->new('uint8_t');
        $outNdimQ = $lib->new('size_t');
        $outShapeQ = $lib->new("size_t[{$maxNdim}]");

        $outHandleR = $lib->new('struct NdArrayHandle*');
        $outDtypeR = $lib->new('uint8_t');
        $outNdimR = $lib->new('size_t');
        $outShapeR = $lib->new("size_t[{$maxNdim}]");

        $status = $lib->ndarray_qr(
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

        $lib->checkStatus($status);

        $qShape = $lib->readSizeTArray($outShapeQ, $outNdimQ->cdata);
        $rShape = $lib->readSizeTArray($outShapeR, $outNdimR->cdata);

        $qDtype = DType::from($outDtypeQ->cdata);
        $rDtype = DType::from($outDtypeR->cdata);

        $q = new NDArray($outHandleQ, new ArrayMetadata($qShape), $qDtype);
        $r = new NDArray($outHandleR, new ArrayMetadata($rShape), $rDtype);

        return [$q, $r];
    }

    /**
     * Compute eigenvalue decomposition.
     *
     * Decomposes square matrix A into eigenvalues and right eigenvectors:
     * A * v_i = lambda_i * v_i
     *
     * For real input matrices, eigenvalues and eigenvectors are complex.
     * For complex input matrices, output type matches input type.
     *
     * @return array{0: NDArray, 1: NDArray} [eigenvalues, eigenvectors]
     */
    public function eig(): array
    {
        $lib = Lib::get();
        $meta = $this->meta()->toCData();
        $maxNdim = 8;

        $outHandleEigvals = $lib->new('struct NdArrayHandle*');
        $outDtypeEigvals = $lib->new('uint8_t');
        $outNdimEigvals = $lib->new('size_t');
        $outShapeEigvals = $lib->new("size_t[{$maxNdim}]");

        $outHandleEigvecs = $lib->new('struct NdArrayHandle*');
        $outDtypeEigvecs = $lib->new('uint8_t');
        $outNdimEigvecs = $lib->new('size_t');
        $outShapeEigvecs = $lib->new("size_t[{$maxNdim}]");

        $status = $lib->ndarray_eig(
            $this->handle,
            Lib::addr($meta),
            Lib::addr($outHandleEigvals),
            Lib::addr($outDtypeEigvals),
            Lib::addr($outNdimEigvals),
            $outShapeEigvals,
            $maxNdim,
            Lib::addr($outHandleEigvecs),
            Lib::addr($outDtypeEigvecs),
            Lib::addr($outNdimEigvecs),
            $outShapeEigvecs,
        );

        $lib->checkStatus($status);

        $eigvalsShape = $lib->readSizeTArray($outShapeEigvals, $outNdimEigvals->cdata);
        $eigvecsShape = $lib->readSizeTArray($outShapeEigvecs, $outNdimEigvecs->cdata);

        $eigvalsDtype = DType::from($outDtypeEigvals->cdata);
        $eigvecsDtype = DType::from($outDtypeEigvecs->cdata);

        $eigvals = new NDArray($outHandleEigvals, new ArrayMetadata($eigvalsShape), $eigvalsDtype);
        $eigvecs = new NDArray($outHandleEigvecs, new ArrayMetadata($eigvecsShape), $eigvecsDtype);

        return [$eigvals, $eigvecs];
    }

    /**
     * Compute eigenvalues only (no eigenvectors) for a general matrix.
     *
     * For real input matrices, eigenvalues are complex.
     * For complex input matrices, output type matches input type.
     */
    public function eigvals(): NDArray
    {
        $lib = Lib::get();
        $meta = $this->meta()->toCData();
        $maxNdim = 8;

        $outHandleEigvals = $lib->new('struct NdArrayHandle*');
        $outDtypeEigvals = $lib->new('uint8_t');
        $outNdimEigvals = $lib->new('size_t');
        $outShapeEigvals = $lib->new("size_t[{$maxNdim}]");

        $status = $lib->ndarray_eigvals(
            $this->handle,
            Lib::addr($meta),
            Lib::addr($outHandleEigvals),
            Lib::addr($outDtypeEigvals),
            Lib::addr($outNdimEigvals),
            $outShapeEigvals,
            $maxNdim,
        );

        $lib->checkStatus($status);

        $eigvalsShape = $lib->readSizeTArray($outShapeEigvals, $outNdimEigvals->cdata);
        $eigvalsDtype = DType::from($outDtypeEigvals->cdata);

        return new NDArray($outHandleEigvals, new ArrayMetadata($eigvalsShape), $eigvalsDtype);
    }

    /**
     * Compute eigenvalue decomposition for a Hermitian/symmetric matrix.
     *
     * Eigenvalues are always real. Eigenvectors have the same type as input.
     *
     * @param bool $upper If true, use upper triangular part. If false, use lower.
     *
     * @return array{0: NDArray, 1: NDArray} [eigenvalues, eigenvectors]
     */
    public function eigh(bool $upper = false): array
    {
        $lib = Lib::get();
        $meta = $this->meta()->toCData();
        $maxNdim = 8;
        $uplo = $upper ? 1 : 0;

        $outHandleEigvals = $lib->new('struct NdArrayHandle*');
        $outDtypeEigvals = $lib->new('uint8_t');
        $outNdimEigvals = $lib->new('size_t');
        $outShapeEigvals = $lib->new("size_t[{$maxNdim}]");

        $outHandleEigvecs = $lib->new('struct NdArrayHandle*');
        $outDtypeEigvecs = $lib->new('uint8_t');
        $outNdimEigvecs = $lib->new('size_t');
        $outShapeEigvecs = $lib->new("size_t[{$maxNdim}]");

        $status = $lib->ndarray_eigh(
            $this->handle,
            Lib::addr($meta),
            $uplo,
            Lib::addr($outHandleEigvals),
            Lib::addr($outDtypeEigvals),
            Lib::addr($outNdimEigvals),
            $outShapeEigvals,
            $maxNdim,
            Lib::addr($outHandleEigvecs),
            Lib::addr($outDtypeEigvecs),
            Lib::addr($outNdimEigvecs),
            $outShapeEigvecs,
        );

        $lib->checkStatus($status);

        $eigvalsShape = $lib->readSizeTArray($outShapeEigvals, $outNdimEigvals->cdata);
        $eigvecsShape = $lib->readSizeTArray($outShapeEigvecs, $outNdimEigvecs->cdata);

        $eigvalsDtype = DType::from($outDtypeEigvals->cdata);
        $eigvecsDtype = DType::from($outDtypeEigvecs->cdata);

        $eigvals = new NDArray($outHandleEigvals, new ArrayMetadata($eigvalsShape), $eigvalsDtype);
        $eigvecs = new NDArray($outHandleEigvecs, new ArrayMetadata($eigvecsShape), $eigvecsDtype);

        return [$eigvals, $eigvecs];
    }

    /**
     * Compute eigenvalues only (no eigenvectors) for a Hermitian/symmetric matrix.
     *
     * Eigenvalues are always real.
     *
     * @param bool $upper If true, use upper triangular part. If false, use lower.
     */
    public function eigvalsh(bool $upper = false): NDArray
    {
        $lib = Lib::get();
        $meta = $this->meta()->toCData();
        $maxNdim = 8;
        $uplo = $upper ? 1 : 0;

        $outHandleEigvals = $lib->new('struct NdArrayHandle*');
        $outDtypeEigvals = $lib->new('uint8_t');
        $outNdimEigvals = $lib->new('size_t');
        $outShapeEigvals = $lib->new("size_t[{$maxNdim}]");

        $status = $lib->ndarray_eigvalsh(
            $this->handle,
            Lib::addr($meta),
            $uplo,
            Lib::addr($outHandleEigvals),
            Lib::addr($outDtypeEigvals),
            Lib::addr($outNdimEigvals),
            $outShapeEigvals,
            $maxNdim,
        );

        $lib->checkStatus($status);

        $eigvalsShape = $lib->readSizeTArray($outShapeEigvals, $outNdimEigvals->cdata);
        $eigvalsDtype = DType::from($outDtypeEigvals->cdata);

        return new NDArray($outHandleEigvals, new ArrayMetadata($eigvalsShape), $eigvalsDtype);
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
        $lib = Lib::get();
        $aMeta = $this->meta()->toCData();
        $bMeta = $b->meta()->toCData();
        $maxNdim = 8;

        $outSolution = $lib->new('struct NdArrayHandle*');
        $outDtypeSol = $lib->new('uint8_t');
        $outNdimSol = $lib->new('size_t');
        $outShapeSol = $lib->new("size_t[{$maxNdim}]");

        $outResiduals = $lib->new('struct NdArrayHandle*');
        $outDtypeRes = $lib->new('uint8_t');
        $outNdimRes = $lib->new('size_t');
        $outShapeRes = $lib->new("size_t[{$maxNdim}]");

        $outRank = $lib->new('int32_t');

        $outS = $lib->new('struct NdArrayHandle*');
        $outDtypeS = $lib->new('uint8_t');
        $outNdimS = $lib->new('size_t');
        $outShapeS = $lib->new("size_t[{$maxNdim}]");

        $status = $lib->ndarray_lstsq(
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

        $lib->checkStatus($status);

        $solShape = $lib->readSizeTArray($outShapeSol, $outNdimSol->cdata);
        $solutionDtype = DType::from($outDtypeSol->cdata);
        $solution = new NDArray($outSolution, new ArrayMetadata($solShape), $solutionDtype);

        $residuals = null;
        if (!\FFI::isNull($outResiduals)) {
            $resShape = $lib->readSizeTArray($outShapeRes, $outNdimRes->cdata);
            $residualsDtype = DType::from($outDtypeRes->cdata);
            $residuals = new NDArray($outResiduals, new ArrayMetadata($resShape), $residualsDtype);
        }

        $sShape = $lib->readSizeTArray($outShapeS, $outNdimS->cdata);
        $sDtype = DType::from($outDtypeS->cdata);
        $s = new NDArray($outS, new ArrayMetadata($sShape), $sDtype);

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
        $lib = Lib::get();
        $meta = $this->meta()->toCData();
        $maxNdim = 8;

        $outHandle = $lib->new('struct NdArrayHandle*');
        $outDtype = $lib->new('uint8_t');
        $outNdim = $lib->new('size_t');
        $outShape = $lib->new("size_t[{$maxNdim}]");

        $rcondPtr = null !== $rcond ? $lib->new('double') : null;
        if (null !== $rcondPtr) {
            $rcondPtr->cdata = $rcond;
        }

        $status = $lib->ndarray_pinv(
            $this->handle,
            Lib::addr($meta),
            null !== $rcondPtr ? Lib::addr($rcondPtr) : null,
            Lib::addr($outHandle),
            Lib::addr($outDtype),
            Lib::addr($outNdim),
            $outShape,
            $maxNdim,
        );

        $lib->checkStatus($status);

        $shape = $lib->readSizeTArray($outShape, $outNdim->cdata);
        $dtype = DType::from($outDtype->cdata);

        return new NDArray($outHandle, new ArrayMetadata($shape), $dtype);
    }

    /**
     * Compute the 2-norm condition number of a matrix.
     */
    public function cond(): Complex|float|int
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
        $lib = Lib::get();
        $meta = $this->meta()->toCData();

        $outRank = $lib->new('int32_t');
        $tolPtr = null !== $tol ? $lib->new('double') : null;
        if (null !== $tolPtr) {
            $tolPtr->cdata = $tol;
        }

        $status = $lib->ndarray_rank(
            $this->handle,
            Lib::addr($meta),
            null !== $tolPtr ? Lib::addr($tolPtr) : null,
            Lib::addr($outRank),
        );

        $lib->checkStatus($status);

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
