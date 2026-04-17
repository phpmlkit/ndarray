<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Unit;

use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\Exceptions\MathException;
use PhpMlKit\NDArray\Exceptions\ShapeException;
use PhpMlKit\NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Tests for linear algebra operations.
 *
 * @internal
 *
 * @coversNothing
 */
class LinearAlgebraTest extends TestCase
{
    public function testDot1D(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $b = NDArray::array([4, 5, 6], DType::Float64);
        $result = $a->dot($b);

        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        $this->assertEqualsWithDelta(32, $result->toArray(), 0.0001);
    }

    public function testDot2D(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);
        $b = NDArray::array([[5, 6], [7, 8]], DType::Float64);
        $result = $a->dot($b);

        // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        // = [[19, 22], [43, 50]]
        $this->assertEqualsWithDelta([[19, 22], [43, 50]], $result->toArray(), 0.0001);
    }

    public function testDot2D1D(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);
        $b = NDArray::array([5, 6], DType::Float64);
        $result = $a->dot($b);

        // [1*5+2*6, 3*5+4*6] = [17, 39]
        $this->assertEqualsWithDelta([17, 39], $result->toArray(), 0.0001);
    }

    public function testDot1D2D(): void
    {
        $a = NDArray::array([1, 2], DType::Float64);
        $b = NDArray::array([[5, 6], [7, 8]], DType::Float64);
        $result = $a->dot($b);

        // [1*5+2*7, 1*6+2*8] = [19, 22]
        $this->assertEqualsWithDelta([19, 22], $result->toArray(), 0.0001);
    }

    public function testDotShapeMismatch(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $b = NDArray::array([4, 5], DType::Float64);

        $this->expectException(ShapeException::class);
        $a->dot($b);
    }

    public function testMatmul2D(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);
        $b = NDArray::array([[5, 6], [7, 8]], DType::Float64);
        $result = $a->matmul($b);

        $this->assertEqualsWithDelta([[19, 22], [43, 50]], $result->toArray(), 0.0001);
    }

    public function testMatmulRequires2D(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $b = NDArray::array([[1, 2], [3, 4]], DType::Float64);

        $this->expectException(ShapeException::class);
        $a->matmul($b);
    }

    public function testDiagonal(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], DType::Float64);
        $result = $a->diagonal();

        $this->assertEquals([1, 5, 9], $result->toArray());
    }

    public function testDiagonalNonSquare(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->diagonal();

        // min(2, 3) = 2 elements
        $this->assertEquals([1, 5], $result->toArray());
    }

    public function testDiagonalRequires2D(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);

        $this->expectException(ShapeException::class);
        $a->diagonal();
    }

    public function testTrace(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], DType::Float64);
        $result = $a->trace();

        // 1 + 5 + 9 = 15
        $this->assertEqualsWithDelta(15, $result->toArray(), 0.0001);
    }

    public function testTraceNonSquare(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->trace();

        // 1 + 5 = 6
        $this->assertEqualsWithDelta(6, $result->toArray(), 0.0001);
    }

    public function testTraceRequires2D(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);

        $this->expectException(ShapeException::class);
        $a->trace();
    }

    public function testNormScalarL2(): void
    {
        $a = NDArray::array([3, 4], DType::Float64);
        $this->assertEqualsWithDelta(5.0, $a->norm(2), 1e-10);
    }

    public function testNormScalarL1AndInf(): void
    {
        $a = NDArray::array([-3, 4, -5], DType::Float64);
        $this->assertEqualsWithDelta(12.0, $a->norm(1), 1e-10);
        $this->assertEqualsWithDelta(5.0, $a->norm(\INF), 1e-10);
        $this->assertEqualsWithDelta(3.0, $a->norm(-\INF), 1e-10);
    }

    public function testNormAxisL2(): void
    {
        $a = NDArray::array([[3, 4], [5, 12]], DType::Float64);
        $result = $a->norm(2, axis: 1);
        $this->assertEqualsWithDelta([5.0, 13.0], $result->toArray(), 1e-10);
    }

    public function testNormAxisKeepdims(): void
    {
        $a = NDArray::array([[1, -2, 3], [4, -5, 6]], DType::Float64);
        $result = $a->norm(1, axis: 1, keepdims: true);
        $this->assertSame([2, 1], $result->shape());
        $this->assertEqualsWithDelta([[6.0], [15.0]], $result->toArray(), 1e-10);
    }

    public function testNormFrobeniusDefaultForMatrix(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);
        $expected = sqrt(1 + 4 + 9 + 16);
        $this->assertEqualsWithDelta($expected, $a->norm(), 1e-10);
        $this->assertEqualsWithDelta($expected, $a->norm('fro'), 1e-10);
    }

    public function testNormFroRequire2D(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $this->expectException(\InvalidArgumentException::class);
        $a->norm('fro');
    }

    // VIEW/SUBSET TESTS (from LinearAlgebraViewTest.php)

    public function testDotOnView(): void
    {
        // Create a 3x3 matrix and take a 2x2 slice
        $a = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Float64);

        // Get a 2x2 view: rows 0-1, cols 0-1 = [[1, 2], [4, 5]]
        $aView = $a->slice(['0:2', '0:2']);

        $b = NDArray::array([[1, 0], [0, 1]], DType::Float64);
        $result = $aView->dot($b);

        // [[1, 2], [4, 5]] @ [[1, 0], [0, 1]] = [[1, 2], [4, 5]]
        $this->assertEqualsWithDelta([[1, 2], [4, 5]], $result->toArray(), 0.0001);
    }

    public function testDotOnSlicedRow(): void
    {
        // Create a 3x3 matrix
        $a = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Float64);

        // Get row 1 as a view using slice with single row
        $aView = $a->slice(['1:2', ':']);
        $this->assertSame([1, 3], $aView->shape());

        $b = NDArray::array([1, 1, 1], DType::Float64);

        $result = $aView->dot($b);

        // 4 + 5 + 6 = 15 (result is 1D array with one element)
        $this->assertEqualsWithDelta([15.0], $result->toArray(), 0.0001);
    }

    public function testMatmulOnView(): void
    {
        // Create a 4x4 matrix and take a 2x2 slice
        $a = NDArray::array([
            [1, 2, 0, 0],
            [3, 4, 0, 0],
            [0, 0, 5, 6],
            [0, 0, 7, 8],
        ], DType::Float64);

        // Get top-left 2x2 view
        $aView = $a->slice(['0:2', '0:2']);

        $b = NDArray::array([[1, 0], [0, 1]], DType::Float64);
        $result = $aView->matmul($b);

        $this->assertEqualsWithDelta([[1, 2], [3, 4]], $result->toArray(), 0.0001);
    }

    public function testDiagonalOnView(): void
    {
        // Create a 4x4 matrix
        $a = NDArray::array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ], DType::Float64);

        // Get top-left 3x3 view
        $aView = $a->slice(['0:3', '0:3']);
        $result = $aView->diagonal();

        // Diagonal of [[1, 2, 3], [5, 6, 7], [9, 10, 11]] is [1, 6, 11]
        $this->assertEquals([1, 6, 11], $result->toArray());
    }

    public function testDiagonalOnNonSquareView(): void
    {
        $a = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Float64);

        // Get 2x3 view: rows 0-1, all cols
        $aView = $a->slice(['0:2', ':']);
        $result = $aView->diagonal();

        // Diagonal of [[1, 2, 3], [4, 5, 6]] is [1, 5]
        $this->assertEquals([1, 5], $result->toArray());
    }

    public function testTraceOnView(): void
    {
        $a = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Float64);

        // Get top-left 2x2 view
        $aView = $a->slice(['0:2', '0:2']);
        $result = $aView->trace();

        // Trace of [[1, 2], [4, 5]] is 1 + 5 = 6
        $this->assertEqualsWithDelta(6, $result->toArray(), 0.0001);
    }

    public function testTraceOnFullMatrixView(): void
    {
        $a = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Float64);

        // Get full matrix as a view
        $aView = $a->slice([':', ':']);
        $result = $aView->trace();

        // Trace of full 3x3 is 1 + 5 + 9 = 15
        $this->assertEqualsWithDelta(15, $result->toArray(), 0.0001);
    }

    public function testDotBothViews(): void
    {
        $a = NDArray::array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ], DType::Float64);

        $b = NDArray::array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], DType::Float64);

        // Get top-left 2x2 views of both
        $aView = $a->slice(['0:2', '0:2']);
        $bView = $b->slice(['0:2', '0:2']);

        $result = $aView->dot($bView);

        // [[1, 2], [5, 6]] @ [[1, 0], [0, 1]] = [[1, 2], [5, 6]]
        $this->assertEqualsWithDelta([[1, 2], [5, 6]], $result->toArray(), 0.0001);
    }

    // =========================================================================
    // SVD Tests
    // =========================================================================

    public function testSvdDecomposition(): void
    {
        $a = NDArray::array([
            [3, 0],
            [0, 2],
        ], DType::Float64);

        [$u, $s, $vt] = $a->svd();

        // U should be 2x2 orthogonal
        $this->assertSame([2, 2], $u->shape());

        // S should be 1D array with 2 elements (min(2, 2) = 2)
        $this->assertSame([2], $s->shape());

        // VT should be 2x2 orthogonal
        $this->assertSame([2, 2], $vt->shape());

        // For diagonal matrix [[3, 0], [0, 2]], singular values are 3 and 2
        $singularValues = $s->toArray();
        $this->assertEqualsWithDelta([3.0, 2.0], $singularValues, 1e-10);
    }

    public function testSvdReconstruction(): void
    {
        $a = NDArray::array([
            [4, 0],
            [3, -5],
        ], DType::Float64);

        [$u, $s, $vt] = $a->svd();

        // Reconstruct: A = U * diag(S) * VT
        $reconstructed = $u->matmul($s->diag())->matmul($vt);

        $this->assertEqualsWithDelta(
            [[4.0, 0.0], [3.0, -5.0]],
            $reconstructed->toArray(),
            1e-9
        );
    }

    public function testSvdValuesOnly(): void
    {
        $a = NDArray::array([
            [3, 0],
            [0, 2],
        ], DType::Float64);

        $s = $a->svd(computeUv: false);

        $this->assertInstanceOf(NDArray::class, $s);
        $this->assertSame([2], $s->shape());

        $this->assertEqualsWithDelta([3.0, 2.0], $s->toArray(), 1e-10);
    }

    public function testSvdRequires2D(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);

        $this->expectException(ShapeException::class);
        $this->expectExceptionMessage('SVD requires 2D matrix');
        $a->svd();
    }

    public function testSvdNonSquareMatrix(): void
    {
        // 3x2 matrix
        $a = NDArray::array([
            [1, 2],
            [3, 4],
            [5, 6],
        ], DType::Float64);

        [$u, $s, $vt] = $a->svd();

        // Verify shapes
        $this->assertSame([3, 3], $u->shape());
        $this->assertSame([2], $s->shape());
        $this->assertSame([2, 2], $vt->shape());

        // Reconstruct and verify: U * S * VT where S is diagonal padded to 3x2
        $sDiag = NDArray::array([
            [$s[0], 0.0],
            [0.0, $s[1]],
            [0.0, 0.0],
        ], DType::Float64);

        $reconstructed = $u->matmul($sDiag)->matmul($vt);

        $this->assertEqualsWithDelta(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            $reconstructed->toArray(),
            1e-9
        );

        // Known singular values for [[1,2],[3,4],[5,6]] from LAPACK
        $singularValues = $s->toArray();
        $this->assertEqualsWithDelta(9.525518, $singularValues[0], 1e-6);
        $this->assertEqualsWithDelta(0.514301, $singularValues[1], 1e-6);
    }

    public function testSvdOrthogonality(): void
    {
        // Test that U and VT are orthogonal (U^T * U = I, VT * VT^T = I)
        $a = NDArray::array([
            [4, 0],
            [3, -5],
        ], DType::Float64);

        [$u, $s, $vt] = $a->svd();

        // U^T * U should be identity
        $uT = $u->transpose();
        $identityU = $uT->matmul($u);
        $this->assertEqualsWithDelta(
            [[1.0, 0.0], [0.0, 1.0]],
            $identityU->toArray(),
            1e-10
        );

        // VT * VT^T should be identity
        $vtT = $vt->transpose();
        $identityVT = $vt->matmul($vtT);
        $this->assertEqualsWithDelta(
            [[1.0, 0.0], [0.0, 1.0]],
            $identityVT->toArray(),
            1e-10
        );
    }

    // =========================================================================
    // Diag Tests
    // =========================================================================

    public function testDiagFrom1D(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $result = $a->diag();

        $this->assertSame([3, 3], $result->shape());
        $this->assertEqualsWithDelta([
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 3],
        ], $result->toArray(), 0.0001);
    }

    public function testDiagFrom2D(): void
    {
        $a = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Float64);
        $result = $a->diag();

        $this->assertSame([3], $result->shape());
        $this->assertEquals([1, 5, 9], $result->toArray());
    }

    public function testDiagRequiresValidDimensions(): void
    {
        $a = NDArray::array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ], DType::Float64);

        $this->expectException(ShapeException::class);
        $this->expectExceptionMessage('Diagonal requires a 2D array');
        $a->diag();
    }

    public function testDiagFrom1DWithPositiveOffset(): void
    {
        $a = NDArray::array([1, 2], DType::Float64);
        $result = $a->diag(offset: 1);

        $this->assertSame([3, 3], $result->shape());
        $this->assertEqualsWithDelta([
            [0, 1, 0],
            [0, 0, 2],
            [0, 0, 0],
        ], $result->toArray(), 0.0001);
    }

    public function testDiagFrom1DWithNegativeOffset(): void
    {
        $a = NDArray::array([1, 2], DType::Float64);
        $result = $a->diag(offset: -1);

        $this->assertSame([3, 3], $result->shape());
        $this->assertEqualsWithDelta([
            [0, 0, 0],
            [1, 0, 0],
            [0, 2, 0],
        ], $result->toArray(), 0.0001);
    }

    public function testDiagFrom2DWithPositiveOffset(): void
    {
        $a = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Float64);
        $result = $a->diag(offset: 1);

        $this->assertSame([2], $result->shape());
        $this->assertEquals([2, 6], $result->toArray());
    }

    public function testDiagFrom2DWithNegativeOffset(): void
    {
        $a = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Float64);
        $result = $a->diag(offset: -1);

        $this->assertSame([2], $result->shape());
        $this->assertEquals([4, 8], $result->toArray());
    }

    // =========================================================================
    // Solve Tests
    // =========================================================================

    public function testSolve1D(): void
    {
        $a = NDArray::array([
            [3., 2., -1.],
            [2., -2., 4.],
            [-2., 1., -2.],
        ], DType::Float64);
        $b = NDArray::array([1., -2., 0.], DType::Float64);

        $x = $a->solve($b);

        $this->assertSame([3], $x->shape());
        $this->assertEqualsWithDelta([1., -2., -2.], $x->toArray(), 1e-9);
    }

    public function testSolve2D(): void
    {
        $a = NDArray::array([
            [3., 2., -1.],
            [2., -2., 4.],
            [-2., 1., -2.],
        ], DType::Float64);
        $b = NDArray::array([
            [1., 0.],
            [-2., 1.],
            [0., 2.],
        ], DType::Float64);

        $x = $a->solve($b);

        $this->assertSame([3, 2], $x->shape());
        // Verify A * x ≈ b by reconstruction
        $reconstructed = $a->matmul($x);
        $this->assertEqualsWithDelta($b->toArray(), $reconstructed->toArray(), 1e-9);
    }

    public function testSolveRequires2DA(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $b = NDArray::array([1, 2, 3], DType::Float64);

        $this->expectException(ShapeException::class);
        $a->solve($b);
    }

    public function testSolveRequires1DOr2DB(): void
    {
        $a = NDArray::array([
            [1, 2],
            [3, 4],
        ], DType::Float64);
        $b = NDArray::array([[[1, 2], [3, 4]]], DType::Float64);

        $this->expectException(ShapeException::class);
        $a->solve($b);
    }

    // =========================================================================
    // Inverse Tests
    // =========================================================================

    public function testInv(): void
    {
        $a = NDArray::array([
            [4, 7],
            [2, 6],
        ], DType::Float64);

        $inv = $a->inv();

        $this->assertSame([2, 2], $inv->shape());
        // Expected inverse: [[0.6, -0.7], [-0.2, 0.4]]
        $this->assertEqualsWithDelta([
            [0.6, -0.7],
            [-0.2, 0.4],
        ], $inv->toArray(), 1e-10);

        // Verify A * A^-1 = I
        $identity = $a->matmul($inv);
        $this->assertEqualsWithDelta([
            [1., 0.],
            [0., 1.],
        ], $identity->toArray(), 1e-9);
    }

    public function testInvRequiresSquare(): void
    {
        $a = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
        ], DType::Float64);

        $this->expectException(ShapeException::class);
        $a->inv();
    }

    // =========================================================================
    // Determinant Tests
    // =========================================================================

    public function testDet(): void
    {
        $a = NDArray::array([
            [4, 7],
            [2, 6],
        ], DType::Float64);

        $det = $a->det();

        // 4*6 - 7*2 = 24 - 14 = 10
        $this->assertEqualsWithDelta(10.0, $det, 1e-10);
    }

    public function testDet3x3(): void
    {
        $a = NDArray::array([
            [6, 1, 1],
            [4, -2, 5],
            [2, 8, 7],
        ], DType::Float64);

        $det = $a->det();

        // Known determinant: -306
        $this->assertEqualsWithDelta(-306.0, $det, 1e-8);
    }

    public function testDetRequiresSquare(): void
    {
        $a = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
        ], DType::Float64);

        $this->expectException(MathException::class);
        $a->det();
    }

    // =========================================================================
    // QR Decomposition Tests
    // =========================================================================

    public function testQr(): void
    {
        $a = NDArray::array([
            [12., -51., 4.],
            [6., 167., -68.],
            [-4., 24., -41.],
        ], DType::Float64);

        [$q, $r] = $a->qr();

        $this->assertSame([3, 3], $q->shape());
        $this->assertSame([3, 3], $r->shape());

        // Q * R should reconstruct A
        $reconstructed = $q->matmul($r);
        $this->assertEqualsWithDelta($a->toArray(), $reconstructed->toArray(), 1e-8);

        // Q^T * Q should be identity
        $qT = $q->transpose();
        $identity = $qT->matmul($q);
        $this->assertEqualsWithDelta([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ], $identity->toArray(), 1e-8);
    }

    public function testQrRequires2D(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);

        $this->expectException(ShapeException::class);
        $a->qr();
    }

    // =========================================================================
    // Cholesky Decomposition Tests
    // =========================================================================

    public function testCholeskyLower(): void
    {
        $a = NDArray::array([
            [4., 12., -16.],
            [12., 37., -43.],
            [-16., -43., 98.],
        ], DType::Float64);

        $l = $a->cholesky();

        $this->assertSame([3, 3], $l->shape());
        $this->assertEqualsWithDelta([
            [2., 0., 0.],
            [6., 1., 0.],
            [-8., 5., 3.],
        ], $l->toArray(), 1e-8);

        // L * L^T should reconstruct A
        $lT = $l->transpose();
        $reconstructed = $l->matmul($lT);
        $this->assertEqualsWithDelta($a->toArray(), $reconstructed->toArray(), 1e-8);
    }

    public function testCholeskyUpper(): void
    {
        $a = NDArray::array([
            [4., 12., -16.],
            [12., 37., -43.],
            [-16., -43., 98.],
        ], DType::Float64);

        $u = $a->cholesky(upper: true);

        $this->assertSame([3, 3], $u->shape());
        $this->assertEqualsWithDelta([
            [2., 6., -8.],
            [0., 1., 5.],
            [0., 0., 3.],
        ], $u->toArray(), 1e-8);

        // U^T * U should reconstruct A
        $uT = $u->transpose();
        $reconstructed = $uT->matmul($u);
        $this->assertEqualsWithDelta($a->toArray(), $reconstructed->toArray(), 1e-8);
    }

    public function testCholeskyRequiresSquare(): void
    {
        $a = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
        ], DType::Float64);

        $this->expectException(ShapeException::class);
        $a->cholesky();
    }

    public function testCholeskyRequiresPositiveDefinite(): void
    {
        // This matrix is symmetric but not positive definite
        $a = NDArray::array([
            [1., 2., 3.],
            [2., 4., 5.],
            [3., 5., 6.],
        ], DType::Float64);

        $this->expectException(MathException::class);
        $a->cholesky();
    }

    // =========================================================================
    // Least Squares Tests
    // =========================================================================

    public function testLstsq1D(): void
    {
        $a = NDArray::array([
            [1., 1., 1.],
            [2., 3., 4.],
            [3., 5., 2.],
            [4., 2., 5.],
            [5., 4., 3.],
        ], DType::Float64);
        $b = NDArray::array([-10., 12., 14., 16., 18.], DType::Float64);

        [$x, $residuals, $rank, $s] = $a->lstsq($b);

        $this->assertSame([3], $x->shape());
        $this->assertEqualsWithDelta(
            [2.0, 1.0, 1.0],
            $x->toArray(),
            1e-10
        );
        $this->assertSame(3, $rank);
        $this->assertSame([3], $s->shape());
        $this->assertNotNull($residuals);
        // residuals is a 0D scalar for 1D b
        $this->assertSame([], $residuals->shape());
    }

    public function testLstsq2D(): void
    {
        $a = NDArray::array([
            [1., 1., 1.],
            [2., 3., 4.],
            [3., 5., 2.],
            [4., 2., 5.],
            [5., 4., 3.],
        ], DType::Float64);
        $b = NDArray::array([
            [-10., -3.],
            [12., 14.],
            [14., 12.],
            [16., 16.],
            [18., 16.],
        ], DType::Float64);

        [$x, $residuals, $rank, $s] = $a->lstsq($b);

        $this->assertSame([3, 2], $x->shape());
        $this->assertEqualsWithDelta([
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
        ], $x->toArray(), 1e-10);
        $this->assertSame(3, $rank);
        $this->assertNotNull($residuals);
        // residuals is a 1D array with length = number of columns for 2D b
        $this->assertSame([2], $residuals->shape());
    }

    public function testLstsqRequires2DA(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $b = NDArray::array([1, 2, 3], DType::Float64);

        $this->expectException(ShapeException::class);
        $a->lstsq($b);
    }

    // =========================================================================
    // Pseudo-inverse Tests
    // =========================================================================

    public function testPinv(): void
    {
        $a = NDArray::array([
            [1., 2., 3.],
            [4., 5., 6.],
        ], DType::Float64);

        $pinv = $a->pinv();

        $this->assertSame([3, 2], $pinv->shape());

        // A * A^+ * A ≈ A
        $reconstructed = $a->matmul($pinv)->matmul($a);
        $this->assertEqualsWithDelta(
            $a->toArray(),
            $reconstructed->toArray(),
            1e-8
        );
    }

    public function testPinvRequires2D(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);

        $this->expectException(ShapeException::class);
        $a->pinv();
    }

    // =========================================================================
    // Condition Number Tests
    // =========================================================================

    public function testCond(): void
    {
        $a = NDArray::array([
            [1., 2.],
            [3., 4.],
        ], DType::Float64);

        $cond = $a->cond();

        // Known condition number for [[1,2],[3,4]] is ~14.93
        $this->assertEqualsWithDelta(14.93, $cond, 0.1);
    }

    public function testCondSingular(): void
    {
        $a = NDArray::array([
            [1., 2.],
            [2., 4.],
        ], DType::Float64);

        $cond = $a->cond();

        $this->assertGreaterThan(1e15, $cond);
    }

    public function testCondRequires2D(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);

        $this->expectException(ShapeException::class);
        $a->cond();
    }

    // =========================================================================
    // Rank Tests
    // =========================================================================

    public function testRank(): void
    {
        $a = NDArray::array([
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.],
        ], DType::Float64);

        // This matrix has rank 2
        $this->assertSame(2, $a->rank());
    }

    public function testRankFull(): void
    {
        $a = NDArray::array([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ], DType::Float64);

        $this->assertSame(3, $a->rank());
    }

    public function testRankWithTol(): void
    {
        $a = NDArray::array([
            [1., 2.],
            [2., 4.],
        ], DType::Float64);

        // With a very high tolerance, rank should be 0
        $this->assertSame(0, $a->rank(tol: 10.0));
    }

    public function testRankRequires2D(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);

        $this->expectException(ShapeException::class);
        $a->rank();
    }
}
