<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Unit;

use PhpMlKit\NDArray\DType;
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
}
