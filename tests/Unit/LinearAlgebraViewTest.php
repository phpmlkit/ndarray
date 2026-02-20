<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Unit;

use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Tests for linear algebra operations on views.
 */
class LinearAlgebraViewTest extends TestCase
{
    public function testDotOnView(): void
    {
        // Create a 3x3 matrix and take a 2x2 slice
        $a = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
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
            [7, 8, 9]
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
            [0, 0, 7, 8]
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
            [13, 14, 15, 16]
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
            [7, 8, 9]
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
            [7, 8, 9]
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
            [7, 8, 9]
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
            [13, 14, 15, 16]
        ], DType::Float64);
        
        $b = NDArray::array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], DType::Float64);
        
        // Get top-left 2x2 views of both
        $aView = $a->slice(['0:2', '0:2']);
        $bView = $b->slice(['0:2', '0:2']);
        
        $result = $aView->dot($bView);
        
        // [[1, 2], [5, 6]] @ [[1, 0], [0, 1]] = [[1, 2], [5, 6]]
        $this->assertEqualsWithDelta([[1, 2], [5, 6]], $result->toArray(), 0.0001);
    }
}
