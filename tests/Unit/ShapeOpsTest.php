<?php

declare(strict_types=1);

namespace NDArray\Tests\Unit;

use NDArray\DType;
use NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Tests for shape manipulation operations
 */
final class ShapeOpsTest extends TestCase
{
    // =========================================================================
    // Reshape Tests
    // =========================================================================

    public function testReshape1DTo2D(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5, 6], DType::Float64);
        $result = $a->reshape([2, 3]);
        
        $this->assertSame([2, 3], $result->shape());
        $this->assertEqualsWithDelta([[1, 2, 3], [4, 5, 6]], $result->toArray(), 0.0001);
    }

    public function testReshape2DTo1D(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->reshape([6]);
        
        $this->assertSame([6], $result->shape());
        $this->assertEqualsWithDelta([1, 2, 3, 4, 5, 6], $result->toArray(), 0.0001);
    }

    public function testReshape3D(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5, 6, 7, 8], DType::Float64);
        $result = $a->reshape([2, 2, 2]);
        
        $this->assertSame([2, 2, 2], $result->shape());
    }

    public function testReshapePreservesDtype(): void
    {
        $a = NDArray::array([1, 2, 3, 4], DType::Int32);
        $result = $a->reshape([2, 2]);
        
        $this->assertSame(DType::Int32, $result->dtype());
    }

    // =========================================================================
    // Transpose Tests
    // =========================================================================

    public function testTranspose2D(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->transpose();
        
        $this->assertSame([3, 2], $result->shape());
        $this->assertEqualsWithDelta([[1, 4], [2, 5], [3, 6]], $result->toArray(), 0.0001);
    }

    public function testTransposePreservesData(): void
    {
        $a = NDArray::array([[1, 2], [3, 4], [5, 6]], DType::Float64);
        $result = $a->transpose();
        
        $this->assertSame([2, 3], $result->shape());
        $this->assertEqualsWithDelta([[1, 3, 5], [2, 4, 6]], $result->toArray(), 0.0001);
    }

    // =========================================================================
    // Swap Axes Tests
    // =========================================================================

    public function testSwapAxes2D(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->swapAxes(0, 1);
        
        $this->assertSame([3, 2], $result->shape());
        $this->assertEqualsWithDelta([[1, 4], [2, 5], [3, 6]], $result->toArray(), 0.0001);
    }

    public function testSwapAxesThenFlatten(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->swapAxes(0, 1)->flatten();
        
        $this->assertSame([6], $result->shape());
        $this->assertEqualsWithDelta([1, 4, 2, 5, 3, 6], $result->toArray(), 0.0001);
    }

    // =========================================================================
    // Flatten Tests
    // =========================================================================

    public function testFlatten2D(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->flatten();
        
        $this->assertSame([6], $result->shape());
        $this->assertEqualsWithDelta([1, 2, 3, 4, 5, 6], $result->toArray(), 0.0001);
    }

    public function testFlatten3D(): void
    {
        $a = NDArray::array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], DType::Float64);
        $result = $a->flatten();
        
        $this->assertSame([8], $result->shape());
        $this->assertEqualsWithDelta([1, 2, 3, 4, 5, 6, 7, 8], $result->toArray(), 0.0001);
    }

    // =========================================================================
    // Squeeze Tests
    // =========================================================================

    public function testSqueezeAllOnes(): void
    {
        $a = NDArray::array([[[[1]]]], DType::Float64);
        $result = $a->squeeze();
        
        // Should keep at least 1 dimension
        $this->assertSame([1], $result->shape());
    }

    public function testSqueezeSpecificAxis(): void
    {
        $a = NDArray::zeros([3, 1, 4], DType::Float64);
        $result = $a->squeeze([1]);
        
        $this->assertSame([3, 4], $result->shape());
    }

    public function testSqueezeMultipleAxes(): void
    {
        $a = NDArray::zeros([1, 3, 1, 4, 1], DType::Float64);
        $result = $a->squeeze([0, 2, 4]);
        
        $this->assertSame([3, 4], $result->shape());
    }

    // =========================================================================
    // Expand Dims Tests
    // =========================================================================

    public function testExpandDimsAtBeginning(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $result = $a->expandDims(0);
        
        $this->assertSame([1, 3], $result->shape());
    }

    public function testExpandDimsAtEnd(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $result = $a->expandDims(1);
        
        $this->assertSame([3, 1], $result->shape());
    }

    public function testExpandDimsMiddle(): void
    {
        $a = NDArray::zeros([2, 3], DType::Float64);
        $result = $a->expandDims(1);
        
        $this->assertSame([2, 1, 3], $result->shape());
    }

    public function testExpandDimsNegativeAxis(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $result = $a->expandDims(-1);
        
        $this->assertSame([3, 1], $result->shape());
    }

    // =========================================================================
    // Insert Axis Tests
    // =========================================================================

    public function testInsertAxisAtBeginning(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $result = $a->insertAxis(0);
        
        $this->assertSame([1, 3], $result->shape());
        $this->assertEqualsWithDelta([[1, 2, 3]], $result->toArray(), 0.0001);
    }

    public function testInsertAxisAtEnd(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $result = $a->insertAxis(1);
        
        $this->assertSame([3, 1], $result->shape());
        $this->assertEqualsWithDelta([[1], [2], [3]], $result->toArray(), 0.0001);
    }

    public function testInsertAxisInMiddle(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->insertAxis(1);
        
        $this->assertSame([2, 1, 3], $result->shape());
    }

    public function testInsertAxisNegativeIndex(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $result = $a->insertAxis(-1);
        
        $this->assertSame([3, 1], $result->shape());
    }

    // =========================================================================
    // Merge Axes Tests
    // =========================================================================

    public function testMergeAxes2DInto1D(): void
    {
        $a = NDArray::array([[1, 2], [3, 4], [5, 6]], DType::Float64);
        $result = $a->mergeAxes(0, 1);
        
        // Merging axis 0 into axis 1: [3, 2] -> [6]
        $this->assertSame([6], $result->shape());
    }

    public function testMergeAxes3D(): void
    {
        $a = NDArray::zeros([2, 3, 4], DType::Float64);
        $result = $a->mergeAxes(1, 2);
        
        // Merging axis 1 into axis 2: [2, 3, 4] -> [2, 12]
        $this->assertSame([2, 12], $result->shape());
    }

    // =========================================================================
    // Invert Axis Tests
    // =========================================================================

    public function testInvertAxis1D(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5], DType::Float64);
        $result = $a->invertAxis(0);
        
        $this->assertSame([5], $result->shape());
        $this->assertEqualsWithDelta([5, 4, 3, 2, 1], $result->toArray(), 0.0001);
    }

    public function testInvertAxis2D(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->invertAxis(0);
        
        $this->assertSame([2, 3], $result->shape());
        $this->assertEqualsWithDelta([[4, 5, 6], [1, 2, 3]], $result->toArray(), 0.0001);
    }

    public function testInvertAxisNegativeIndex(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5], DType::Float64);
        $result = $a->invertAxis(-1);
        
        $this->assertSame([5], $result->shape());
        $this->assertEqualsWithDelta([5, 4, 3, 2, 1], $result->toArray(), 0.0001);
    }

    // =========================================================================
    // Ravel Tests
    // =========================================================================

    public function testRavel2D(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->ravel();
        
        $this->assertSame([6], $result->shape());
        $this->assertEqualsWithDelta([1, 2, 3, 4, 5, 6], $result->toArray(), 0.0001);
    }

    // =========================================================================
    // Permute Axes Tests
    // =========================================================================

    public function testPermuteAxes2D(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->permuteAxes([1, 0]);
        
        $this->assertSame([3, 2], $result->shape());
        $this->assertEqualsWithDelta([[1, 4], [2, 5], [3, 6]], $result->toArray(), 0.0001);
    }

    public function testPermuteAxes3D(): void
    {
        $a = NDArray::array([
            [[1, 2], [3, 4], [5, 6]],
            [[7, 8], [9, 10], [11, 12]]
        ], DType::Float64);
        // Original shape: [2, 3, 2]
        $result = $a->permuteAxes([1, 0, 2]);
        // New shape: [3, 2, 2]
        
        $this->assertSame([3, 2, 2], $result->shape());
        $this->assertEqualsWithDelta([
            [[1, 2], [7, 8]],
            [[3, 4], [9, 10]],
            [[5, 6], [11, 12]]
        ], $result->toArray(), 0.0001);
    }

    public function testPermuteAxesWithNegativeIndices(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->permuteAxes([-1, -2]);
        
        $this->assertSame([3, 2], $result->shape());
        $this->assertEqualsWithDelta([[1, 4], [2, 5], [3, 6]], $result->toArray(), 0.0001);
    }

    public function testPermuteAxesIdentity(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->permuteAxes([0, 1]);
        
        $this->assertSame([2, 3], $result->shape());
        $this->assertEqualsWithDelta([[1, 2, 3], [4, 5, 6]], $result->toArray(), 0.0001);
    }

    public function testPermuteAxesThenFlatten(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->permuteAxes([1, 0])->flatten();
        
        $this->assertSame([6], $result->shape());
        $this->assertEqualsWithDelta([1, 4, 2, 5, 3, 6], $result->toArray(), 0.0001);
    }

    public function testPermuteAxesWithInvalidNumberOfAxes(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);
        
        $this->expectException(\NDArray\Exceptions\ShapeException::class);
        $a->permuteAxes([0, 1, 2]); // 3 axes for 2D array
    }

    public function testPermuteAxesWithDuplicateAxes(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);
        
        $this->expectException(\NDArray\Exceptions\ShapeException::class);
        $a->permuteAxes([0, 0]); // Duplicate axis
    }

    public function testPermuteAxesWithOutOfBoundsAxis(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);
        
        $this->expectException(\NDArray\Exceptions\ShapeException::class);
        $a->permuteAxes([0, 5]); // Axis 5 is out of bounds
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    public function testChainedOperations(): void
    {
        // Create 1D array, reshape to 2D, transpose, then flatten
        $a = NDArray::array([1, 2, 3, 4, 5, 6], DType::Float64);
        $result = $a->reshape([2, 3])->transpose()->flatten();
        
        $this->assertSame([6], $result->shape());
        // After reshape: [[1,2,3],[4,5,6]], transpose: [[1,4],[2,5],[3,6]], flatten: [1,4,2,5,3,6]
        $this->assertEqualsWithDelta([1, 4, 2, 5, 3, 6], $result->toArray(), 0.0001);
    }

    public function testSqueezeThenExpand(): void
    {
        $a = NDArray::zeros([1, 3, 1, 4, 1], DType::Float64);
        $squeezed = $a->squeeze();
        $expanded = $squeezed->expandDims(0);
        
        $this->assertSame([1, 3, 4], $expanded->shape());
    }

    public function testInsertAxisThenFlatten(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $result = $a->insertAxis(0)->flatten();
        
        $this->assertSame([3], $result->shape());
        $this->assertEqualsWithDelta([1, 2, 3], $result->toArray(), 0.0001);
    }

    public function testTransposeThenInsertAxis(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->transpose()->insertAxis(1);
        
        $this->assertSame([3, 1, 2], $result->shape());
    }

    public function testInvertAxisThenFlatten(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5], DType::Float64);
        $result = $a->invertAxis(0)->flatten();
        
        $this->assertSame([5], $result->shape());
        $this->assertEqualsWithDelta([5, 4, 3, 2, 1], $result->toArray(), 0.0001);
    }

    public function testComplexChaining(): void
    {
        // Create array, reshape, transpose, insert axis, flatten
        $a = NDArray::array([1, 2, 3, 4, 5, 6, 7, 8], DType::Float64);
        $result = $a->reshape([2, 2, 2])
                    ->transpose()
                    ->insertAxis(0)
                    ->flatten();
        
        $this->assertSame([8], $result->shape());
    }

    public function testPermuteAxesChaining(): void
    {
        // Permute axes multiple times
        $a = NDArray::array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ], DType::Float64);
        
        // First permute: [1, 0, 2] - swap first two axes
        // Then flatten
        $result = $a->permuteAxes([1, 0, 2])->flatten();
        
        $this->assertSame([8], $result->shape());
        // After permute: [[[1,2],[5,6]], [[3,4],[7,8]]]
        // Flatten: [1,2,5,6,3,4,7,8]
        $this->assertEqualsWithDelta([1, 2, 5, 6, 3, 4, 7, 8], $result->toArray(), 0.0001);
    }

    public function testPermuteAxesThenReshape(): void
    {
        $a = NDArray::array([
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]]
        ], DType::Float64);
        
        // Shape [2, 2, 3] -> permute [1, 0, 2] -> [2, 2, 3] -> reshape to [4, 3]
        $result = $a->permuteAxes([1, 0, 2])->reshape([4, 3]);
        
        $this->assertSame([4, 3], $result->shape());
        $this->assertEqualsWithDelta([
            [1, 2, 3],
            [7, 8, 9],
            [4, 5, 6],
            [10, 11, 12]
        ], $result->toArray(), 0.0001);
    }
}
