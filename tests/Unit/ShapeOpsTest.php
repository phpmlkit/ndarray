<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Unit;

use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\NDArray;
use PhpMlKit\NDArray\PadMode;
use PhpMlKit\NDArray\Exceptions\ShapeException;
use PHPUnit\Framework\TestCase;

/**
 * Tests for shape manipulation operations
 */
final class ShapeOpsTest extends TestCase
{
    // =========================================================================
    // Pad Tests
    // =========================================================================

    public function testPadConstant1D(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Int64);
        $result = $a->pad(2);

        $this->assertSame([7], $result->shape());
        $this->assertSame([0, 0, 1, 2, 3, 0, 0], $result->toArray());
    }

    public function testPadConstant2DWithPerAxisWidth(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Int64);
        $result = $a->pad([[1, 0], [0, 1]], PadMode::Constant, 9);

        $this->assertSame([3, 3], $result->shape());
        $this->assertSame([
            [9, 9, 9],
            [1, 2, 9],
            [3, 4, 9],
        ], $result->toArray());
    }

    public function testPadSymmetric1D(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Int64);
        $result = $a->pad([2, 1], PadMode::Symmetric);

        $this->assertSame([6], $result->shape());
        $this->assertSame([2, 1, 1, 2, 3, 3], $result->toArray());
    }

    public function testPadReflect1D(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Int64);
        $result = $a->pad([2, 1], PadMode::Reflect);

        $this->assertSame([6], $result->shape());
        $this->assertSame([3, 2, 1, 2, 3, 2], $result->toArray());
    }

    public function testPadBool(): void
    {
        $a = NDArray::array([[true, false]], DType::Bool);
        $result = $a->pad([[0, 1], [1, 0]], PadMode::Constant, false);

        $this->assertSame([[false, true, false], [false, false, false]], $result->toArray());
    }

    public function testPadOnView(): void
    {
        $a = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Int64);

        $view = $a->slice(['::2', '1:']); // [[2,3], [8,9]]
        $result = $view->pad(1, PadMode::Reflect);

        $this->assertSame([4, 4], $result->shape());
        $this->assertSame([
            [9, 8, 9, 8],
            [3, 2, 3, 2],
            [9, 8, 9, 8],
            [3, 2, 3, 2],
        ], $result->toArray());
    }

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
        
        $this->expectException(ShapeException::class);
        $a->permuteAxes([0, 1, 2]); // 3 axes for 2D array
    }

    public function testPermuteAxesWithDuplicateAxes(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);
        
        $this->expectException(ShapeException::class);
        $a->permuteAxes([0, 0]); // Duplicate axis
    }

    public function testPermuteAxesWithOutOfBoundsAxis(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);

        $this->expectException(ShapeException::class);
        $a->permuteAxes([0, 5]); // Axis 5 is out of bounds
    }

    // =========================================================================
    // Tile Tests
    // =========================================================================

    public function testTile1DScalar(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Int64);
        $result = $a->tile(2);

        $this->assertSame([6], $result->shape());
        $this->assertSame([1, 2, 3, 1, 2, 3], $result->toArray());
    }

    public function testTile1DArray(): void
    {
        $a = NDArray::array([1, 2], DType::Int64);
        $result = $a->tile([3]);

        $this->assertSame([6], $result->shape());
        $this->assertSame([1, 2, 1, 2, 1, 2], $result->toArray());
    }

    public function testTile2DRepeatRows(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Int64);
        $result = $a->tile([2, 1]);

        $this->assertSame([4, 2], $result->shape());
        $this->assertSame([
            [1, 2],
            [3, 4],
            [1, 2],
            [3, 4],
        ], $result->toArray());
    }

    public function testTile2DRepeatCols(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Int64);
        $result = $a->tile([1, 2]);

        $this->assertSame([2, 4], $result->shape());
        $this->assertSame([
            [1, 2, 1, 2],
            [3, 4, 3, 4],
        ], $result->toArray());
    }

    public function testTile2DRepeatBoth(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Int64);
        $result = $a->tile([2, 3]);

        $this->assertSame([4, 6], $result->shape());
        $this->assertSame([
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
        ], $result->toArray());
    }

    public function testTile3D(): void
    {
        $a = NDArray::array([[[1, 2]], [[3, 4]]], DType::Int64); // Shape [2, 1, 2]
        $result = $a->tile([1, 2, 1]);

        $this->assertSame([2, 2, 2], $result->shape());
        $this->assertSame([
            [[1, 2], [1, 2]],
            [[3, 4], [3, 4]],
        ], $result->toArray());
    }

    public function testTileWithHigherRepsThanDims(): void
    {
        // When reps has more dims than array, array gets padded with 1s at the beginning
        $a = NDArray::array([1, 2], DType::Int64); // 1D
        $result = $a->tile([2, 3]); // reps is 2D

        $this->assertSame([2, 6], $result->shape());
    }

    public function testTilePreservesDtype(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float32);
        $result = $a->tile([2, 2]);

        $this->assertSame(DType::Float32, $result->dtype());
    }

    public function testTileFloat64(): void
    {
        $a = NDArray::array([[1.5, 2.5], [3.5, 4.5]], DType::Float64);
        $result = $a->tile([1, 2]);

        $this->assertSame([2, 4], $result->shape());
        $this->assertEqualsWithDelta([
            [1.5, 2.5, 1.5, 2.5],
            [3.5, 4.5, 3.5, 4.5],
        ], $result->toArray(), 0.0001);
    }

    public function testTileBool(): void
    {
        $a = NDArray::array([[true, false]], DType::Bool);
        $result = $a->tile([2, 1]);

        $this->assertSame([2, 2], $result->shape());
        $this->assertSame([
            [true, false],
            [true, false],
        ], $result->toArray());
    }

    public function testTileEmptyReps(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Int64);
        $result = $a->tile([]);

        // Empty reps should return array unchanged
        $this->assertSame([3], $result->shape());
        $this->assertSame([1, 2, 3], $result->toArray());
    }

    public function testTileWithNDArrayReps(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Int64);
        $reps = NDArray::array([2, 1], DType::Int64);
        $result = $a->tile($reps);

        $this->assertSame([4, 2], $result->shape());
    }

    // =========================================================================
    // Repeat Tests
    // =========================================================================

    public function testRepeat1DScalar(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Int64);
        $result = $a->repeat(2);

        $this->assertSame([6], $result->shape());
        $this->assertSame([1, 1, 2, 2, 3, 3], $result->toArray());
    }

    public function testRepeat1DWithArray(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Int64);
        $result = $a->repeat([1, 0, 2]);

        $this->assertSame([3], $result->shape());
        $this->assertSame([1, 3, 3], $result->toArray());
    }

    public function testRepeat2DFlattened(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Int64);
        $result = $a->repeat(2);

        $this->assertSame([8], $result->shape());
        $this->assertSame([1, 1, 2, 2, 3, 3, 4, 4], $result->toArray());
    }

    public function testRepeat2DAxis0(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Int64);
        $result = $a->repeat(2, axis: 0);

        $this->assertSame([4, 2], $result->shape());
        $this->assertSame([
            [1, 2],
            [1, 2],
            [3, 4],
            [3, 4],
        ], $result->toArray());
    }

    public function testRepeat2DAxis1(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Int64);
        $result = $a->repeat(2, axis: 1);

        $this->assertSame([2, 4], $result->shape());
        $this->assertSame([
            [1, 1, 2, 2],
            [3, 3, 4, 4],
        ], $result->toArray());
    }

    public function testRepeat2DWithArrayAxis0(): void
    {
        $a = NDArray::array([[1, 2], [3, 4], [5, 6]], DType::Int64);
        $result = $a->repeat([2, 1, 3], axis: 0);

        $this->assertSame([6, 2], $result->shape());
        $this->assertSame([
            [1, 2],
            [1, 2],
            [3, 4],
            [5, 6],
            [5, 6],
            [5, 6],
        ], $result->toArray());
    }

    public function testRepeat2DWithArrayAxis1(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Int64);
        $result = $a->repeat([1, 2], axis: 1);

        $this->assertSame([2, 3], $result->shape());
        $this->assertSame([
            [1, 2, 2],
            [3, 4, 4],
        ], $result->toArray());
    }

    public function testRepeat3DAxis0(): void
    {
        $a = NDArray::array([[[1, 2]], [[3, 4]]], DType::Int64); // Shape [2, 1, 2]
        $result = $a->repeat(2, axis: 0);

        $this->assertSame([4, 1, 2], $result->shape());
    }

    public function testRepeat3DAxis1(): void
    {
        $a = NDArray::array([[[1, 2]], [[3, 4]]], DType::Int64); // Shape [2, 1, 2]
        $result = $a->repeat(2, axis: 1);

        $this->assertSame([2, 2, 2], $result->shape());
        $this->assertSame([
            [[1, 2], [1, 2]],
            [[3, 4], [3, 4]],
        ], $result->toArray());
    }

    public function testRepeatPreservesDtype(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float32);
        $result = $a->repeat(2, axis: 0);

        $this->assertSame(DType::Float32, $result->dtype());
    }

    public function testRepeatFloat64(): void
    {
        $a = NDArray::array([[1.5, 2.5], [3.5, 4.5]], DType::Float64);
        $result = $a->repeat(2, axis: 1);

        $this->assertSame([2, 4], $result->shape());
        $this->assertEqualsWithDelta([
            [1.5, 1.5, 2.5, 2.5],
            [3.5, 3.5, 4.5, 4.5],
        ], $result->toArray(), 0.0001);
    }

    public function testRepeatBool(): void
    {
        $a = NDArray::array([[true, false]], DType::Bool);
        $result = $a->repeat(2, axis: 0);

        $this->assertSame([2, 2], $result->shape());
        $this->assertSame([
            [true, false],
            [true, false],
        ], $result->toArray());
    }

    public function testRepeatWithNDArrayRepeats(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Int64);
        $repeats = NDArray::array([2, 1], DType::Int64);
        $result = $a->repeat($repeats, axis: 1);

        $this->assertSame([2, 3], $result->shape());
        $this->assertSame([
            [1, 1, 2],
            [3, 3, 4],
        ], $result->toArray());
    }

    public function testRepeatOnceReturnsSame(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Int64);
        $result = $a->repeat(1, axis: 0);

        $this->assertSame([2, 2], $result->shape());
        $this->assertSame([
            [1, 2],
            [3, 4],
        ], $result->toArray());
    }

    // =========================================================================
    // Static Convenience Methods Tests
    // =========================================================================

    public function testTileArrayWithPlainArray(): void
    {
        $result = NDArray::tileArray([1, 2, 3], 2);

        $this->assertSame([6], $result->shape());
        $this->assertSame([1, 2, 3, 1, 2, 3], $result->toArray());
    }

    public function testTileArrayWithNDArray(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Int64);
        $result = NDArray::tileArray($a, [2, 1]);

        $this->assertSame([4, 2], $result->shape());
        $this->assertSame([
            [1, 2],
            [3, 4],
            [1, 2],
            [3, 4],
        ], $result->toArray());
    }

    public function testTileArrayReturnsNDArray(): void
    {
        $result = NDArray::tileArray([1, 2], 2);

        $this->assertInstanceOf(NDArray::class, $result);
    }

    public function testRepeatArrayWithPlainArray(): void
    {
        $result = NDArray::repeatArray([1, 2, 3], 2);

        $this->assertSame([6], $result->shape());
        $this->assertSame([1, 1, 2, 2, 3, 3], $result->toArray());
    }

    public function testRepeatArrayWithNDArray(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Int64);
        $result = NDArray::repeatArray($a, 2, axis: 0);

        $this->assertSame([4, 2], $result->shape());
        $this->assertSame([
            [1, 2],
            [1, 2],
            [3, 4],
            [3, 4],
        ], $result->toArray());
    }

    public function testRepeatArrayWithAxisParameter(): void
    {
        $result = NDArray::repeatArray([[1, 2], [3, 4]], 2, axis: 1);

        $this->assertSame([2, 4], $result->shape());
        $this->assertSame([
            [1, 1, 2, 2],
            [3, 3, 4, 4],
        ], $result->toArray());
    }

    public function testRepeatArrayReturnsNDArray(): void
    {
        $result = NDArray::repeatArray([1, 2], 2);

        $this->assertInstanceOf(NDArray::class, $result);
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
