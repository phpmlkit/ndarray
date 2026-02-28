<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Unit;

use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\Exceptions\ShapeException;
use PhpMlKit\NDArray\NDArray;
use PhpMlKit\NDArray\PadMode;
use PHPUnit\Framework\TestCase;

/**
 * Tests for shape manipulation operations.
 *
 * @internal
 *
 * @coversNothing
 */
final class ShapeOpsTest extends TestCase
{
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

    public function testSwapAxes2D(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->swapaxes(0, 1);

        $this->assertSame([3, 2], $result->shape());
        $this->assertEqualsWithDelta([[1, 4], [2, 5], [3, 6]], $result->toArray(), 0.0001);
    }

    public function testSwapAxesThenFlatten(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->swapaxes(0, 1)->flatten();

        $this->assertSame([6], $result->shape());
        $this->assertEqualsWithDelta([1, 4, 2, 5, 3, 6], $result->toArray(), 0.0001);
    }

    public function testSwapAxesReturnsView(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->swapaxes(0, 1);

        // Verify it's a view
        $this->assertTrue($result->isView());

        // Verify modification affects original
        $result->setAt(0, 999);
        $this->assertEqualsWithDelta(999, $a->getAt(0), 0.0001);
    }

    public function testSwapAxes3D(): void
    {
        $a = NDArray::arange(24)->reshape([2, 3, 4]);
        $result = $a->swapaxes(0, 2);

        $this->assertSame([4, 3, 2], $result->shape());
    }

    public function testSwapAxesNegativeIndices(): void
    {
        $a = NDArray::array([[1, 2], [3, 4], [5, 6]], DType::Float64);
        $result = $a->swapaxes(0, -1);

        $this->assertSame([2, 3], $result->shape());
        $this->assertEqualsWithDelta([[1, 3, 5], [2, 4, 6]], $result->toArray(), 0.0001);
    }

    public function testSwapAxesSameAxisNoOp(): void
    {
        $a = NDArray::array([1, 2, 3]);
        $result = $a->swapaxes(0, 0);

        // Should return the same object when swapping same axis
        $this->assertSame($a, $result);
    }

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

    public function testInsertAxisAtBeginning(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $result = $a->insertaxis(0);

        $this->assertSame([1, 3], $result->shape());
        $this->assertEqualsWithDelta([[1, 2, 3]], $result->toArray(), 0.0001);
    }

    public function testInsertAxisAtEnd(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $result = $a->insertaxis(1);

        $this->assertSame([3, 1], $result->shape());
        $this->assertEqualsWithDelta([[1], [2], [3]], $result->toArray(), 0.0001);
    }

    public function testInsertAxisInMiddle(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->insertaxis(1);

        $this->assertSame([2, 1, 3], $result->shape());
    }

    public function testInsertAxisNegativeIndex(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $result = $a->insertaxis(-1);

        $this->assertSame([3, 1], $result->shape());
    }

    public function testMergeAxes2DInto1D(): void
    {
        $a = NDArray::array([[1, 2], [3, 4], [5, 6]], DType::Float64);
        $result = $a->mergeaxes(0, 1);

        // Merging axis 0 into axis 1: [3, 2] -> [6]
        $this->assertSame([6], $result->shape());
    }

    public function testMergeAxes3D(): void
    {
        $a = NDArray::zeros([2, 3, 4], DType::Float64);
        $result = $a->mergeaxes(1, 2);

        // Merging axis 1 into axis 2: [2, 3, 4] -> [2, 12]
        $this->assertSame([2, 12], $result->shape());
    }

    public function testMergeReturnsView(): void
    {
        $a = NDArray::array([[1, 2], [3, 4], [5, 6]], DType::Float64);
        $result = $a->mergeaxes(0, 1);

        // Verify it returns a view
        $this->assertTrue($result->isView());

        // Verify modification affects original
        $result->setAt(0, 999);
        $this->assertEqualsWithDelta(999, $a->getAt(0), 0.0001);
    }

    public function testMergeWithNegativeIndices(): void
    {
        $a = NDArray::zeros([2, 3, 4], DType::Float64);
        $result = $a->mergeaxes(-2, -1);  // Same as mergeaxes(1, 2)

        $this->assertSame([2, 12], $result->shape());
        $this->assertTrue($result->isView());
    }

    public function testFlip1D(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5], DType::Float64);
        $result = $a->flip(0);

        $this->assertSame([5], $result->shape());
        $this->assertEqualsWithDelta([5, 4, 3, 2, 1], $result->toArray(), 0.0001);
    }

    public function testFlip2D(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->flip(0);

        $this->assertSame([2, 3], $result->shape());
        $this->assertEqualsWithDelta([[4, 5, 6], [1, 2, 3]], $result->toArray(), 0.0001);
    }

    public function testFlipNegativeIndex(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5], DType::Float64);
        $result = $a->flip(-1);

        $this->assertSame([5], $result->shape());
        $this->assertEqualsWithDelta([5, 4, 3, 2, 1], $result->toArray(), 0.0001);
    }

    public function testFlipAllAxes(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->flip();

        $this->assertSame([2, 3], $result->shape());
        $this->assertEqualsWithDelta([[6, 5, 4], [3, 2, 1]], $result->toArray(), 0.0001);
    }

    public function testFlipMultipleAxes(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->flip([0, 1]);

        $this->assertSame([2, 3], $result->shape());
        $this->assertEqualsWithDelta([[6, 5, 4], [3, 2, 1]], $result->toArray(), 0.0001);
    }

    public function testRavel2D(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->ravel();

        $this->assertSame([6], $result->shape());
        $this->assertEqualsWithDelta([1, 2, 3, 4, 5, 6], $result->toArray(), 0.0001);
    }

    public function testPermuteAxes2D(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->permute(1, 0);

        $this->assertSame([3, 2], $result->shape());
        $this->assertEqualsWithDelta([[1, 4], [2, 5], [3, 6]], $result->toArray(), 0.0001);
    }

    public function testPermuteAxes3D(): void
    {
        $a = NDArray::array([
            [[1, 2], [3, 4], [5, 6]],
            [[7, 8], [9, 10], [11, 12]],
        ], DType::Float64);
        // Original shape: [2, 3, 2]
        $result = $a->permute(1, 0, 2);
        // New shape: [3, 2, 2]

        $this->assertSame([3, 2, 2], $result->shape());
        $this->assertEqualsWithDelta([
            [[1, 2], [7, 8]],
            [[3, 4], [9, 10]],
            [[5, 6], [11, 12]],
        ], $result->toArray(), 0.0001);
    }

    public function testPermuteAxesWithNegativeIndices(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->permute(-1, -2);

        $this->assertSame([3, 2], $result->shape());
        $this->assertEqualsWithDelta([[1, 4], [2, 5], [3, 6]], $result->toArray(), 0.0001);
    }

    public function testPermuteAxesIdentity(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->permute(0, 1);

        $this->assertSame([2, 3], $result->shape());
        $this->assertEqualsWithDelta([[1, 2, 3], [4, 5, 6]], $result->toArray(), 0.0001);
    }

    public function testPermuteAxesThenFlatten(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->permute(1, 0)->flatten();

        $this->assertSame([6], $result->shape());
        $this->assertEqualsWithDelta([1, 4, 2, 5, 3, 6], $result->toArray(), 0.0001);
    }

    public function testPermuteAxesWithInvalidNumberOfAxes(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);

        $this->expectException(ShapeException::class);
        $a->permute(0, 1, 2); // 3 axes for 2D array
    }

    public function testPermuteAxesWithDuplicateAxes(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);

        $this->expectException(ShapeException::class);
        $a->permute(0, 0); // Duplicate axis
    }

    public function testPermuteAxesWithOutOfBoundsAxis(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);

        $this->expectException(ShapeException::class);
        $a->permute(0, 5); // Axis 5 is out of bounds
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
        $result = $a->insertaxis(0)->flatten();

        $this->assertSame([3], $result->shape());
        $this->assertEqualsWithDelta([1, 2, 3], $result->toArray(), 0.0001);
    }

    public function testTransposeThenInsertAxis(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->transpose()->insertaxis(1);

        $this->assertSame([3, 1, 2], $result->shape());
    }

    public function testFlipThenFlatten(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5], DType::Float64);
        $result = $a->flip(0)->flatten();

        $this->assertSame([5], $result->shape());
        $this->assertEqualsWithDelta([5, 4, 3, 2, 1], $result->toArray(), 0.0001);
    }

    public function testComplexChaining(): void
    {
        // Create array, reshape, transpose, insert axis, flatten
        $a = NDArray::array([1, 2, 3, 4, 5, 6, 7, 8], DType::Float64);
        $result = $a->reshape([2, 2, 2])
            ->transpose()
            ->insertaxis(0)
            ->flatten();

        $this->assertSame([8], $result->shape());
    }

    public function testPermuteAxesChaining(): void
    {
        // Permute axes multiple times
        $a = NDArray::array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ], DType::Float64);

        // First permute: [1, 0, 2] - swap first two axes
        // Then flatten
        $result = $a->permute(1, 0, 2)->flatten();

        $this->assertSame([8], $result->shape());
        // After permute: [[[1,2],[5,6]], [[3,4],[7,8]]]
        // Flatten: [1,2,5,6,3,4,7,8]
        $this->assertEqualsWithDelta([1, 2, 5, 6, 3, 4, 7, 8], $result->toArray(), 0.0001);
    }

    public function testPermuteAxesThenReshape(): void
    {
        $a = NDArray::array([
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ], DType::Float64);

        // Shape [2, 2, 3] -> permute [1, 0, 2] -> [2, 2, 3] -> reshape to [4, 3]
        $result = $a->permute(1, 0, 2)->reshape([4, 3]);

        $this->assertSame([4, 3], $result->shape());
        $this->assertEqualsWithDelta([
            [1, 2, 3],
            [7, 8, 9],
            [4, 5, 6],
            [10, 11, 12],
        ], $result->toArray(), 0.0001);
    }

    // VIEW/SUBSET TESTS (from ShapeOpsViewTest)

    // ========================================================================
    // Slice Shape Tests
    // ========================================================================

    public function test1DSliceShape(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], DType::Float64);
        $view = $a->slice(['2:8']); // [3, 4, 5, 6, 7, 8]

        $this->assertSame([6], $view->shape());
        $this->assertEqualsWithDelta([3, 4, 5, 6, 7, 8], $view->toArray(), 0.0001);
    }

    public function test2DSliceShape(): void
    {
        $a = NDArray::array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ], DType::Float64);
        $view = $a->slice(['0:2', '1:3']); // [[2, 3], [6, 7]]

        $this->assertSame([2, 2], $view->shape());
        $this->assertEqualsWithDelta([[2, 3], [6, 7]], $view->toArray(), 0.0001);
    }

    public function test3DSliceShape(): void
    {
        $a = NDArray::array([
            [[1, 2], [3, 4], [5, 6]],
            [[7, 8], [9, 10], [11, 12]],
        ], DType::Float64);
        $view = $a->slice(['0:1', ':', ':']); // [[[1, 2], [3, 4], [5, 6]]]

        $this->assertSame([1, 3, 2], $view->shape());
    }

    public function test2DRowSliceShape(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], DType::Float64);
        $row = $a->slice(['1:2']); // [[4, 5, 6]]

        $this->assertSame([1, 3], $row->shape());
        $this->assertEqualsWithDelta([[4, 5, 6]], $row->toArray(), 0.0001);
    }

    public function test2DColumnSliceShape(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], DType::Float64);
        $col = $a->slice([':', '1:2']); // [[2], [5], [8]]

        $this->assertSame([3, 1], $col->shape());
        $this->assertEqualsWithDelta([[2], [5], [8]], $col->toArray(), 0.0001);
    }

    public function testStridedSliceShape(): void
    {
        $a = NDArray::array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ], DType::Float64);
        $strided = $a->slice(['::2', '::2']); // [[1, 3], [9, 11]]

        $this->assertSame([2, 2], $strided->shape());
        $this->assertEqualsWithDelta([[1, 3], [9, 11]], $strided->toArray(), 0.0001);
    }

    public function testSingleElementSliceShape(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        // String indices '1' and '2' are converted to integers, extracting single element
        $single = $a->slice(['1', '2']);

        // All dimensions indexed with integers returns scalar (shape [])
        $this->assertSame([], $single->shape());
        $this->assertEqualsWithDelta(6, $single->toArray(), 0.0001);
    }

    public function testEmptySliceShape(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5], DType::Float64);
        $empty = $a->slice(['2:2']); // Empty slice

        $this->assertSame([0], $empty->shape());
    }

    public function testNegativeIndexSliceShape(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], DType::Float64);
        $view = $a->slice(['-2:', '-2:']); // Last 2 rows, last 2 columns

        $this->assertSame([2, 2], $view->shape());
        $this->assertEqualsWithDelta([[5, 6], [8, 9]], $view->toArray(), 0.0001);
    }

    // ========================================================================
    // Array Access Tests
    // ========================================================================

    public function testArrayAccessRow(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], DType::Float64);
        $row = $a[1]; // [4, 5, 6]

        $this->assertSame([3], $row->shape());
        $this->assertEqualsWithDelta([4, 5, 6], $row->toArray(), 0.0001);
    }

    public function testArrayAccessFirstRow(): void
    {
        $a = NDArray::array([[1, 2, 3, 4], [5, 6, 7, 8]], DType::Int32);
        $row = $a[0]; // [1, 2, 3, 4]

        $this->assertSame([4], $row->shape());
        $this->assertSame(DType::Int32, $row->dtype());
    }

    // ========================================================================
    // DType Preservation on Views
    // ========================================================================

    public function testViewPreservesDtype(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Int32);
        $view = $a->slice(['0:1', ':']);

        $this->assertSame(DType::Int32, $view->dtype());
    }

    public function testFloat64ViewPreservesDtype(): void
    {
        $a = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Float64);
        $view = $a->slice(['0:2', '1:3']);

        $this->assertSame(DType::Float64, $view->dtype());
    }

    // ========================================================================
    // Complex Slice Tests
    // ========================================================================

    public function test3DComplexSlice(): void
    {
        $a = NDArray::array([
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            [[[9, 10], [11, 12]], [[13, 14], [15, 16]]],
        ], DType::Float64);
        $view = $a->slice(['0:1', ':', ':', ':']);

        $this->assertSame([1, 2, 2, 2], $view->shape());
    }

    public function testSliceWithMultipleDimensions(): void
    {
        $a = NDArray::array([
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ], DType::Float64);
        // String index '0' is converted to integer 0, extracting first row of second dimension
        $view = $a->slice([':', '0', ':']);

        $this->assertSame([2, 3], $view->shape());
        $this->assertEquals([[1, 2, 3], [7, 8, 9]], $view->toArray());
    }

    // ========================================================================
    // Shape Operations on Views
    // ========================================================================

    public function testViewReshape(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5, 6, 7, 8], DType::Float64);
        $view = $a->slice(['2:6']); // [3, 4, 5, 6]
        $result = $view->reshape([2, 2]);

        $this->assertSame([2, 2], $result->shape());
        $this->assertEqualsWithDelta([[3, 4], [5, 6]], $result->toArray(), 0.0001);
    }

    public function testViewFlatten(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], DType::Float64);
        $view = $a->slice(['0:2', '1:3']); // [[2, 3], [5, 6]]
        $result = $view->flatten();

        $this->assertSame([4], $result->shape());
        $this->assertEqualsWithDelta([2, 3, 5, 6], $result->toArray(), 0.0001);
    }

    public function testViewTranspose(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], DType::Float64);
        $view = $a->slice(['0:2', ':']); // [[1, 2, 3], [4, 5, 6]]
        $result = $view->transpose();

        $this->assertSame([3, 2], $result->shape());
        $this->assertEqualsWithDelta([[1, 4], [2, 5], [3, 6]], $result->toArray(), 0.0001);
    }

    public function testViewSwapAxes(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], DType::Float64);
        $view = $a->slice(['0:2', ':']); // [[1, 2, 3], [4, 5, 6]]
        $result = $view->swapaxes(0, 1);

        $this->assertSame([3, 2], $result->shape());
        $this->assertEqualsWithDelta([[1, 4], [2, 5], [3, 6]], $result->toArray(), 0.0001);
    }

    public function testViewSqueeze(): void
    {
        $a = NDArray::array([[[[1, 2]]], [[[3, 4]]]], DType::Float64);
        $view = $a->slice(['0:1', ':', ':', ':']); // [[[[1, 2]]]]
        $result = $view->squeeze();

        $this->assertSame([2], $result->shape());
        $this->assertEqualsWithDelta([1, 2], $result->toArray(), 0.0001);
    }

    public function testViewInsertAxis(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5, 6], DType::Float64);
        $view = $a->slice(['1:4']); // [2, 3, 4]
        $result = $view->insertaxis(0);

        $this->assertSame([1, 3], $result->shape());
        $this->assertEqualsWithDelta([[2, 3, 4]], $result->toArray(), 0.0001);
    }

    public function testViewExpandDims(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5, 6], DType::Float64);
        $view = $a->slice(['1:4']); // [2, 3, 4]
        $result = $view->expandDims(1);

        $this->assertSame([3, 1], $result->shape());
        $this->assertEqualsWithDelta([[2], [3], [4]], $result->toArray(), 0.0001);
    }

    public function testViewFlip(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5, 6], DType::Float64);
        $view = $a->slice(['1:4']); // [2, 3, 4]
        $result = $view->flip(0);

        $this->assertSame([3], $result->shape());
        $this->assertEqualsWithDelta([4, 3, 2], $result->toArray(), 0.0001);
    }

    public function testViewRavel(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], DType::Float64);
        $view = $a->slice(['1:3', '0:2']); // [[4, 5], [7, 8]]
        $result = $view->ravel();

        $this->assertSame([4], $result->shape());
        $this->assertEqualsWithDelta([4, 5, 7, 8], $result->toArray(), 0.0001);
    }

    public function testViewMergeAxes(): void
    {
        $a = NDArray::array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]],
        ], DType::Float64);
        $view = $a->slice(['0:2', ':', ':']); // First 2 rows of 3D array [2, 2, 2]
        $result = $view->mergeaxes(0, 1);

        // Merging axes: [2, 2, 2] -> squeeze removes length-1 axis
        $this->assertSame([4, 2], $result->shape());
    }

    public function testViewTransposeThenFlatten(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], DType::Float64);
        $view = $a->slice(['0:2', ':']); // [[1, 2, 3], [4, 5, 6]]
        $result = $view->transpose()->flatten();

        $this->assertSame([6], $result->shape());
        $this->assertEqualsWithDelta([1, 4, 2, 5, 3, 6], $result->toArray(), 0.0001);
    }

    public function testViewComplexChaining(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], DType::Float64);
        $view = $a->slice(['2:10']); // [3, 4, 5, 6, 7, 8, 9, 10]
        $result = $view->reshape([2, 2, 2])
            ->transpose()
            ->insertaxis(0)
            ->flatten();

        $this->assertSame([8], $result->shape());
    }

    public function testViewSwapAxesThenFlatten(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], DType::Float64);
        $view = $a->slice(['0:2', ':']); // [[1, 2, 3], [4, 5, 6]]
        $result = $view->swapaxes(0, 1)->flatten();

        $this->assertSame([6], $result->shape());
        $this->assertEqualsWithDelta([1, 4, 2, 5, 3, 6], $result->toArray(), 0.0001);
    }

    public function testStridedViewFlatten(): void
    {
        $a = NDArray::array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ], DType::Float64);
        $view = $a->slice(['::2', '::2']); // [[1, 3], [9, 11]]
        $result = $view->flatten();

        $this->assertSame([4], $result->shape());
        $this->assertEqualsWithDelta([1, 3, 9, 11], $result->toArray(), 0.0001);
    }

    public function testStridedViewTranspose(): void
    {
        $a = NDArray::array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ], DType::Float64);
        $view = $a->slice(['::2', '::2']); // [[1, 3], [9, 11]]
        $result = $view->transpose();

        $this->assertSame([2, 2], $result->shape());
        $this->assertEqualsWithDelta([[1, 9], [3, 11]], $result->toArray(), 0.0001);
    }

    // ========================================================================
    // Permute Axes on Views Tests
    // ========================================================================

    public function testViewPermuteAxes(): void
    {
        $a = NDArray::array([
            [[1, 2], [3, 4], [5, 6]],
            [[7, 8], [9, 10], [11, 12]],
        ], DType::Float64);
        $view = $a->slice(['0:1', ':', ':']); // First row only
        $result = $view->permute(1, 0, 2);

        // Original view shape: [1, 3, 2]
        // After permute: [3, 1, 2]
        $this->assertSame([3, 1, 2], $result->shape());
        $this->assertEqualsWithDelta([
            [[1, 2]],
            [[3, 4]],
            [[5, 6]],
        ], $result->toArray(), 0.0001);
    }

    public function testViewPermuteAxesThenFlatten(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], DType::Float64);
        $view = $a->slice(['0:2', ':']); // [[1, 2, 3], [4, 5, 6]]
        $result = $view->permute(1, 0)->flatten();

        $this->assertSame([6], $result->shape());
        $this->assertEqualsWithDelta([1, 4, 2, 5, 3, 6], $result->toArray(), 0.0001);
    }

    public function testStridedViewPermuteAxes(): void
    {
        $a = NDArray::array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]],
            [[13, 14], [15, 16]],
        ], DType::Float64);
        $view = $a->slice(['::2', ':', ':']); // [[[1,2],[3,4]], [[9,10],[11,12]]]
        $result = $view->permute(2, 0, 1);

        // Original view shape: [2, 2, 2]
        // After permute [2, 0, 1]: [2, 2, 2]
        // permute(2,0,1): new axis 0 = old axis 2, new axis 1 = old axis 0, new axis 2 = old axis 1
        $this->assertSame([2, 2, 2], $result->shape());
        $this->assertEqualsWithDelta([
            [[1, 3], [9, 11]],
            [[2, 4], [10, 12]],
        ], $result->toArray(), 0.0001);
    }

    public function testViewPermuteAxesChaining(): void
    {
        $a = NDArray::array([
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ], DType::Float64);
        $view = $a->slice(['0:1', ':', '1:3']); // [[[2, 3], [5, 6]]]

        // Chain: permute -> insert -> flatten
        $result = $view
            ->permute(2, 1, 0)
            ->insertaxis(0)
            ->flatten();

        $this->assertSame([4], $result->shape());
        // After permute [2,1,0]: [[[2],[5]],[[3],[6]]], after insert and flatten
        $this->assertEqualsWithDelta([2, 5, 3, 6], $result->toArray(), 0.0001);
    }

    public function testViewPermuteAxesWithSqueeze(): void
    {
        $a = NDArray::array([[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], DType::Float64);
        $view = $a->slice(['0:1', ':', ':', ':']); // [[[[1,2]],[[3,4]]]]
        $permuted = $view->permute(3, 2, 1, 0);
        $squeezed = $permuted->squeeze();

        // Shape after permute: [2, 1, 2, 1]
        // After squeeze: [2, 2]
        $this->assertSame([2, 2], $squeezed->shape());
    }

    // ========================================================================
    // Tile Tests on Views
    // ========================================================================

    public function testViewTile1D(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5, 6], DType::Int64);
        $view = $a->slice(['1:4']); // [2, 3, 4]
        $result = $view->tile(2);

        $this->assertSame([6], $result->shape());
        $this->assertSame([2, 3, 4, 2, 3, 4], $result->toArray());
    }

    public function testViewTile2DRepeatRows(): void
    {
        $a = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Int64);
        $view = $a->slice(['0:2', '1:3']); // [[2, 3], [5, 6]]
        $result = $view->tile([2, 1]);

        $this->assertSame([4, 2], $result->shape());
        $this->assertSame([
            [2, 3],
            [5, 6],
            [2, 3],
            [5, 6],
        ], $result->toArray());
    }

    public function testViewTile2DRepeatCols(): void
    {
        $a = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Int64);
        $view = $a->slice(['0:2', '0:2']); // [[1, 2], [4, 5]]
        $result = $view->tile([1, 2]);

        $this->assertSame([2, 4], $result->shape());
        $this->assertSame([
            [1, 2, 1, 2],
            [4, 5, 4, 5],
        ], $result->toArray());
    }

    public function testViewTile2DRepeatBoth(): void
    {
        $a = NDArray::array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ], DType::Int64);
        $view = $a->slice(['::2', '1:4']); // [[2, 3, 4], [10, 11, 12]]
        $result = $view->tile([2, 1]);

        $this->assertSame([4, 3], $result->shape());
        $this->assertSame([
            [2, 3, 4],
            [10, 11, 12],
            [2, 3, 4],
            [10, 11, 12],
        ], $result->toArray());
    }

    public function testViewTilePreservesDtype(): void
    {
        $a = NDArray::array([[1, 2], [3, 4], [5, 6]], DType::Float64);
        $view = $a->slice(['0:2', ':']); // [[1, 2], [3, 4]]
        $result = $view->tile([1, 2]);

        $this->assertSame(DType::Float64, $result->dtype());
    }

    public function testStridedViewTile(): void
    {
        $a = NDArray::array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ], DType::Int64);
        $view = $a->slice(['::2', '::2']); // [[1, 3], [9, 11]]
        $result = $view->tile([1, 2]);

        $this->assertSame([2, 4], $result->shape());
        $this->assertSame([
            [1, 3, 1, 3],
            [9, 11, 9, 11],
        ], $result->toArray());
    }

    public function testViewTile3D(): void
    {
        $a = NDArray::array([
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            [[[9, 10], [11, 12]], [[13, 14], [15, 16]]],
        ], DType::Int64);
        $view = $a->slice(['0:1', ':', ':', ':']); // First block
        $result = $view->tile([1, 2, 1, 1]);

        $this->assertSame([1, 4, 2, 2], $result->shape());
    }

    public function testViewTileWithSlice(): void
    {
        $a = NDArray::array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
        ], DType::Int64);
        // Slice to get middle rows and columns
        $view = $a->slice(['1:3', '1:4']); // [[7, 8, 9], [12, 13, 14]]
        $result = $view->tile([2, 1]);

        $this->assertSame([4, 3], $result->shape());
        $this->assertSame([
            [7, 8, 9],
            [12, 13, 14],
            [7, 8, 9],
            [12, 13, 14],
        ], $result->toArray());
    }

    // ========================================================================
    // Repeat Tests on Views
    // ========================================================================

    public function testViewRepeat1D(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5, 6], DType::Int64);
        $view = $a->slice(['1:4']); // [2, 3, 4]
        $result = $view->repeat(2);

        $this->assertSame([6], $result->shape());
        $this->assertSame([2, 2, 3, 3, 4, 4], $result->toArray());
    }

    public function testViewRepeat2DAxis0(): void
    {
        $a = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Int64);
        $view = $a->slice(['0:2', '1:3']); // [[2, 3], [5, 6]]
        $result = $view->repeat(2, axis: 0);

        $this->assertSame([4, 2], $result->shape());
        $this->assertSame([
            [2, 3],
            [2, 3],
            [5, 6],
            [5, 6],
        ], $result->toArray());
    }

    public function testViewRepeat2DAxis1(): void
    {
        $a = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Int64);
        $view = $a->slice(['0:2', '0:2']); // [[1, 2], [4, 5]]
        $result = $view->repeat(2, axis: 1);

        $this->assertSame([2, 4], $result->shape());
        $this->assertSame([
            [1, 1, 2, 2],
            [4, 4, 5, 5],
        ], $result->toArray());
    }

    public function testViewRepeatWithArrayAxis0(): void
    {
        $a = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Int64);
        $view = $a->slice(['0:2', ':']); // [[1, 2, 3], [4, 5, 6]]
        $result = $view->repeat([1, 2], axis: 0);

        $this->assertSame([3, 3], $result->shape());
        $this->assertSame([
            [1, 2, 3],
            [4, 5, 6],
            [4, 5, 6],
        ], $result->toArray());
    }

    public function testViewRepeatWithArrayAxis1(): void
    {
        $a = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Int64);
        $view = $a->slice(['0:2', '0:2']); // [[1, 2], [4, 5]]
        $result = $view->repeat([2, 1], axis: 1);

        $this->assertSame([2, 3], $result->shape());
        $this->assertSame([
            [1, 1, 2],
            [4, 4, 5],
        ], $result->toArray());
    }

    public function testViewRepeatPreservesDtype(): void
    {
        $a = NDArray::array([[1, 2], [3, 4], [5, 6]], DType::Float64);
        $view = $a->slice(['0:2', ':']); // [[1, 2], [3, 4]]
        $result = $view->repeat(2, axis: 0);

        $this->assertSame(DType::Float64, $result->dtype());
    }

    public function testStridedViewRepeat(): void
    {
        $a = NDArray::array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ], DType::Int64);
        $view = $a->slice(['::2', '::2']); // [[1, 3], [9, 11]]
        $result = $view->repeat(2, axis: 1);

        $this->assertSame([2, 4], $result->shape());
        $this->assertSame([
            [1, 1, 3, 3],
            [9, 9, 11, 11],
        ], $result->toArray());
    }

    public function testViewRepeat3D(): void
    {
        $a = NDArray::array([
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            [[[9, 10], [11, 12]], [[13, 14], [15, 16]]],
        ], DType::Int64);
        $view = $a->slice(['0:1', ':', ':', ':']); // First block
        $result = $view->repeat(2, axis: 1);

        $this->assertSame([1, 4, 2, 2], $result->shape());
    }

    public function testViewRepeatWithSlice(): void
    {
        $a = NDArray::array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
        ], DType::Int64);
        // Slice to get middle rows and columns
        $view = $a->slice(['1:3', '1:4']); // [[7, 8, 9], [12, 13, 14]]
        $result = $view->repeat(2, axis: 0);

        $this->assertSame([4, 3], $result->shape());
        $this->assertSame([
            [7, 8, 9],
            [7, 8, 9],
            [12, 13, 14],
            [12, 13, 14],
        ], $result->toArray());
    }

    public function testViewRepeatOnceReturnsSame(): void
    {
        $a = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Int64);
        $view = $a->slice(['0:2', '0:2']); // [[1, 2], [4, 5]]
        $result = $view->repeat(1, axis: 0);

        $this->assertSame([2, 2], $result->shape());
        $this->assertSame([
            [1, 2],
            [4, 5],
        ], $result->toArray());
    }

    // ========================================================================
    // Combined Operations Tests
    // ========================================================================

    public function testViewTileThenFlatten(): void
    {
        $a = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Int64);
        $view = $a->slice(['0:2', '0:2']); // [[1, 2], [4, 5]]
        $result = $view->tile([1, 2])->flatten();

        $this->assertSame([8], $result->shape());
        $this->assertSame([1, 2, 1, 2, 4, 5, 4, 5], $result->toArray());
    }

    public function testViewRepeatThenFlatten(): void
    {
        $a = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Int64);
        $view = $a->slice(['0:2', '0:2']); // [[1, 2], [4, 5]]
        $result = $view->repeat(2, axis: 1)->flatten();

        $this->assertSame([8], $result->shape());
        $this->assertSame([1, 1, 2, 2, 4, 4, 5, 5], $result->toArray());
    }

    public function testViewTileThenTranspose(): void
    {
        $a = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Int64);
        $view = $a->slice(['0:2', '0:2']); // [[1, 2], [4, 5]]
        $result = $view->tile([2, 1])->transpose();

        $this->assertSame([2, 4], $result->shape());
    }

    public function testViewRepeatThenTranspose(): void
    {
        $a = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Int64);
        $view = $a->slice(['0:2', '0:2']); // [[1, 2], [4, 5]]
        $result = $view->repeat(2, axis: 0)->transpose();

        $this->assertSame([2, 4], $result->shape());
    }
}
