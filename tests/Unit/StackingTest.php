<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Unit;

use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\Exceptions\DTypeException;
use PhpMlKit\NDArray\Exceptions\IndexException;
use PhpMlKit\NDArray\Exceptions\ShapeException;
use PhpMlKit\NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Tests for joining and splitting operations on full arrays.
 *
 * @internal
 *
 * @coversNothing
 */
final class StackingTest extends TestCase
{
    // =========================================================================
    // Concatenate
    // =========================================================================

    public function testConcatenate1D(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $b = NDArray::array([4, 5, 6], DType::Float64);

        $result = NDArray::concatenate([$a, $b], 0);

        $this->assertSame([6], $result->shape());
        $this->assertEqualsWithDelta([1, 2, 3, 4, 5, 6], $result->toArray(), 0.0001);
    }

    public function testConcatenate2DAxis0(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);
        $b = NDArray::array([[5, 6], [7, 8]], DType::Float64);

        $result = NDArray::concatenate([$a, $b], 0);

        $this->assertSame([4, 2], $result->shape());
        $this->assertEqualsWithDelta(
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            $result->toArray(),
            0.0001
        );
    }

    public function testConcatenate2DAxis1(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);
        $b = NDArray::array([[5, 6], [7, 8]], DType::Float64);

        $result = NDArray::concatenate([$a, $b], 1);

        $this->assertSame([2, 4], $result->shape());
        $this->assertEqualsWithDelta(
            [[1, 2, 5, 6], [3, 4, 7, 8]],
            $result->toArray(),
            0.0001
        );
    }

    public function testConcatenateThreeArrays(): void
    {
        $a = NDArray::array([1, 2], DType::Float64);
        $b = NDArray::array([3, 4], DType::Float64);
        $c = NDArray::array([5, 6], DType::Float64);

        $result = NDArray::concatenate([$a, $b, $c], 0);

        $this->assertSame([6], $result->shape());
        $this->assertEqualsWithDelta([1, 2, 3, 4, 5, 6], $result->toArray(), 0.0001);
    }

    public function testConcatenateSingleArray(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $result = NDArray::concatenate([$a], 0);
        $this->assertSame([3], $result->shape());
        $this->assertEqualsWithDelta([1, 2, 3], $result->toArray(), 0.0001);
    }

    public function testConcatenateEmptyArraysThrows(): void
    {
        $this->expectException(ShapeException::class);
        $this->expectExceptionMessage('concatenate requires at least one array');
        NDArray::concatenate([], 0);
    }

    public function testConcatenateAxisOutOfBoundsThrows(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $b = NDArray::array([4, 5, 6], DType::Float64);

        $this->expectException(ShapeException::class);
        $this->expectExceptionMessage('Axis 1 out of bounds');
        NDArray::concatenate([$a, $b], 1);
    }

    public function testConcatenateAxisNegativeOutOfBoundsThrows(): void
    {
        $a = NDArray::array([1, 2], DType::Float64);
        $b = NDArray::array([3, 4], DType::Float64);

        $this->expectException(ShapeException::class);
        $this->expectExceptionMessage('Axis -3 out of bounds');
        NDArray::concatenate([$a, $b], -3);
    }

    public function testConcatenateIncompatibleShapesThrows(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);  // 2x2
        $b = NDArray::array([[5, 6, 7], [8, 9, 10]], DType::Float64);  // 2x3 - different cols

        $this->expectException(ShapeException::class);
        NDArray::concatenate([$a, $b], 0);
    }

    public function testConcatenateIncompatibleShapesAxis1Throws(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);  // 2x2
        $b = NDArray::array([[5, 6], [7, 8], [9, 10]], DType::Float64);  // 3x2 - different rows

        $this->expectException(ShapeException::class);
        NDArray::concatenate([$a, $b], 1);
    }

    public function testConcatenateDifferentNdimsThrows(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);  // 1D
        $b = NDArray::array([[4, 5, 6]], DType::Float64);  // 2D

        $this->expectException(ShapeException::class);
        $this->expectExceptionMessage('same number of dimensions');
        NDArray::concatenate([$a, $b], 0);
    }

    public function testConcatenateDifferentDtypesThrows(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $b = NDArray::array([4, 5, 6], DType::Int64);

        $this->expectException(DTypeException::class);
        NDArray::concatenate([$a, $b], 0);
    }

    // =========================================================================
    // Stack
    // =========================================================================

    public function testStack1D(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $b = NDArray::array([4, 5, 6], DType::Float64);

        $result = NDArray::stack([$a, $b], 0);

        $this->assertSame([2, 3], $result->shape());
        $this->assertEqualsWithDelta([[1, 2, 3], [4, 5, 6]], $result->toArray(), 0.0001);
    }

    public function testStack2DAxis0(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);
        $b = NDArray::array([[5, 6], [7, 8]], DType::Float64);

        $result = NDArray::stack([$a, $b], 0);

        $this->assertSame([2, 2, 2], $result->shape());
        $this->assertEqualsWithDelta(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            $result->toArray(),
            0.0001
        );
    }

    public function testStackThreeArrays(): void
    {
        $a = NDArray::array([1, 2], DType::Float64);
        $b = NDArray::array([3, 4], DType::Float64);
        $c = NDArray::array([5, 6], DType::Float64);

        $result = NDArray::stack([$a, $b, $c], 0);

        $this->assertSame([3, 2], $result->shape());
        $this->assertEqualsWithDelta([[1, 2], [3, 4], [5, 6]], $result->toArray(), 0.0001);
    }

    public function testStackEmptyArraysThrows(): void
    {
        $this->expectException(ShapeException::class);
        $this->expectExceptionMessage('stack requires at least one array');
        NDArray::stack([], 0);
    }

    public function testStackAxisOutOfBoundsThrows(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $b = NDArray::array([4, 5, 6], DType::Float64);

        $this->expectException(ShapeException::class);
        $this->expectExceptionMessage('Axis 2 out of bounds');
        NDArray::stack([$a, $b], 2);
    }

    public function testStackIncompatibleShapesThrows(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);  // length 3
        $b = NDArray::array([4, 5, 6, 7], DType::Float64);  // length 4

        $this->expectException(ShapeException::class);
        NDArray::stack([$a, $b], 0);
    }

    public function testStackIncompatible2DShapesThrows(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);  // 2x2
        $b = NDArray::array([[5, 6], [7, 8], [9, 10]], DType::Float64);  // 3x2

        $this->expectException(ShapeException::class);
        NDArray::stack([$a, $b], 0);
    }

    public function testStackDifferentNdimsThrows(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);  // 1D
        $b = NDArray::array([[4, 5, 6]], DType::Float64);  // 2D

        $this->expectException(ShapeException::class);
        $this->expectExceptionMessage('same number of dimensions');
        NDArray::stack([$a, $b], 0);
    }

    public function testStackDifferentDtypesThrows(): void
    {
        $a = NDArray::array([1, 2], DType::Float64);
        $b = NDArray::array([3, 4], DType::Float32);

        $this->expectException(DTypeException::class);
        NDArray::stack([$a, $b], 0);
    }

    // =========================================================================
    // VStack / HStack
    // =========================================================================

    public function testVstack(): void
    {
        $a = NDArray::array([[1, 2]], DType::Float64);
        $b = NDArray::array([[3, 4]], DType::Float64);

        $result = NDArray::vstack([$a, $b]);

        $this->assertSame([2, 2], $result->shape());
        $this->assertEqualsWithDelta([[1, 2], [3, 4]], $result->toArray(), 0.0001);
    }

    public function testHstack(): void
    {
        $a = NDArray::array([[1], [2]], DType::Float64);
        $b = NDArray::array([[3], [4]], DType::Float64);

        $result = NDArray::hstack([$a, $b]);

        $this->assertSame([2, 2], $result->shape());
        $this->assertEqualsWithDelta([[1, 3], [2, 4]], $result->toArray(), 0.0001);
    }

    // =========================================================================
    // Split
    // =========================================================================

    public function testSplitEqualSections(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5, 6], DType::Float64);
        $parts = $a->split(3, 0);

        $this->assertCount(3, $parts);
        $this->assertEqualsWithDelta([1, 2], $parts[0]->toArray(), 0.0001);
        $this->assertEqualsWithDelta([3, 4], $parts[1]->toArray(), 0.0001);
        $this->assertEqualsWithDelta([5, 6], $parts[2]->toArray(), 0.0001);
    }

    public function testSplitAtIndices(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5, 6], DType::Float64);
        $parts = $a->split([2, 4], 0);

        $this->assertCount(3, $parts);
        $this->assertEqualsWithDelta([1, 2], $parts[0]->toArray(), 0.0001);
        $this->assertEqualsWithDelta([3, 4], $parts[1]->toArray(), 0.0001);
        $this->assertEqualsWithDelta([5, 6], $parts[2]->toArray(), 0.0001);
    }

    public function testSplit2DAxis1(): void
    {
        $a = NDArray::array([[1, 2, 3, 4], [5, 6, 7, 8]], DType::Float64);
        $parts = $a->split(2, 1);

        $this->assertCount(2, $parts);
        $this->assertEqualsWithDelta([[1, 2], [5, 6]], $parts[0]->toArray(), 0.0001);
        $this->assertEqualsWithDelta([[3, 4], [7, 8]], $parts[1]->toArray(), 0.0001);
    }

    public function testVsplit(): void
    {
        $a = NDArray::array([[1, 2], [3, 4], [5, 6]], DType::Float64);
        $parts = $a->vsplit(3);

        $this->assertCount(3, $parts);
        $this->assertEqualsWithDelta([[1, 2]], $parts[0]->toArray(), 0.0001);
        $this->assertEqualsWithDelta([[3, 4]], $parts[1]->toArray(), 0.0001);
        $this->assertEqualsWithDelta([[5, 6]], $parts[2]->toArray(), 0.0001);
    }

    public function testHsplit(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $parts = $a->hsplit(3);

        $this->assertCount(3, $parts);
        $this->assertEqualsWithDelta([[1], [4]], $parts[0]->toArray(), 0.0001);
        $this->assertEqualsWithDelta([[2], [5]], $parts[1]->toArray(), 0.0001);
        $this->assertEqualsWithDelta([[3], [6]], $parts[2]->toArray(), 0.0001);
    }

    // =========================================================================
    // Negative: Split
    // =========================================================================

    public function testSplitSectionsLessThanOneThrows(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5, 6], DType::Float64);

        $this->expectException(ShapeException::class);
        $this->expectExceptionMessage('Number of sections must be >= 1');
        $a->split(0, 0);
    }

    public function testSplitUnequalDivisionThrows(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5, 6], DType::Float64);

        $this->expectException(ShapeException::class);
        $this->expectExceptionMessage('not divisible by');
        $a->split(4, 0);  // 6 elements into 4 parts
    }

    public function testSplitIndexOutOfBoundsThrows(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5, 6], DType::Float64);

        $this->expectException(IndexException::class);
        $this->expectExceptionMessage('out of bounds');
        $a->split([10], 0);  // index 10 > axis length 6
    }

    public function testSplitIndexExceedsAxisLengthThrows(): void
    {
        $a = NDArray::array([1, 2, 3, 4], DType::Float64);

        $this->expectException(IndexException::class);
        $a->split([2, 5], 0);  // 5 > 4
    }

    public function testSplitIndicesNotAscendingThrows(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5, 6], DType::Float64);

        $this->expectException(IndexException::class);
        $a->split([4, 2], 0);  // 2 < 4, not ascending
    }

    public function testSplitAxisOutOfBoundsThrows(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5, 6], DType::Float64);

        $this->expectException(ShapeException::class);
        $this->expectExceptionMessage('Axis 1 out of bounds');
        $a->split(2, 1);  // 1D array has only axis 0
    }

    public function testVsplitUnequalDivisionThrows(): void
    {
        $a = NDArray::array([[1, 2], [3, 4], [5, 6]], DType::Float64);  // 3 rows

        $this->expectException(ShapeException::class);
        $this->expectExceptionMessage('not divisible by');
        $a->vsplit(2);  // 3 rows not divisible by 2
    }

    public function testHsplitIndexOutOfBoundsThrows(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);  // 3 cols

        $this->expectException(IndexException::class);
        $a->hsplit([5]);  // index 5 > 3 cols
    }
}
