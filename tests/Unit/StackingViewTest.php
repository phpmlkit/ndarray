<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Unit;

use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Tests for joining and splitting operations on views and slices.
 *
 * @internal
 *
 * @coversNothing
 */
final class StackingViewTest extends TestCase
{
    // =========================================================================
    // Concatenate with Views
    // =========================================================================

    public function testConcatenateWithSlices(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5, 6], DType::Float64);
        $slice1 = $a->slice(['0:3']);  // [1, 2, 3]
        $slice2 = $a->slice(['3:6']);  // [4, 5, 6]

        $result = NDArray::concatenate([$slice1, $slice2], 0);

        $this->assertSame([6], $result->shape());
        $this->assertEqualsWithDelta([1, 2, 3, 4, 5, 6], $result->toArray(), 0.0001);
    }

    public function testConcatenateViewWithFullArray(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $b = NDArray::array([4, 5, 6, 7], DType::Float64);
        $view = $b->slice(['0:2']);  // [4, 5]

        $result = NDArray::concatenate([$a, $view], 0);

        $this->assertSame([5], $result->shape());
        $this->assertEqualsWithDelta([1, 2, 3, 4, 5], $result->toArray(), 0.0001);
    }

    public function testConcatenate2DRowSlices(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], DType::Float64);
        $rows1 = $a->slice(['0:2', ':']);  // rows 0-1
        $rows2 = $a->slice(['2:3', ':']);  // row 2

        $result = NDArray::concatenate([$rows1, $rows2], 0);

        $this->assertSame([3, 3], $result->shape());
        $this->assertEqualsWithDelta(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            $result->toArray(),
            0.0001
        );
    }

    // =========================================================================
    // Stack with Views
    // =========================================================================

    public function testStackWithSlices(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5, 6], DType::Float64);
        $slice1 = $a->slice(['0:3']);  // [1, 2, 3]
        $slice2 = $a->slice(['3:6']);  // [4, 5, 6]

        $result = NDArray::stack([$slice1, $slice2], 0);

        $this->assertSame([2, 3], $result->shape());
        $this->assertEqualsWithDelta([[1, 2, 3], [4, 5, 6]], $result->toArray(), 0.0001);
    }

    public function testStackStridedViews(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5, 6], DType::Float64);
        $evens = $a->slice(['::2']);  // [1, 3, 5]
        $odds = $a->slice(['1::2']);  // [2, 4, 6]

        $result = NDArray::stack([$evens, $odds], 0);

        $this->assertSame([2, 3], $result->shape());
        $this->assertEqualsWithDelta([[1, 3, 5], [2, 4, 6]], $result->toArray(), 0.0001);
    }

    // =========================================================================
    // Split on Views
    // =========================================================================

    public function testSplitOnSlice(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5, 6, 7, 8], DType::Float64);
        $slice = $a->slice(['2:6']);  // [3, 4, 5, 6]

        $parts = $slice->split(2, 0);

        $this->assertCount(2, $parts);
        $this->assertEqualsWithDelta([3, 4], $parts[0]->toArray(), 0.0001);
        $this->assertEqualsWithDelta([5, 6], $parts[1]->toArray(), 0.0001);
    }

    public function testSplitViewSharesBase(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5, 6], DType::Float64);
        $parts = $a->split([2, 4], 0);

        $this->assertTrue($parts[0]->isView());
        $this->assertTrue($parts[1]->isView());
        $this->assertTrue($parts[2]->isView());

        $this->assertEqualsWithDelta([1, 2], $parts[0]->toArray(), 0.0001);
        $this->assertEqualsWithDelta([3, 4], $parts[1]->toArray(), 0.0001);
        $this->assertEqualsWithDelta([5, 6], $parts[2]->toArray(), 0.0001);
    }

    public function testVsplitOn2DView(): void
    {
        $a = NDArray::array([
            [1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
        ], DType::Float64);
        $view = $a->slice(['1:4', ':']);  // rows 1-3

        $parts = $view->vsplit(3);

        $this->assertCount(3, $parts);
        $this->assertEqualsWithDelta([[3, 4]], $parts[0]->toArray(), 0.0001);
        $this->assertEqualsWithDelta([[5, 6]], $parts[1]->toArray(), 0.0001);
        $this->assertEqualsWithDelta([[7, 8]], $parts[2]->toArray(), 0.0001);
    }

    public function testHsplitOn2DView(): void
    {
        $a = NDArray::array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ], DType::Float64);
        $view = $a->slice([':', '1:3']);  // columns 1-2

        $parts = $view->hsplit(2);

        $this->assertCount(2, $parts);
        $this->assertEqualsWithDelta([[2], [6]], $parts[0]->toArray(), 0.0001);
        $this->assertEqualsWithDelta([[3], [7]], $parts[1]->toArray(), 0.0001);
    }
}
