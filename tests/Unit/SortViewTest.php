<?php

declare(strict_types=1);

namespace NDArray\Tests\Unit;

use NDArray\DType;
use NDArray\NDArray;
use NDArray\SortKind;
use PHPUnit\Framework\TestCase;

/**
 * Tests for sort/argsort operations on views and slices.
 */
final class SortViewTest extends TestCase
{
    public function testSortOn1DView(): void
    {
        $a = NDArray::array([9, 3, 7, 1, 5], DType::Int32);
        $view = $a->slice(['1:5']); // [3, 7, 1, 5]

        $result = $view->sort(axis: null);

        $this->assertSame([4], $result->shape());
        $this->assertSame([1, 3, 5, 7], $result->toArray());
    }

    public function testSortOn2DViewLastAxis(): void
    {
        $a = NDArray::array([
            [9, 1, 5, 3],
            [8, 2, 6, 4],
            [7, 0, 11, 10],
        ], DType::Int32);
        $view = $a->slice(['0:2', '1:4']); // [[1, 5, 3], [2, 6, 4]]

        $result = $view->sort(axis: -1, kind: SortKind::HeapSort);

        $this->assertSame([2, 3], $result->shape());
        $this->assertSame([[1, 3, 5], [2, 4, 6]], $result->toArray());
    }

    public function testArgsortOn2DStridedView(): void
    {
        $a = NDArray::array([
            [7, 2, 9, 1],
            [6, 3, 8, 0],
            [5, 4, 11, 10],
        ], DType::Int32);
        $view = $a->slice(['::2', '0:4:2']); // [[7, 9], [5, 11]]

        $result = $view->argsort(axis: -1);

        $this->assertSame([2, 2], $result->shape());
        $this->assertSame([[0, 1], [0, 1]], $result->toArray());
    }

    public function testArgsortOnFlatView(): void
    {
        $a = NDArray::array([10, 3, 8, 1, 5, 2], DType::Int32);
        $view = $a->slice(['1:6:2']); // [3, 1, 2]

        $result = $view->argsort(axis: null, kind: SortKind::Stable);

        $this->assertSame([3], $result->shape());
        $this->assertSame([1, 2, 0], $result->toArray());
    }

    public function testSortNaNOnView(): void
    {
        $a = NDArray::array([5.0, NAN, 2.0, NAN, 1.0], DType::Float64);
        $view = $a->slice(['1:5']); // [NaN, 2, NaN, 1]

        $result = $view->sort(axis: null, kind: SortKind::MergeSort)->toArray();

        $this->assertEqualsWithDelta(1.0, $result[0], 0.0001);
        $this->assertEqualsWithDelta(2.0, $result[1], 0.0001);
        $this->assertTrue(is_nan($result[2]));
        $this->assertTrue(is_nan($result[3]));
    }

    public function testBoolSortOnView(): void
    {
        $a = NDArray::array([true, false, true, false, true], DType::Bool);
        $view = $a->slice(['1:5']); // [false, true, false, true]

        $result = $view->sort(axis: null);

        $this->assertSame([false, false, true, true], $result->toArray());
    }
}
