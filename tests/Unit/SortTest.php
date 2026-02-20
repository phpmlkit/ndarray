<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Unit;

use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\Exceptions\NDArrayException;
use PhpMlKit\NDArray\NDArray;
use PhpMlKit\NDArray\SortKind;
use PHPUnit\Framework\TestCase;

/**
 * Tests for sort/argsort operations.
 *
 * @internal
 *
 * @coversNothing
 */
final class SortTest extends TestCase
{
    public function testSortDefaultLastAxis(): void
    {
        $a = NDArray::array([[3, 1, 2], [6, 5, 4]], DType::Int32);
        $result = $a->sort();

        $this->assertSame([2, 3], $result->shape());
        $this->assertSame([[1, 2, 3], [4, 5, 6]], $result->toArray());
        $this->assertSame(DType::Int32, $result->dtype());
    }

    public function testSortAxisNullFlattens(): void
    {
        $a = NDArray::array([[3, 1, 2], [6, 5, 4]], DType::Int32);
        $result = $a->sort(axis: null);

        $this->assertSame([6], $result->shape());
        $this->assertSame([1, 2, 3, 4, 5, 6], $result->toArray());
    }

    public function testArgsortDefaultLastAxis(): void
    {
        $a = NDArray::array([[30, 10, 20], [60, 50, 40]], DType::Int32);
        $result = $a->argsort();

        $this->assertSame(DType::Int64, $result->dtype());
        $this->assertSame([[1, 2, 0], [2, 1, 0]], $result->toArray());
    }

    public function testArgsortAxisNullFlattens(): void
    {
        $a = NDArray::array([[3, 1, 2], [6, 5, 4]], DType::Int32);
        $result = $a->argsort(axis: null);

        $this->assertSame([6], $result->shape());
        $this->assertSame([1, 2, 0, 5, 4, 3], $result->toArray());
    }

    public function testSortAxisNegativeOneOn3D(): void
    {
        $a = NDArray::array([
            [[3, 1, 2], [9, 8, 7]],
            [[6, 5, 4], [0, -1, -2]],
        ], DType::Int32);

        $result = $a->sort(axis: -1);

        $this->assertSame([2, 2, 3], $result->shape());
        $this->assertSame([
            [[1, 2, 3], [7, 8, 9]],
            [[4, 5, 6], [-2, -1, 0]],
        ], $result->toArray());
    }

    public function testSortAllKindsProduceSortedOutput(): void
    {
        $a = NDArray::array([[3, 1, 2], [6, 5, 4]], DType::Int32);
        $expected = [[1, 2, 3], [4, 5, 6]];

        $kinds = [
            SortKind::QuickSort,
            SortKind::MergeSort,
            SortKind::HeapSort,
            SortKind::Stable,
        ];

        foreach ($kinds as $kind) {
            $result = $a->sort(kind: $kind);
            $this->assertSame($expected, $result->toArray(), "kind={$kind->name} failed");
        }
    }

    public function testStableArgsortPreservesEqualOrder(): void
    {
        $a = NDArray::array([2, 1, 1, 3], DType::Int32);
        $result = $a->argsort(kind: SortKind::Stable);

        // equal values at indices 1 and 2 should keep original order
        $this->assertSame([1, 2, 0, 3], $result->toArray());
    }

    public function testSortNaNPlacedAtEnd(): void
    {
        $a = NDArray::array([\NAN, 2.0, 1.0, \NAN], DType::Float64);
        $result = $a->sort(axis: null);
        $out = $result->toArray();

        $this->assertSame([4], $result->shape());
        $this->assertEqualsWithDelta(1.0, $out[0], 0.0001);
        $this->assertEqualsWithDelta(2.0, $out[1], 0.0001);
        $this->assertTrue(is_nan($out[2]));
        $this->assertTrue(is_nan($out[3]));
    }

    public function testBoolSortOrdering(): void
    {
        $a = NDArray::array([true, false, true, false], DType::Bool);
        $result = $a->sort(axis: null);

        $this->assertSame([false, false, true, true], $result->toArray());
    }

    public function testArgsortReconstructsSortedValues(): void
    {
        $a = NDArray::array([7, 1, 4, 4, 3], DType::Int32);
        $indices = $a->argsort(axis: null, kind: SortKind::MergeSort)->toArray();
        $values = $a->toArray();

        $reconstructed = [];
        foreach ($indices as $i) {
            $reconstructed[] = $values[$i];
        }

        $this->assertSame([1, 3, 4, 4, 7], $reconstructed);
    }

    public function testSortAxisOutOfBoundsThrows(): void
    {
        $a = NDArray::array([[1, 2, 3]], DType::Int32);
        $this->expectException(NDArrayException::class);
        $a->sort(axis: 2);
    }

    public function testTopkDefaultLargestAlongLastAxis(): void
    {
        $a = NDArray::array([[1, 7, 3, 5], [9, 2, 8, 4]], DType::Int32);
        $topk = $a->topk(2);

        $this->assertSame([2, 2], $topk['values']->shape());
        $this->assertSame([[7, 5], [9, 8]], $topk['values']->toArray());
        $this->assertSame([[1, 3], [0, 2]], $topk['indices']->toArray());
        $this->assertSame(DType::Int64, $topk['indices']->dtype());
    }

    public function testTopkSmallestAlongAxis(): void
    {
        $a = NDArray::array([[1, 7, 3, 5], [9, 2, 8, 4]], DType::Int32);
        $topk = $a->topk(2, axis: 1, largest: false, kind: SortKind::MergeSort);

        $this->assertSame([[1, 3], [2, 4]], $topk['values']->toArray());
        $this->assertSame([[0, 2], [1, 3]], $topk['indices']->toArray());
    }

    public function testTopkFlattened(): void
    {
        $a = NDArray::array([[1, 7, 3], [5, 9, 2]], DType::Int32);
        $topk = $a->topk(3, axis: null, kind: SortKind::HeapSort);

        $this->assertSame([3], $topk['values']->shape());
        $this->assertSame([9, 7, 5], $topk['values']->toArray());
        $this->assertSame([4, 1, 3], $topk['indices']->toArray());
    }

    public function testTopkAllKindsConsistentValues(): void
    {
        $a = NDArray::array([4, 1, 7, 3, 6, 2], DType::Int32);
        $expected = [7, 6, 4];
        $kinds = [SortKind::QuickSort, SortKind::MergeSort, SortKind::HeapSort, SortKind::Stable];

        foreach ($kinds as $kind) {
            $topk = $a->topk(3, axis: null, kind: $kind);
            $this->assertSame($expected, $topk['values']->toArray(), "kind={$kind->name} failed");
        }
    }

    public function testTopkSortedFalseReturnsSelectionInInputOrder(): void
    {
        $a = NDArray::array([4, 1, 7, 3, 6, 2], DType::Int32);
        $topk = $a->topk(3, axis: null, largest: true, sorted: false);

        // top-3 set is {7, 6, 4}; sorted=false returns them in source index order
        $this->assertSame([4, 7, 6], $topk['values']->toArray());
        $this->assertSame([0, 2, 4], $topk['indices']->toArray());
    }

    public function testTopkOutOfBoundsThrows(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Int32);
        $this->expectException(NDArrayException::class);
        $a->topk(4);
    }
}
