<?php

declare(strict_types=1);

namespace NDArray\Tests\Unit;

use NDArray\DType;
use NDArray\Exceptions\IndexException;
use NDArray\Exceptions\ShapeException;
use NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Tests for NDArray slicing and advanced assignment.
 */
final class SlicingTest extends TestCase
{
    // =========================================================================
    // Basic Slicing
    // =========================================================================

    public function testSlice1D(): void
    {
        $arr = NDArray::arange(0, 10); // [0, 1, ..., 9]

        // 0:5
        $slice = $arr->slice(['0:5']);
        $this->assertSame([5], $slice->shape());
        $this->assertSame([0, 1, 2, 3, 4], $slice->toArray());

        // 2: (to end)
        $slice = $arr->slice(['2:']);
        $this->assertSame([8], $slice->shape());
        $this->assertSame([2, 3, 4, 5, 6, 7, 8, 9], $slice->toArray());

        // :3 (from start)
        $slice = $arr->slice([':3']);
        $this->assertSame([3], $slice->shape());
        $this->assertSame([0, 1, 2], $slice->toArray());

        // : (all)
        $slice = $arr->slice([':']);
        $this->assertSame([10], $slice->shape());
        $this->assertSame($arr->toArray(), $slice->toArray());
    }

    public function testSlice2D(): void
    {
        $arr = NDArray::array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ]); // Shape [3, 4]

        // Row slice 0:2 -> rows 0, 1
        $slice = $arr->slice(['0:2']);
        $this->assertSame([2, 4], $slice->shape());
        $this->assertSame([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ], $slice->toArray());

        // Col slice :2 -> cols 0, 1 for all rows
        $slice = $arr->slice([':', '0:2']);
        $this->assertSame([3, 2], $slice->shape());
        $this->assertSame([
            [1, 2],
            [5, 6],
            [9, 10],
        ], $slice->toArray());

        // Sub-block 1:3, 1:3
        $slice = $arr->slice(['1:3', '1:3']);
        $this->assertSame([2, 2], $slice->shape());
        $this->assertSame([
            [6, 7],
            [10, 11],
        ], $slice->toArray());
    }

    // =========================================================================
    // Step Slicing
    // =========================================================================

    public function testSliceStep1D(): void
    {
        $arr = NDArray::arange(0, 10);

        // ::2 (evens)
        $slice = $arr->slice(['::2']);
        $this->assertSame([5], $slice->shape());
        $this->assertSame([0, 2, 4, 6, 8], $slice->toArray());

        // 1::2 (odds)
        $slice = $arr->slice(['1::2']);
        $this->assertSame([5], $slice->shape());
        $this->assertSame([1, 3, 5, 7, 9], $slice->toArray());

        // 1:8:3 (1, 4, 7)
        $slice = $arr->slice(['1:8:3']);
        $this->assertSame([3], $slice->shape());
        $this->assertSame([1, 4, 7], $slice->toArray());
    }

    public function testSliceStep2D(): void
    {
        $arr = NDArray::array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]);

        // Rows ::2, Cols ::2
        $slice = $arr->slice(['::2', '::2']);
        $this->assertSame([2, 2], $slice->shape());
        $this->assertSame([
            [1, 3],
            [9, 11],
        ], $slice->toArray());
    }

    // =========================================================================
    // Mixed Indexing (Int + Slice)
    // =========================================================================

    public function testMixedIndexing(): void
    {
        $arr = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]);

        // Row 1, Cols 0:2
        $slice = $arr->slice([1, '0:2']);
        $this->assertSame([2], $slice->shape());
        $this->assertSame([4, 5], $slice->toArray());

        // Rows 0:2, Col 1
        $slice = $arr->slice(['0:2', 1]);
        $this->assertSame([2], $slice->shape());
        $this->assertSame([2, 5], $slice->toArray());
    }

    // =========================================================================
    // ArrayAccess Syntax
    // =========================================================================

    public function testArrayAccessSlice(): void
    {
        $arr = NDArray::arange(10);

        // $arr['0:3']
        $slice = $arr['0:3'];
        $this->assertInstanceOf(NDArray::class, $slice);
        $this->assertSame([0, 1, 2], $slice->toArray());

        // $arr['::3']
        $slice = $arr['::3'];
        $this->assertSame([0, 3, 6, 9], $slice->toArray());
    }

    public function testArrayAccessMultiDimSlice(): void
    {
        $arr = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
        ]);

        // $arr[':, 1'] -> column 1
        $col = $arr[':, 1'];
        $this->assertSame([2], $col->shape());
        $this->assertSame([2, 5], $col->toArray());

        // $arr['0, 1:'] -> row 0, cols 1:
        $sub = $arr['0, 1:'];
        $this->assertSame([2], $sub->shape());
        $this->assertSame([2, 3], $sub->toArray());
    }

    // =========================================================================
    // Negative Indices
    // =========================================================================

    public function testSliceNegativeIndices(): void
    {
        $arr = NDArray::arange(10);

        // :-1 (all except last)
        $slice = $arr->slice([':-1']);
        $this->assertSame([9], $slice->shape());
        $this->assertSame([0, 1, 2, 3, 4, 5, 6, 7, 8], $slice->toArray());

        // -3: (last 3)
        $slice = $arr->slice(['-3:']);
        $this->assertSame([3], $slice->shape());
        $this->assertSame([7, 8, 9], $slice->toArray());

        // -3:-1
        $slice = $arr->slice(['-3:-1']);
        $this->assertSame([2], $slice->shape());
        $this->assertSame([7, 8], $slice->toArray());
    }

    // =========================================================================
    // Assignment (assign)
    // =========================================================================

    public function testAssignScalarToSlice(): void
    {
        $arr = NDArray::zeros([5]);
        $arr->slice(['1:4'])->assign(9);

        $this->assertSame([0.0, 9.0, 9.0, 9.0, 0.0], $arr->toArray());
    }

    public function testAssignArrayToSlice(): void
    {
        $arr = NDArray::zeros([5]);
        $arr->slice(['1:4'])->assign([10, 20, 30]);

        $this->assertSame([0.0, 10.0, 20.0, 30.0, 0.0], $arr->toArray());
    }

    public function testAssignArrayTo2DSlice(): void
    {
        $arr = NDArray::zeros([3, 3]);
        // Set center to 1
        $arr->slice(['1:2', '1:2'])->assign(1);
        
        $expected = [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        $this->assertSame($expected, $arr->toArray());

        // Set top-left 2x2 to [[1,2],[3,4]]
        $arr->slice([':2', ':2'])->assign([[1, 2], [3, 4]]);
        $expected2 = [
            [1.0, 2.0, 0.0],
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        $this->assertSame($expected2, $arr->toArray());
    }

    public function testArrayAccessSliceAssignment(): void
    {
        $arr = NDArray::zeros([5], DType::Int32);
        
        // $arr['1:3'] = 5
        $arr['1:3'] = 5;
        $this->assertSame([0, 5, 5, 0, 0], $arr->toArray());

        // $arr['3:'] = [8, 9]
        $arr['3:'] = [8, 9];
        $this->assertSame([0, 5, 5, 8, 9], $arr->toArray());
    }

    public function testArrayAccessPartialAssignment(): void
    {
        $arr = NDArray::zeros([2, 2], DType::Int32);
        
        // $arr[0] = [1, 2]
        $arr[0] = [1, 2];
        
        $this->assertSame([
            [1, 2],
            [0, 0],
        ], $arr->toArray());

        // $arr[1] = 9 (scalar broadcast)
        $arr[1] = 9;
        
        $this->assertSame([
            [1, 2],
            [9, 9],
        ], $arr->toArray());
    }

    // =========================================================================
    // Error Cases
    // =========================================================================

    public function testAssignShapeMismatchThrows(): void
    {
        $arr = NDArray::zeros([5]);
        
        $this->expectException(ShapeException::class);
        $this->expectExceptionMessage('Cannot assign array of size 2 to view of size 3');
        
        $arr->slice(['0:3'])->assign([1, 2]);
    }

    public function testSliceStepZeroThrows(): void
    {
        $arr = NDArray::zeros([5]);
        
        $this->expectException(IndexException::class);
        $this->expectExceptionMessage('step cannot be zero');
        
        $arr->slice(['::0']);
    }

    public function testInvalidSelectorThrows(): void
    {
        $arr = NDArray::zeros([5]);
        
        $this->expectException(IndexException::class);
        $this->expectExceptionMessage('Invalid slice component');
        
        $arr['invalid:selector'];
    }
}
