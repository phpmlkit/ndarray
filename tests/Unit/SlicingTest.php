<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Unit;

use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\Exceptions\IndexException;
use PhpMlKit\NDArray\Exceptions\ShapeException;
use PhpMlKit\NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Tests for NDArray slicing and advanced assignment.
 *
 * @internal
 *
 * @coversNothing
 */
final class SlicingTest extends TestCase
{
    public function testSlice1D(): void
    {
        $arr = NDArray::arange(0, 10);

        $slice = $arr->slice(['0:5']);
        $this->assertSame([5], $slice->shape());
        $this->assertSame([0, 1, 2, 3, 4], $slice->toArray());

        $slice = $arr->slice(['2:']);
        $this->assertSame([8], $slice->shape());
        $this->assertSame([2, 3, 4, 5, 6, 7, 8, 9], $slice->toArray());

        $slice = $arr->slice([':3']);
        $this->assertSame([3], $slice->shape());
        $this->assertSame([0, 1, 2], $slice->toArray());

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
        ]);

        $slice = $arr->slice(['0:2']);
        $this->assertSame([2, 4], $slice->shape());
        $this->assertSame([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ], $slice->toArray());

        $slice = $arr->slice([':', '0:2']);
        $this->assertSame([3, 2], $slice->shape());
        $this->assertSame([
            [1, 2],
            [5, 6],
            [9, 10],
        ], $slice->toArray());

        $slice = $arr->slice(['1:3', '1:3']);
        $this->assertSame([2, 2], $slice->shape());
        $this->assertSame([
            [6, 7],
            [10, 11],
        ], $slice->toArray());
    }

    public function testSliceStep1D(): void
    {
        $arr = NDArray::arange(0, 10);

        $slice = $arr->slice(['::2']);
        $this->assertSame([5], $slice->shape());
        $this->assertSame([0, 2, 4, 6, 8], $slice->toArray());

        $slice = $arr->slice(['1::2']);
        $this->assertSame([5], $slice->shape());
        $this->assertSame([1, 3, 5, 7, 9], $slice->toArray());

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

        $slice = $arr->slice(['::2', '::2']);
        $this->assertSame([2, 2], $slice->shape());
        $this->assertSame([
            [1, 3],
            [9, 11],
        ], $slice->toArray());
    }

    public function testMixedIndexing(): void
    {
        $arr = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]);

        $slice = $arr->slice([1, '0:2']);
        $this->assertSame([2], $slice->shape());
        $this->assertSame([4, 5], $slice->toArray());

        $slice = $arr->slice(['0:2', 1]);
        $this->assertSame([2], $slice->shape());
        $this->assertSame([2, 5], $slice->toArray());
    }

    public function testArrayAccessSlice(): void
    {
        $arr = NDArray::arange(10);

        $slice = $arr['0:3'];
        $this->assertInstanceOf(NDArray::class, $slice);
        $this->assertSame([0, 1, 2], $slice->toArray());

        $slice = $arr['::3'];
        $this->assertSame([0, 3, 6, 9], $slice->toArray());
    }

    public function testArrayAccessMultiDimSlice(): void
    {
        $arr = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
        ]);

        $col = $arr[':, 1'];
        $this->assertSame([2], $col->shape());
        $this->assertSame([2, 5], $col->toArray());

        $sub = $arr['0, 1:'];
        $this->assertSame([2], $sub->shape());
        $this->assertSame([2, 3], $sub->toArray());
    }

    public function testSliceNegativeIndices(): void
    {
        $arr = NDArray::arange(10);

        $slice = $arr->slice([':-1']);
        $this->assertSame([9], $slice->shape());
        $this->assertSame([0, 1, 2, 3, 4, 5, 6, 7, 8], $slice->toArray());

        $slice = $arr->slice(['-3:']);
        $this->assertSame([3], $slice->shape());
        $this->assertSame([7, 8, 9], $slice->toArray());

        $slice = $arr->slice(['-3:-1']);
        $this->assertSame([2], $slice->shape());
        $this->assertSame([7, 8], $slice->toArray());
    }

    public function testAssignScalarToSlice(): void
    {
        $arr = NDArray::zeros([5]);
        $arr->slice(['1:4'])->assign(9);

        $this->assertSame([0.0, 9.0, 9.0, 9.0, 0.0], $arr->toArray());
    }

    public function testAssignNDArrayToSlice(): void
    {
        $arr = NDArray::zeros([5]);
        $arr->slice(['1:4'])->assign(NDArray::array([10.0, 20.0, 30.0]));

        $this->assertSame([0.0, 10.0, 20.0, 30.0, 0.0], $arr->toArray());
    }

    public function testAssignNDArrayTo2DSlice(): void
    {
        $arr = NDArray::zeros([3, 3]);
        $arr->slice(['1:2', '1:2'])->assign(1);

        $expected = [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        $this->assertSame($expected, $arr->toArray());

        // Set top-left 2x2 to [[1,2],[3,4]]
        $arr->slice([':2', ':2'])->assign(NDArray::array([[1.0, 2.0], [3.0, 4.0]]));
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

        $arr['1:3'] = 5;
        $this->assertSame([0, 5, 5, 0, 0], $arr->toArray());

        $arr['3:'] = NDArray::array([8, 9], DType::Int32);
        $this->assertSame([0, 5, 5, 8, 9], $arr->toArray());
    }

    public function testArrayAccessBroadcastAssignmentToColumnSlice(): void
    {
        $arr = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Int32);

        $arr[':, 0'] = 5;

        $this->assertSame([
            [5, 2, 3],
            [5, 5, 6],
            [5, 8, 9],
        ], $arr->toArray());
    }

    public function testArrayAccessPartialAssignment(): void
    {
        $arr = NDArray::zeros([2, 2], DType::Int32);

        $arr[0] = NDArray::array([1, 2], DType::Int32);

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

    public function testEllipsisBasic(): void
    {
        $arr = NDArray::array([
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ]); // Shape [2, 2, 3]

        // ... should expand to cover all dimensions (same as arr[:])
        $slice = $arr->slice(['...']);
        $this->assertSame([2, 2, 3], $slice->shape());
        $this->assertSame($arr->toArray(), $slice->toArray());
    }

    public function testEllipsisAtEnd(): void
    {
        $arr = NDArray::array([
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ]); // Shape [2, 2, 3]

        // 0, ... -> first element of first dim, all remaining dims
        $slice = $arr->slice([0, '...']);
        $this->assertSame([2, 3], $slice->shape());
        $this->assertSame([[1, 2, 3], [4, 5, 6]], $slice->toArray());
    }

    public function testEllipsisAtBeginning(): void
    {
        $arr = NDArray::array([
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ]); // Shape [2, 2, 3]

        // ..., 1:2 -> all leading dims, then slice last dim
        $slice = $arr->slice(['...', '1:2']);
        $this->assertSame([2, 2, 1], $slice->shape());
        $this->assertSame([[[2], [5]], [[8], [11]]], $slice->toArray());
    }

    public function testEllipsisInMiddle(): void
    {
        $arr = NDArray::zeros([2, 3, 4, 5]); // 4D array

        // 0, ..., 1:3 -> first element of first dim, all middle dims, slice last dim
        $slice = $arr->slice([0, '...', '1:3']);
        $this->assertSame([3, 4, 2], $slice->shape());
    }

    public function testEllipsisWithMultipleSelectors(): void
    {
        $arr = NDArray::array([
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ]);

        // 0, ..., 1 -> first element of first dim, all middle dims, second element of last dim
        $slice = $arr->slice([0, '...', 1]);
        $this->assertSame([2], $slice->shape());
        $this->assertSame([2, 5], $slice->toArray());
    }

    public function testEllipsisInArrayAccess(): void
    {
        $arr = NDArray::array([
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ]);

        // Test via ArrayAccess string syntax
        $slice = $arr['0, ..., 1:'];
        $this->assertSame([2, 2], $slice->shape());
        $this->assertSame([[2, 3], [5, 6]], $slice->toArray());
    }

    public function testEllipsisWithStep(): void
    {
        $arr = NDArray::array([
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            [[9, 10, 11, 12], [13, 14, 15, 16]],
        ]); // Shape [2, 2, 4]

        // ::2, ..., 0:4:2
        $slice = $arr->slice(['::2', '...', '0:4:2']);
        $this->assertSame([1, 2, 2], $slice->shape());
        $this->assertSame([[[1, 3], [5, 7]]], $slice->toArray());
    }

    public function testEllipsisMultipleEllipsisThrows(): void
    {
        $arr = NDArray::zeros([2, 3, 4]);

        $this->expectException(IndexException::class);
        $this->expectExceptionMessage('Only one ellipsis');

        $arr->slice(['...', 0, '...']);
    }

    public function testEllipsisTooManyIndicesThrows(): void
    {
        $arr = NDArray::zeros([2, 3, 4]);

        $this->expectException(IndexException::class);
        $this->expectExceptionMessage('Too many indices');

        // 5 explicit indices + ellipsis = way too many
        $arr->slice([0, 0, 0, 0, 0, '...']);
    }

    public function testEllipsis4DArray(): void
    {
        $arr = NDArray::arange(120)->reshape([2, 3, 4, 5]); // 2x3x4x5

        // ..., 2:4 should select last two elements of last dimension for all
        $slice = $arr->slice(['...', '2:4']);
        $this->assertSame([2, 3, 4, 2], $slice->shape());
    }

    public function testEllipsisAssignment(): void
    {
        $arr = NDArray::zeros([2, 2, 3]);

        // Assign scalar through ellipsis slice
        $arr->slice([0, '...'])->assign(5);

        $expected = [
            [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ];
        $this->assertSame($expected, $arr->toArray());
    }

    public function testAssignShapeMismatchThrows(): void
    {
        $arr = NDArray::zeros([5]);

        $this->expectException(ShapeException::class);
        $this->expectExceptionMessage('Cannot assign array of size 2 to view of size 3');

        $arr->slice(['0:3'])->assign(NDArray::array([1, 2]));
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

        $slice = $arr['invalid:selector'];
    }

    public function testAssignPHPArrayThrows(): void
    {
        $arr = NDArray::zeros([5]);

        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('Assignment value must be scalar or NDArray');

        $arr->slice(['0:3'])->assign([1, 2, 3]);
    }

    public function testSlice3DBasic(): void
    {
        $arr = NDArray::arange(8)->reshape([2, 2, 2]);

        $slice = $arr->slice([0]);
        $this->assertSame([2, 2], $slice->shape());
        $this->assertSame([[0, 1], [2, 3]], $slice->toArray());

        $slice = $arr->slice([':', ':', 0]);
        $this->assertSame([2, 2], $slice->shape());
        $this->assertSame([[0, 2], [4, 6]], $slice->toArray());
    }

    public function testSlice3DStrided(): void
    {
        $arr = NDArray::arange(27)->reshape([3, 3, 3]);

        $slice = $arr->slice([':', '::2', ':']);
        $this->assertSame([3, 2, 3], $slice->shape());
        $this->assertEquals(6, $slice->get(0, 1, 0));
    }

    public function testAssignScalarTo3DSlice(): void
    {
        $arr = NDArray::zeros([3, 3, 3], DType::Int32);

        $arr->slice([1])->assign(1);

        $this->assertEquals(0, $arr->get(0, 0, 0));
        $this->assertEquals(1, $arr->get(1, 0, 0));
        $this->assertEquals(1, $arr->get(1, 2, 2));
        $this->assertEquals(0, $arr->get(2, 0, 0));
    }

    public function testAssignScalarToStrided3DSlice(): void
    {
        $arr = NDArray::zeros([4, 4, 4], DType::Int32);

        $arr->slice(['::2'])->assign(5);

        $this->assertEquals(5, $arr->get(0, 0, 0));
        $this->assertEquals(0, $arr->get(1, 0, 0));
        $this->assertEquals(5, $arr->get(2, 0, 0));
        $this->assertEquals(0, $arr->get(3, 0, 0));
    }

    public function testStringIndexConvertsToInteger(): void
    {
        $arr = NDArray::array([[1, 2, 3], [4, 5, 6]]);

        $byString = $arr->slice([':', '0']);
        $byInt = $arr->slice([':', 0]);

        $this->assertSame($byInt->shape(), $byString->shape());
        $this->assertEquals($byInt->toArray(), $byString->toArray());
        $this->assertEquals([1, 4], $byString->toArray());
    }

    public function testStringIndexExtractsColumnFrom2D(): void
    {
        $arr = NDArray::array([[1, 2, 3], [4, 5, 6]]);

        $col = $arr->slice([':', '1']);

        $this->assertSame([2], $col->shape());
        $this->assertEquals([2, 5], $col->toArray());
    }

    public function testStringIndexExtractsRowFrom3D(): void
    {
        $arr = NDArray::array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ]);

        $row = $arr->slice([':', '0', ':']);

        $this->assertSame([2, 2], $row->shape());
        $this->assertEquals([[1, 2], [5, 6]], $row->toArray());
    }

    public function testNegativeStringIndex(): void
    {
        $arr = NDArray::array([[1, 2, 3], [4, 5, 6]]);

        $col = $arr->slice([':', '-1']);

        $this->assertSame([2], $col->shape());
        $this->assertEquals([3, 6], $col->toArray());
    }

    public function testMixedStringAndIntegerSelectors(): void
    {
        $arr = NDArray::array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ]);

        $slice = $arr->slice(['0', '0', ':']);

        $this->assertSame([2], $slice->shape());
        $this->assertEquals([1, 2], $slice->toArray());
    }
}
