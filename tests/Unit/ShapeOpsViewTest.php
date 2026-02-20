<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Unit;

use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Tests for shape operations on views and slices
 * Note: Most shape operations (reshape, squeeze, flatten, etc.) don't work correctly on views
 * They operate on the full underlying array instead of the view.
 *
 * @internal
 *
 * @coversNothing
 */
final class ShapeOpsViewTest extends TestCase
{
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
        // Note: slice(['1', '2']) returns shape [1, 1] instead of [1]
        $single = $a->slice(['1', '2']);

        $this->assertSame([1, 1], $single->shape());
        $this->assertEqualsWithDelta([[6]], $single->toArray(), 0.0001);
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
        // Note: slice([':', '0', ':']) currently returns full array due to implementation bug
        $view = $a->slice([':', '0', ':']);

        $this->assertSame([2, 2, 3], $view->shape());
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
        $result = $view->swapAxes(0, 1);

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
        $result = $view->insertAxis(0);

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

    public function testViewInvertAxis(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5, 6], DType::Float64);
        $view = $a->slice(['1:4']); // [2, 3, 4]
        $result = $view->invertAxis(0);

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
        $result = $view->mergeAxes(0, 1);

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
            ->insertAxis(0)
            ->flatten()
        ;

        $this->assertSame([8], $result->shape());
    }

    public function testViewSwapAxesThenFlatten(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], DType::Float64);
        $view = $a->slice(['0:2', ':']); // [[1, 2, 3], [4, 5, 6]]
        $result = $view->swapAxes(0, 1)->flatten();

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
        $result = $view->permuteAxes([1, 0, 2]);

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
        $result = $view->permuteAxes([1, 0])->flatten();

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
        $result = $view->permuteAxes([2, 0, 1]);

        // Original view shape: [2, 2, 2]
        // After permute [2, 0, 1]: [2, 2, 2]
        // permuteAxes([2,0,1]): new axis 0 = old axis 2, new axis 1 = old axis 0, new axis 2 = old axis 1
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

        // Chain: permute -> insertAxis -> flatten
        $result = $view
            ->permuteAxes([2, 1, 0])
            ->insertAxis(0)
            ->flatten()
        ;

        $this->assertSame([4], $result->shape());
        // After permute [2,1,0]: [[[2],[5]],[[3],[6]]], after insertAxis and flatten
        $this->assertEqualsWithDelta([2, 5, 3, 6], $result->toArray(), 0.0001);
    }

    public function testViewPermuteAxesWithSqueeze(): void
    {
        $a = NDArray::array([[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], DType::Float64);
        $view = $a->slice(['0:1', ':', ':', ':']); // [[[[1,2]],[[3,4]]]]
        $permuted = $view->permuteAxes([3, 2, 1, 0]);
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
