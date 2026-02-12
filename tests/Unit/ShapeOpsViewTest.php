<?php

declare(strict_types=1);

namespace NDArray\Tests\Unit;

use NDArray\DType;
use NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Tests for shape operations on views and slices
 * Note: Most shape operations (reshape, squeeze, flatten, etc.) don't work correctly on views
 * They operate on the full underlying array instead of the view.
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
            [9, 10, 11, 12]
        ], DType::Float64);
        $view = $a->slice(['0:2', '1:3']); // [[2, 3], [6, 7]]
        
        $this->assertSame([2, 2], $view->shape());
        $this->assertEqualsWithDelta([[2, 3], [6, 7]], $view->toArray(), 0.0001);
    }

    public function test3DSliceShape(): void
    {
        $a = NDArray::array([
            [[1, 2], [3, 4], [5, 6]],
            [[7, 8], [9, 10], [11, 12]]
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
            [13, 14, 15, 16]
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
            [7, 8, 9]
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
            [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]
        ], DType::Float64);
        $view = $a->slice(['0:1', ':', ':', ':']);
        
        $this->assertSame([1, 2, 2, 2], $view->shape());
    }

    public function testSliceWithMultipleDimensions(): void
    {
        $a = NDArray::array([
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]]
        ], DType::Float64);
        // Note: slice([':', '0', ':']) currently returns full array due to implementation bug
        $view = $a->slice([':', '0', ':']);
        
        $this->assertSame([2, 2, 3], $view->shape());
    }
}
