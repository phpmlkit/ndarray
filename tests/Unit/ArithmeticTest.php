<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Unit;

use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\Exceptions\DTypeException;
use PhpMlKit\NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Tests for arithmetic operations (add, subtract, multiply, divide, negative).
 *
 * @internal
 *
 * @coversNothing
 */
final class ArithmeticTest extends TestCase
{
    public function testAddArraysSameShape(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);
        $b = NDArray::array([[5, 6], [7, 8]], DType::Float64);

        $result = $a->add($b);

        $this->assertSame([2, 2], $result->shape());
        $this->assertEqualsWithDelta([[6, 8], [10, 12]], $result->toArray(), 0.0001);
    }

    public function testAddScalar(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);

        $result = $a->add(10);

        $this->assertSame([2, 2], $result->shape());
        $this->assertEqualsWithDelta([[11, 12], [13, 14]], $result->toArray(), 0.0001);
    }

    public function testSubtractArrays(): void
    {
        $a = NDArray::array([[10, 20], [30, 40]], DType::Float64);
        $b = NDArray::array([[1, 2], [3, 4]], DType::Float64);

        $result = $a->subtract($b);

        $this->assertEqualsWithDelta([[9, 18], [27, 36]], $result->toArray(), 0.0001);
    }

    public function testMultiplyArrays(): void
    {
        $a = NDArray::array([[2, 3], [4, 5]], DType::Float64);
        $b = NDArray::array([[2, 2], [2, 2]], DType::Float64);

        $result = $a->multiply($b);

        $this->assertEqualsWithDelta([[4, 6], [8, 10]], $result->toArray(), 0.0001);
    }

    public function testDivideArrays(): void
    {
        $a = NDArray::array([[10, 20], [30, 40]], DType::Float64);
        $b = NDArray::array([[2, 4], [5, 8]], DType::Float64);

        $result = $a->divide($b);

        $this->assertEqualsWithDelta([[5, 5], [6, 5]], $result->toArray(), 0.0001);
    }

    public function testMultiplyScalar(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);

        $result = $a->multiply(2.5);

        $this->assertEqualsWithDelta([[2.5, 5], [7.5, 10]], $result->toArray(), 0.0001);
    }

    public function testDivideScalar(): void
    {
        $a = NDArray::array([[10, 20], [30, 40]], DType::Float64);

        $result = $a->divide(2);

        $this->assertEqualsWithDelta([[5, 10], [15, 20]], $result->toArray(), 0.0001);
    }

    public function testChainedOperations(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $b = NDArray::array([4, 5, 6], DType::Float64);

        $result = $a->add($b)->multiply(2);

        $this->assertEqualsWithDelta([10, 14, 18], $result->toArray(), 0.0001);
    }

    public function testIntegerArithmetic(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Int32);
        $b = NDArray::array([[5, 6], [7, 8]], DType::Int32);

        $result = $a->add($b);

        $this->assertSame([2, 2], $result->shape());
        // Result should be Int32 or promoted type
        $this->assertEquals([[6, 8], [10, 12]], $result->toArray());
    }

    public function testMixedTypesPromotion(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Int32);
        $b = NDArray::array([1.5, 2.5, 3.5], DType::Float64);

        $result = $a->add($b);

        // Result should be Float64 (higher precision)
        $this->assertSame(DType::Float64, $result->dtype());
        $this->assertEqualsWithDelta([2.5, 4.5, 6.5], $result->toArray(), 0.0001);
    }

    public function testFloat32Operations(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float32);
        $b = NDArray::array([[5, 6], [7, 8]], DType::Float32);

        $result = $a->add($b);

        $this->assertSame([2, 2], $result->shape());
        $this->assertEqualsWithDelta([[6, 8], [10, 12]], $result->toArray(), 0.0001);
    }

    public function testNegativeFloat(): void
    {
        $a = NDArray::array([1.0, -2.0, 3.5], DType::Float64);
        $result = $a->negative();
        $this->assertEqualsWithDelta([-1.0, 2.0, -3.5], $result->toArray(), 0.0001);
    }

    public function testNegativeInt(): void
    {
        $a = NDArray::array([1, -2, 3], DType::Int32);
        $result = $a->negative();
        $this->assertEquals([-1, 2, -3], $result->toArray());
    }

    public function testNegativeUnsignedShouldError(): void
    {
        $this->expectException(DTypeException::class);
        $a = NDArray::array([1, 2, 3], DType::UInt32);
        $a->negative();
    }

    public function testAddOn1DSlice(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5, 6], DType::Float64);
        $slice = $a->slice(['2:5']); // [3, 4, 5]
        $b = NDArray::array([10, 20, 30], DType::Float64);

        $result = $slice->add($b);

        $this->assertSame([3], $result->shape());
        $this->assertEqualsWithDelta([13, 24, 35], $result->toArray(), 0.0001);
    }

    public function testMultiplyOn1DStridedSlice(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5, 6, 7, 8], DType::Float64);
        $slice = $a->slice(['::2']); // [1, 3, 5, 7] - every other element
        $scalar = 10;

        $result = $slice->multiply($scalar);

        $this->assertSame([4], $result->shape());
        $this->assertEqualsWithDelta([10, 30, 50, 70], $result->toArray(), 0.0001);
    }

    public function testSubtractTwoSlices(): void
    {
        $a = NDArray::array([10, 20, 30, 40, 50, 60], DType::Float64);
        $slice1 = $a->slice(['1:4']); // [20, 30, 40]
        $slice2 = $a->slice(['2:5']); // [30, 40, 50]

        $result = $slice1->subtract($slice2);

        $this->assertSame([3], $result->shape());
        $this->assertEqualsWithDelta([-10, -10, -10], $result->toArray(), 0.0001);
    }

    public function testAddOn2DRowSlice(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], DType::Float64);
        $row = $a[1]; // [4, 5, 6]
        $b = NDArray::array([10, 20, 30], DType::Float64);

        $result = $row->add($b);

        $this->assertSame([3], $result->shape());
        $this->assertEqualsWithDelta([14, 25, 36], $result->toArray(), 0.0001);
    }

    public function testMultiplyOn2DColumnSlice(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], DType::Float64);
        $col = $a->slice([':', '1:2']); // [[2], [5], [8]] - shape [3, 1]
        $scalar = 2;

        $result = $col->multiply($scalar);

        $this->assertSame([3, 1], $result->shape());
        $this->assertEqualsWithDelta([[4], [10], [16]], $result->toArray(), 0.0001);
    }

    public function testDivideOn2DSubarray(): void
    {
        $a = NDArray::array([
            [10, 20, 30, 40],
            [50, 60, 70, 80],
            [90, 100, 110, 120],
        ], DType::Float64);
        $subarray = $a->slice(['0:2', '1:3']); // [[20, 30], [60, 70]]
        $b = NDArray::array([[2, 3], [4, 5]], DType::Float64);

        $result = $subarray->divide($b);

        $this->assertSame([2, 2], $result->shape());
        $this->assertEqualsWithDelta([[10, 10], [15, 14]], $result->toArray(), 0.0001);
    }

    public function testAddOn2DStridedView(): void
    {
        $a = NDArray::array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ], DType::Float64);
        $strided = $a->slice(['::2', '::2']); // [[1, 3], [9, 11]] - every other row and column
        $b = NDArray::array([[100, 200], [300, 400]], DType::Float64);

        $result = $strided->add($b);

        $this->assertSame([2, 2], $result->shape());
        $this->assertEqualsWithDelta([[101, 203], [309, 411]], $result->toArray(), 0.0001);
    }

    public function testChainedOperationsOnView(): void
    {
        $a = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Float64);
        $view = $a->slice(['0:2', '1:3']); // [[2, 3], [5, 6]]

        $result = $view->add(10)->multiply(2);

        $this->assertSame([2, 2], $result->shape());
        $this->assertEqualsWithDelta([[24, 26], [30, 32]], $result->toArray(), 0.0001);
    }

    public function testAddOn3DSlice(): void
    {
        $a = NDArray::array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]],
        ], DType::Float64);
        $slice = $a->slice(['1:2']); // [[[5, 6], [7, 8]]] - shape [1, 2, 2]
        $b = NDArray::array([[[100, 200], [300, 400]]], DType::Float64);

        $result = $slice->add($b);

        $this->assertSame([1, 2, 2], $result->shape());
        $this->assertEqualsWithDelta([[[105, 206], [307, 408]]], $result->toArray(), 0.0001);
    }

    public function testMultiplyOn3DDepthSlice(): void
    {
        $a = NDArray::array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ], DType::Float64);
        // String index '0' is converted to integer 0, extracting first element of second dimension
        $depth_slice = $a->slice([':', '0', ':']);
        $scalar = 10;

        $result = $depth_slice->multiply($scalar);

        // After slice, shape is [2, 2], multiplied by scalar stays [2, 2]
        $this->assertSame([2, 2], $result->shape());
        $this->assertEqualsWithDelta([[10, 20], [50, 60]], $result->toArray(), 0.0001);
    }

    public function testSingleElementView(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $single = $a->slice(['1:2', '2:3']); // Single element: [[6]] with shape [1, 1]
        $b = NDArray::array([10], DType::Float64);

        $result = $single->add($b);

        $this->assertSame([1, 1], $result->shape());
        $this->assertEqualsWithDelta([[16]], $result->toArray(), 0.0001);
    }

    public function testSingleElementViewWithScalar(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $single = $a->slice(['1:2', '2:3']); // [[6]]

        $result = $single->add(10);

        $this->assertSame([1, 1], $result->shape());
        $this->assertEqualsWithDelta([[16]], $result->toArray(), 0.0001);
    }

    public function testEmptyView(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5], DType::Float64);
        $empty = $a->slice(['2:2']); // Empty slice

        $this->assertSame([0], $empty->shape());
    }

    public function testNegativeIndexView(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], DType::Float64);
        $view = $a->slice(['-2:', '-2:']); // Last 2 rows, last 2 columns: [[5, 6], [8, 9]]
        $b = NDArray::array([[100, 200], [300, 400]], DType::Float64);

        $result = $view->add($b);

        $this->assertSame([2, 2], $result->shape());
        $this->assertEqualsWithDelta([[105, 206], [308, 409]], $result->toArray(), 0.0001);
    }

    public function testRemIntArrays(): void
    {
        $a = NDArray::array([10, 20, 30], DType::Int32);
        $b = NDArray::array([3, 7, 8], DType::Int32);
        $result = $a->rem($b);

        $this->assertSame([3], $result->shape());
        $this->assertSame(DType::Int32, $result->dtype());
        $this->assertSame([1, 6, 6], $result->toArray());
    }

    public function testRemFloatArrays(): void
    {
        $a = NDArray::array([10.5, 20.5, 30.5], DType::Float64);
        $b = NDArray::array([3.0, 7.0, 8.0], DType::Float64);
        $result = $a->rem($b);

        $this->assertEqualsWithDelta([1.5, 6.5, 6.5], $result->toArray(), 0.0001);
    }

    public function testRemWithNegativeNumbers(): void
    {
        $a = NDArray::array([-10, -20, 30], DType::Int32);
        $b = NDArray::array([3, 7, 8], DType::Int32);
        $result = $a->rem($b);

        $this->assertSame([-1, -6, 6], $result->toArray());
    }

    public function testRemScalar(): void
    {
        $a = NDArray::array([10, 20, 30], DType::Int32);
        $result = $a->rem(7);

        $this->assertSame([3, 6, 2], $result->toArray());
    }

    public function testRemBroadcasting(): void
    {
        $a = NDArray::array([[10, 20, 30], [40, 50, 60]], DType::Int32);
        $b = NDArray::array([7], DType::Int32);
        $result = $a->rem($b);

        $this->assertSame([2, 3], $result->shape());
        $this->assertSame([[3, 6, 2], [5, 1, 4]], $result->toArray());
    }

    public function testRemByZeroShouldError(): void
    {
        $this->expectException(\Exception::class);

        $a = NDArray::array([10, 20, 30], DType::Int32);
        $result = $a->rem(0);
    }

    public function testRemOnViews(): void
    {
        $a = NDArray::array([10, 20, 30, 40, 50, 60], DType::Int32);
        $b = NDArray::array([3, 4, 5, 6, 7, 8], DType::Int32);

        $view_a = $a->slice(['1:4']);
        $view_b = $b->slice(['1:4']);

        $result = $view_a->rem($view_b);

        $this->assertSame([3], $result->shape());
        $this->assertSame([0, 0, 4], $result->toArray());
    }
}
