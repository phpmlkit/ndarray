<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Unit;

use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\Exceptions\ShapeException;
use PhpMlKit\NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Tests for remainder operations (rem/%).
 *
 * @internal
 *
 * @coversNothing
 */
final class RemTest extends TestCase
{
    // ========================================================================
    // Array-Array Operations
    // ========================================================================

    public function testRemIntArrays(): void
    {
        $a = NDArray::array([10, 20, 30], DType::Int32);
        $b = NDArray::array([3, 7, 8], DType::Int32);
        $result = $a->rem($b);

        $this->assertSame([3], $result->shape());
        $this->assertSame(DType::Int32, $result->dtype());
        $this->assertSame([1, 6, 6], $result->toArray());
    }

    public function testRemUintArrays(): void
    {
        $a = NDArray::array([10, 20, 30], DType::UInt32);
        $b = NDArray::array([3, 7, 8], DType::UInt32);
        $result = $a->rem($b);

        $this->assertSame([1, 6, 6], $result->toArray());
    }

    public function testRemFloatArrays(): void
    {
        $a = NDArray::array([10.5, 20.5, 30.5], DType::Float64);
        $b = NDArray::array([3.0, 7.0, 8.0], DType::Float64);
        $result = $a->rem($b);

        $this->assertEqualsWithDelta([1.5, 6.5, 6.5], $result->toArray(), 0.0001);
    }

    public function testRem2DArrays(): void
    {
        $a = NDArray::array([[10, 20], [30, 40]], DType::Int32);
        $b = NDArray::array([[3, 7], [8, 6]], DType::Int32);
        $result = $a->rem($b);

        $this->assertSame([[1, 6], [6, 4]], $result->toArray());
    }

    public function testRemWithNegativeNumbers(): void
    {
        $a = NDArray::array([-10, -20, 30], DType::Int32);
        $b = NDArray::array([3, 7, 8], DType::Int32);
        $result = $a->rem($b);

        $this->assertSame([-1, -6, 6], $result->toArray());
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
        $b = NDArray::array([3, 0, 8], DType::Int32);
        $result = $a->rem($b);
    }

    public function testRemAllIntTypes(): void
    {
        $types = [
            DType::Int8,
            DType::Int16,
            DType::Int32,
            DType::Int64,
            DType::UInt8,
            DType::UInt16,
            DType::UInt32,
            DType::UInt64,
        ];

        foreach ($types as $dtype) {
            $a = NDArray::array([10, 20, 30], $dtype);
            $b = NDArray::array([3, 7, 8], $dtype);
            $result = $a->rem($b);

            $this->assertSame($dtype, $result->dtype(), "Failed for {$dtype->name}");
            $this->assertSame([1, 6, 6], $result->toArray());
        }
    }

    public function testRemMixedTypesPromotion(): void
    {
        $a = NDArray::array([10, 20], DType::Int32);
        $b = NDArray::array([3, 7], DType::Int64);
        $result = $a->rem($b);

        $this->assertSame(DType::Int64, $result->dtype());
        $this->assertSame([1, 6], $result->toArray());
    }

    // ========================================================================
    // Array-Scalar Operations
    // ========================================================================

    public function testRemIntScalar(): void
    {
        $a = NDArray::array([10, 20, 30], DType::Int32);
        $result = $a->rem(7);

        $this->assertSame([3, 6, 2], $result->toArray());
    }

    public function testRemFloatScalar(): void
    {
        $a = NDArray::array([10.5, 20.5, 30.5], DType::Float64);
        $result = $a->rem(3.0);

        $this->assertEqualsWithDelta([1.5, 2.5, 0.5], $result->toArray(), 0.0001);
    }

    public function testRemScalarByZeroShouldError(): void
    {
        $this->expectException(\Exception::class);

        $a = NDArray::array([10, 20, 30], DType::Int32);
        $result = $a->rem(0);
    }

    // ========================================================================
    // Bool Type (Not Supported)
    // ========================================================================

    public function testRemBoolShouldError(): void
    {
        $this->expectException(\Exception::class);
        $this->expectExceptionMessage('not supported for Bool');

        $a = NDArray::array([true, false, true], DType::Bool);
        $b = NDArray::array([true, true, false], DType::Bool);
        $result = $a->rem($b);
    }

    public function testRemBoolScalarShouldError(): void
    {
        $this->expectException(\Exception::class);
        $this->expectExceptionMessage('not supported for Bool');

        $a = NDArray::array([true, false, true], DType::Bool);
        $result = $a->rem(1);
    }

    // ========================================================================
    // View Operations
    // ========================================================================

    public function testRemOnViews(): void
    {
        $a = NDArray::array([10, 20, 30, 40, 50, 60], DType::Int32);
        $b = NDArray::array([3, 4, 5, 6, 7, 8], DType::Int32);

        $view_a = $a->slice(['1:4']); // [20, 30, 40]
        $view_b = $b->slice(['1:4']); // [4, 5, 6]

        $result = $view_a->rem($view_b);

        $this->assertSame([3], $result->shape());
        $this->assertSame([0, 0, 4], $result->toArray());
    }

    public function testRemScalarOnView(): void
    {
        $a = NDArray::array([10, 20, 30, 40, 50, 60], DType::Int32);
        $view = $a->slice(['2:5']); // [30, 40, 50]

        $result = $view->rem(7);

        $this->assertSame([2, 5, 1], $result->toArray());
    }

    public function testRemViewWithScalarBroadcast(): void
    {
        $a = NDArray::array([
            [10, 20, 30],
            [40, 50, 60],
        ], DType::Int32);

        // Get row as view
        $row = $a->get(1); // [40, 50, 60]
        $result = $row->rem(7);

        $this->assertSame([5, 1, 4], $result->toArray());
    }

    // ========================================================================
    // Edge Cases
    // ========================================================================

    public function testRemWithRemainderOfZero(): void
    {
        $a = NDArray::array([12, 24, 36], DType::Int32);
        $b = NDArray::array([3, 4, 6], DType::Int32);
        $result = $a->rem($b);

        $this->assertSame([0, 0, 0], $result->toArray());
    }

    public function testRemSingleElementArrays(): void
    {
        $a = NDArray::array([17], DType::Int32);
        $b = NDArray::array([5], DType::Int32);
        $result = $a->rem($b);

        $this->assertSame([2], $result->toArray());
    }

    public function testRemEmptyArray(): void
    {
        $a = NDArray::zeros([0], DType::Int32);
        $b = NDArray::zeros([0], DType::Int32);
        $result = $a->rem($b);

        $this->assertSame([], $result->toArray());
    }

    public function testRemIncompatibleShapesShouldError(): void
    {
        $this->expectException(ShapeException::class);

        $a = NDArray::array([10, 20, 30], DType::Int32);
        $b = NDArray::array([3, 7], DType::Int32);
        $result = $a->rem($b);
    }

    // ========================================================================
    // Zero-Dimensional Arrays
    // ========================================================================

    public function testRemZeroDimArray(): void
    {
        $a = NDArray::array([42], DType::Int32)->reshape([]);
        $b = NDArray::array([5], DType::Int32)->reshape([]);

        $result = $a->rem($b);

        $this->assertSame([], $result->shape());
        $this->assertSame(2, $result->toArray());
    }

    public function testRemZeroDimWithScalar(): void
    {
        $a = NDArray::array([42], DType::Int32)->reshape([]);
        $result = $a->rem(5);

        $this->assertSame(2, $result->toArray());
    }
}
