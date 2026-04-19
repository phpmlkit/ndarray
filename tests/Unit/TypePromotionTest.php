<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Unit;

use PhpMlKit\NDArray\Complex;
use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * @internal
 *
 * @coversNothing
 */
final class TypePromotionTest extends TestCase
{
    // ==================== Arithmetic Operations ====================

    public function testIntArrayPlusFloatScalarPromotesToFloat64(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Int64);
        $result = $a->add(1.5);

        $this->assertSame(DType::Float64, $result->dtype());
        $this->assertEqualsWithDelta([2.5, 3.5, 4.5], $result->toArray(), 0.0001);
    }

    public function testIntArrayMinusFloatScalarPromotesToFloat64(): void
    {
        $a = NDArray::array([10, 20, 30], DType::Int64);
        $result = $a->subtract(2.5);

        $this->assertSame(DType::Float64, $result->dtype());
        $this->assertEqualsWithDelta([7.5, 17.5, 27.5], $result->toArray(), 0.0001);
    }

    public function testIntArrayTimesFloatScalarPromotesToFloat64(): void
    {
        $a = NDArray::array([2, 4, 6], DType::Int64);
        $result = $a->multiply(1.5);

        $this->assertSame(DType::Float64, $result->dtype());
        $this->assertEqualsWithDelta([3.0, 6.0, 9.0], $result->toArray(), 0.0001);
    }

    public function testIntArrayDividedByFloatScalarPromotesToFloat64(): void
    {
        $a = NDArray::array([10, 20, 30], DType::Int64);
        $result = $a->divide(2.5);

        $this->assertSame(DType::Float64, $result->dtype());
        $this->assertEqualsWithDelta([4.0, 8.0, 12.0], $result->toArray(), 0.0001);
    }

    public function testFloatArrayPlusIntScalarStaysFloat64(): void
    {
        $a = NDArray::array([1.5, 2.5, 3.5], DType::Float64);
        $result = $a->add(10);

        $this->assertSame(DType::Float64, $result->dtype());
        $this->assertEqualsWithDelta([11.5, 12.5, 13.5], $result->toArray(), 0.0001);
    }

    public function testIntArrayPlusComplexScalarPromotesToComplex128(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Int64);
        $complex = new Complex(3.0, 4.0);
        $result = $a->add($complex);

        $this->assertSame(DType::Complex128, $result->dtype());
        $values = $result->toArray();
        $this->assertCount(3, $values);
        $this->assertEqualsWithDelta(4.0, $values[0]->real, 0.0001);
        $this->assertEqualsWithDelta(4.0, $values[0]->imag, 0.0001);
    }

    public function testFloatArrayPlusComplexScalarPromotesToComplex128(): void
    {
        $a = NDArray::array([1.5, 2.5], DType::Float64);
        $complex = new Complex(1.0, 2.0);
        $result = $a->add($complex);

        $this->assertSame(DType::Complex128, $result->dtype());
        $values = $result->toArray();
        $this->assertEqualsWithDelta(2.5, $values[0]->real, 0.0001);
        $this->assertEqualsWithDelta(2.0, $values[0]->imag, 0.0001);
    }

    public function testIntArrayRemFloatScalarPromotesToFloat64(): void
    {
        $a = NDArray::array([10, 20, 30], DType::Int64);
        $result = $a->rem(3.5);

        $this->assertSame(DType::Float64, $result->dtype());
        $this->assertEqualsWithDelta([3.0, 2.5, 2.0], $result->toArray(), 0.0001);
    }

    // ==================== Comparison Operations ====================

    public function testIntArrayEqFloatScalar(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Int64);
        $result = $a->eq(2.0);

        $this->assertSame(DType::Bool, $result->dtype());
        $this->assertSame([false, true, false], $result->toArray());
    }

    public function testIntArrayGtFloatScalar(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Int64);
        $result = $a->gt(1.5);

        $this->assertSame(DType::Bool, $result->dtype());
        $this->assertSame([false, true, true], $result->toArray());
    }

    public function testFloatArrayLtIntScalar(): void
    {
        $a = NDArray::array([1.5, 2.5, 3.5], DType::Float64);
        $result = $a->lt(3);

        $this->assertSame(DType::Bool, $result->dtype());
        $this->assertSame([true, true, false], $result->toArray());
    }

    // ==================== Bitwise Operations ====================

    public function testIntArrayBitandIntScalar(): void
    {
        $a = NDArray::array([7, 14, 21], DType::Int64);
        $result = $a->bitand(3);

        $this->assertSame(DType::Int64, $result->dtype());
        $this->assertSame([7 & 3, 14 & 3, 21 & 3], $result->toArray());
    }

    public function testIntArrayBitorIntScalar(): void
    {
        $a = NDArray::array([1, 2, 4], DType::Int64);
        $result = $a->bitor(8);

        $this->assertSame(DType::Int64, $result->dtype());
        $this->assertSame([1 | 8, 2 | 8, 4 | 8], $result->toArray());
    }

    public function testIntArrayLeftShiftScalar(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Int64);
        $result = $a->leftShift(2);

        $this->assertSame(DType::Int64, $result->dtype());
        $this->assertSame([4, 8, 12], $result->toArray());
    }

    public function testIntArrayRightShiftScalar(): void
    {
        $a = NDArray::array([8, 16, 32], DType::Int64);
        $result = $a->rightShift(2);

        $this->assertSame(DType::Int64, $result->dtype());
        $this->assertSame([2, 4, 8], $result->toArray());
    }

    // ==================== Edge Cases ====================

    public function testSameTypeNoPromotion(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Int64);
        $result = $a->add(10);

        $this->assertSame(DType::Int64, $result->dtype());
        $this->assertSame([11, 12, 13], $result->toArray());
    }

    public function testFloatArrayPlusFloatScalar(): void
    {
        $a = NDArray::array([1.0, 2.0, 3.0], DType::Float64);
        $result = $a->add(1.5);

        $this->assertSame(DType::Float64, $result->dtype());
        $this->assertEqualsWithDelta([2.5, 3.5, 4.5], $result->toArray(), 0.0001);
    }

    public function testScalarBufferNotCorrupted(): void
    {
        // This test verifies that the scalar buffer is read in its native dtype
        // If read as output dtype, int 1065353216 (0x3F800000 = 1.0f) would be corrupted
        $a = NDArray::array([100], DType::Int64);
        $result = $a->add(1.0);

        $this->assertSame(DType::Float64, $result->dtype());
        $this->assertEqualsWithDelta([101.0], $result->toArray(), 0.0001);
    }
}
