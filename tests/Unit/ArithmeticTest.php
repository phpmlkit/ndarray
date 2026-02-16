<?php

declare(strict_types=1);

namespace NDArray\Tests\Unit;

use NDArray\DType;
use NDArray\Exceptions\DTypeException;
use NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Tests for arithmetic operations (add, subtract, multiply, divide, negative)
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
        $a = NDArray::array([1, 2, 3], DType::Uint32);
        $a->negative();
    }
}
