<?php

declare(strict_types=1);

namespace NDArray\Tests\Unit;

use NDArray\DType;
use NDArray\Exceptions\ShapeException;
use NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Tests for conversion operations (toArray, toScalar, copy, astype).
 */
final class ConversionTest extends TestCase
{
    public function testToScalarFloat64(): void
    {
        $a = NDArray::array([3.14], DType::Float64)->reshape([]);

        $this->assertSame(0, $a->ndim());
        $this->assertSame([], $a->shape());

        $scalar = $a->toScalar();

        $this->assertIsFloat($scalar);
        $this->assertEqualsWithDelta(3.14, $scalar, 0.0001);
    }

    public function testToScalarFloat32(): void
    {
        $a = NDArray::array([2.5], DType::Float32)->reshape([]);

        $scalar = $a->toScalar();

        $this->assertIsFloat($scalar);
        $this->assertEqualsWithDelta(2.5, $scalar, 0.0001);
    }

    public function testToScalarInt64(): void
    {
        $a = NDArray::array([42], DType::Int64)->reshape([]);

        $scalar = $a->toScalar();

        $this->assertIsInt($scalar);
        $this->assertSame(42, $scalar);
    }

    public function testToScalarInt32(): void
    {
        $a = NDArray::array([-100], DType::Int32)->reshape([]);

        $scalar = $a->toScalar();

        $this->assertIsInt($scalar);
        $this->assertSame(-100, $scalar);
    }

    public function testToScalarBool(): void
    {
        $a = NDArray::full([], true, DType::Bool);

        $scalar = $a->toScalar();

        $this->assertIsBool($scalar);
        $this->assertTrue($scalar);
    }

    public function testToScalarBoolFalse(): void
    {
        $a = NDArray::full([], false, DType::Bool);

        $scalar = $a->toScalar();

        $this->assertIsBool($scalar);
        $this->assertFalse($scalar);
    }

    public function testToScalarFromFull(): void
    {
        $a = NDArray::full([], 7.5, DType::Float64);

        $scalar = $a->toScalar();

        $this->assertIsFloat($scalar);
        $this->assertEqualsWithDelta(7.5, $scalar, 0.0001);
    }

    public function testToScalarFromView(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5], DType::Int64);
        $view = $a->slice(['2:3']);  // [3]
        $scalarView = $view->reshape([]);  // 0-d view of element 3

        $scalar = $scalarView->toScalar();

        $this->assertIsInt($scalar);
        $this->assertSame(3, $scalar);
    }

    public function testToScalarNonZeroDimThrows(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);

        $this->expectException(ShapeException::class);
        $this->expectExceptionMessage('toScalar requires a 0-dimensional array');
        $a->toScalar();
    }

    public function testToScalar2DThrows(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);

        $this->expectException(ShapeException::class);
        $this->expectExceptionMessage('toScalar requires a 0-dimensional array');
        $a->toScalar();
    }
}
