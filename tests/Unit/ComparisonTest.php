<?php

declare(strict_types=1);

namespace NDArray\Tests\Unit;

use NDArray\DType;
use NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Tests for element-wise comparison operations (eq, ne, gt, gte, lt, lte)
 */
final class ComparisonTest extends TestCase
{
    public function testEqArraysSameShape(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);
        $b = NDArray::array([[1, 2], [3, 5]], DType::Float64);

        $result = $a->eq($b);

        $this->assertSame(DType::Bool, $result->dtype());
        $this->assertSame([2, 2], $result->shape());
        $this->assertSame([[true, true], [true, false]], $result->toArray());
    }

    public function testEqScalar(): void
    {
        $a = NDArray::array([[1, 2], [3, 2]], DType::Float64);

        $result = $a->eq(2);

        $this->assertSame(DType::Bool, $result->dtype());
        $this->assertSame([[false, true], [false, true]], $result->toArray());
    }

    public function testNeArrays(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);
        $b = NDArray::array([[1, 2], [3, 5]], DType::Float64);

        $result = $a->ne($b);

        $this->assertSame(DType::Bool, $result->dtype());
        $this->assertSame([[false, false], [false, true]], $result->toArray());
    }

    public function testNeScalar(): void
    {
        $a = NDArray::array([[1, 2], [3, 2]], DType::Float64);

        $result = $a->ne(2);

        $this->assertSame([[true, false], [true, false]], $result->toArray());
    }

    public function testGtArrays(): void
    {
        $a = NDArray::array([[1, 4], [3, 2]], DType::Float64);
        $b = NDArray::array([[2, 3], [3, 1]], DType::Float64);

        $result = $a->gt($b);

        $this->assertSame(DType::Bool, $result->dtype());
        $this->assertSame([[false, true], [false, true]], $result->toArray());
    }

    public function testGtScalar(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);

        $result = $a->gt(2);

        $this->assertSame([[false, false], [true, true]], $result->toArray());
    }

    public function testGteArrays(): void
    {
        $a = NDArray::array([[1, 4], [3, 2]], DType::Float64);
        $b = NDArray::array([[2, 3], [3, 1]], DType::Float64);

        $result = $a->gte($b);

        $this->assertSame([[false, true], [true, true]], $result->toArray());
    }

    public function testLtArrays(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);
        $b = NDArray::array([[2, 1], [3, 5]], DType::Float64);

        $result = $a->lt($b);

        $this->assertSame([[true, false], [false, true]], $result->toArray());
    }

    public function testLteArrays(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);
        $b = NDArray::array([[2, 1], [3, 5]], DType::Float64);

        $result = $a->lte($b);

        $this->assertSame([[true, false], [true, true]], $result->toArray());
    }

    public function testEqBroadcasting(): void
    {
        $a = NDArray::array([[1., 1.], [1., 2.], [0., 3.], [0., 4.]], DType::Float64);
        $b = NDArray::array([[0., 1.]], DType::Float64);

        $result = $a->eq($b);

        $this->assertSame(DType::Bool, $result->dtype());
        $this->assertSame([4, 2], $result->shape());
        $this->assertSame([[false, true], [false, false], [true, false], [true, false]], $result->toArray());
    }

    public function testGtBroadcasting(): void
    {
        $a = NDArray::array([[1., 1.], [1., 2.], [0., 3.], [0., 4.]], DType::Float64);
        $b = NDArray::array([[0., 1.]], DType::Float64);

        $result = $a->gt($b);

        $this->assertSame([4, 2], $result->shape());
        $this->assertSame([[true, false], [true, true], [false, true], [false, true]], $result->toArray());
    }
}
