<?php

declare(strict_types=1);

namespace NDArray\Tests\Unit;

use NDArray\DType;
use NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Tests for linear algebra operations.
 */
class LinearAlgebraTest extends TestCase
{
    public function testDot1D(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $b = NDArray::array([4, 5, 6], DType::Float64);
        $result = $a->dot($b);

        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        $this->assertEqualsWithDelta(32, $result->toArray(), 0.0001);
    }

    public function testDot2D(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);
        $b = NDArray::array([[5, 6], [7, 8]], DType::Float64);
        $result = $a->dot($b);

        // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        // = [[19, 22], [43, 50]]
        $this->assertEqualsWithDelta([[19, 22], [43, 50]], $result->toArray(), 0.0001);
    }

    public function testDot2D1D(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);
        $b = NDArray::array([5, 6], DType::Float64);
        $result = $a->dot($b);

        // [1*5+2*6, 3*5+4*6] = [17, 39]
        $this->assertEqualsWithDelta([17, 39], $result->toArray(), 0.0001);
    }

    public function testDot1D2D(): void
    {
        $a = NDArray::array([1, 2], DType::Float64);
        $b = NDArray::array([[5, 6], [7, 8]], DType::Float64);
        $result = $a->dot($b);

        // [1*5+2*7, 1*6+2*8] = [19, 22]
        $this->assertEqualsWithDelta([19, 22], $result->toArray(), 0.0001);
    }

    public function testDotShapeMismatch(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $b = NDArray::array([4, 5], DType::Float64);

        $this->expectException(\NDArray\Exceptions\ShapeException::class);
        $a->dot($b);
    }

    public function testMatmul2D(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);
        $b = NDArray::array([[5, 6], [7, 8]], DType::Float64);
        $result = $a->matmul($b);

        $this->assertEqualsWithDelta([[19, 22], [43, 50]], $result->toArray(), 0.0001);
    }

    public function testMatmulRequires2D(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $b = NDArray::array([[1, 2], [3, 4]], DType::Float64);

        $this->expectException(\NDArray\Exceptions\ShapeException::class);
        $a->matmul($b);
    }

    public function testDiagonal(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], DType::Float64);
        $result = $a->diagonal();

        $this->assertEquals([1, 5, 9], $result->toArray());
    }

    public function testDiagonalNonSquare(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->diagonal();

        // min(2, 3) = 2 elements
        $this->assertEquals([1, 5], $result->toArray());
    }

    public function testDiagonalRequires2D(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);

        $this->expectException(\NDArray\Exceptions\ShapeException::class);
        $a->diagonal();
    }

    public function testTrace(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], DType::Float64);
        $result = $a->trace();

        // 1 + 5 + 9 = 15
        $this->assertEqualsWithDelta(15, $result->toArray(), 0.0001);
    }

    public function testTraceNonSquare(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->trace();

        // 1 + 5 = 6
        $this->assertEqualsWithDelta(6, $result->toArray(), 0.0001);
    }

    public function testTraceRequires2D(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);

        $this->expectException(\NDArray\Exceptions\ShapeException::class);
        $a->trace();
    }

    public function testNormScalarL2(): void
    {
        $a = NDArray::array([3, 4], DType::Float64);
        $this->assertEqualsWithDelta(5.0, $a->norm(2), 1e-10);
    }

    public function testNormScalarL1AndInf(): void
    {
        $a = NDArray::array([-3, 4, -5], DType::Float64);
        $this->assertEqualsWithDelta(12.0, $a->norm(1), 1e-10);
        $this->assertEqualsWithDelta(5.0, $a->norm(INF), 1e-10);
        $this->assertEqualsWithDelta(3.0, $a->norm(-INF), 1e-10);
    }

    public function testNormAxisL2(): void
    {
        $a = NDArray::array([[3, 4], [5, 12]], DType::Float64);
        $result = $a->norm(2, axis: 1);
        $this->assertEqualsWithDelta([5.0, 13.0], $result->toArray(), 1e-10);
    }

    public function testNormAxisKeepdims(): void
    {
        $a = NDArray::array([[1, -2, 3], [4, -5, 6]], DType::Float64);
        $result = $a->norm(1, axis: 1, keepdims: true);
        $this->assertSame([2, 1], $result->shape());
        $this->assertEqualsWithDelta([[6.0], [15.0]], $result->toArray(), 1e-10);
    }

    public function testNormFrobeniusDefaultForMatrix(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);
        $expected = sqrt(1 + 4 + 9 + 16);
        $this->assertEqualsWithDelta($expected, $a->norm(), 1e-10);
        $this->assertEqualsWithDelta($expected, $a->norm('fro'), 1e-10);
    }

    public function testNormFroRequire2D(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $this->expectException(\InvalidArgumentException::class);
        $a->norm('fro');
    }
}
