<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Unit;

use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Tests for element-wise logical operations (and, or, not, xor).
 *
 * @internal
 *
 * @coversNothing
 */
final class LogicalTest extends TestCase
{
    public function testLogicalAndBoolArrays(): void
    {
        $a = NDArray::array([[true, false], [true, false]], DType::Bool);
        $b = NDArray::array([[true, true], [false, false]], DType::Bool);

        $result = $a->and($b);

        $this->assertSame(DType::Bool, $result->dtype());
        $this->assertSame([2, 2], $result->shape());
        $this->assertSame([[true, false], [false, false]], $result->toArray());
    }

    public function testLogicalAndIntegerArrays(): void
    {
        $a = NDArray::array([[1, 0], [1, 0]], DType::Int64);
        $b = NDArray::array([[1, 1], [0, 0]], DType::Int64);

        $result = $a->and($b);

        $this->assertSame(DType::Bool, $result->dtype());
        $this->assertSame([[true, false], [false, false]], $result->toArray());
    }

    public function testLogicalAndFloatArrays(): void
    {
        $a = NDArray::array([[1.5, 0.0], [2.3, 0.0]], DType::Float64);
        $b = NDArray::array([[0.1, 3.2], [0.0, 0.0]], DType::Float64);

        $result = $a->and($b);

        $this->assertSame(DType::Bool, $result->dtype());
        $this->assertSame([[true, false], [false, false]], $result->toArray());
    }

    public function testLogicalAndMixedTypes(): void
    {
        $a = NDArray::array([[1, 0], [1, 0]], DType::Int64);
        $b = NDArray::array([[true, true], [false, false]], DType::Bool);

        $result = $a->and($b);

        $this->assertSame(DType::Bool, $result->dtype());
        $this->assertSame([[true, false], [false, false]], $result->toArray());
    }

    public function testLogicalOrBoolArrays(): void
    {
        $a = NDArray::array([[true, false], [true, false]], DType::Bool);
        $b = NDArray::array([[true, true], [false, false]], DType::Bool);

        $result = $a->or($b);

        $this->assertSame(DType::Bool, $result->dtype());
        $this->assertSame([[true, true], [true, false]], $result->toArray());
    }

    public function testLogicalOrIntegerArrays(): void
    {
        $a = NDArray::array([[1, 0], [1, 0]], DType::Int64);
        $b = NDArray::array([[1, 1], [0, 0]], DType::Int64);

        $result = $a->or($b);

        $this->assertSame(DType::Bool, $result->dtype());
        $this->assertSame([[true, true], [true, false]], $result->toArray());
    }

    public function testLogicalNotBoolArray(): void
    {
        $a = NDArray::array([[true, false], [false, true]], DType::Bool);

        $result = $a->not();

        $this->assertSame(DType::Bool, $result->dtype());
        $this->assertSame([[false, true], [true, false]], $result->toArray());
    }

    public function testLogicalNotIntegerArray(): void
    {
        $a = NDArray::array([[1, 0], [0, 2]], DType::Int64);

        $result = $a->not();

        $this->assertSame(DType::Bool, $result->dtype());
        $this->assertSame([[false, true], [true, false]], $result->toArray());
    }

    public function testLogicalNotFloatArray(): void
    {
        $a = NDArray::array([[1.5, 0.0], [0.0, 2.3]], DType::Float64);

        $result = $a->not();

        $this->assertSame(DType::Bool, $result->dtype());
        $this->assertSame([[false, true], [true, false]], $result->toArray());
    }

    public function testLogicalXorBoolArrays(): void
    {
        $a = NDArray::array([[true, false], [true, false]], DType::Bool);
        $b = NDArray::array([[true, true], [false, false]], DType::Bool);

        $result = $a->xor($b);

        $this->assertSame(DType::Bool, $result->dtype());
        $this->assertSame([[false, true], [true, false]], $result->toArray());
    }

    public function testLogicalXorIntegerArrays(): void
    {
        $a = NDArray::array([[1, 0], [1, 0]], DType::Int64);
        $b = NDArray::array([[1, 1], [0, 0]], DType::Int64);

        $result = $a->xor($b);

        $this->assertSame(DType::Bool, $result->dtype());
        $this->assertSame([[false, true], [true, false]], $result->toArray());
    }

    public function testLogicalAndBroadcasting(): void
    {
        $a = NDArray::array([[1, 0], [1, 0], [0, 1]], DType::Int64);
        $b = NDArray::array([[1, 1]], DType::Int64);

        $result = $a->and($b);

        $this->assertSame(DType::Bool, $result->dtype());
        $this->assertSame([3, 2], $result->shape());
        $this->assertSame([[true, false], [true, false], [false, true]], $result->toArray());
    }

    public function testLogicalOrBroadcasting(): void
    {
        $a = NDArray::array([[1, 0], [0, 0]], DType::Int64);
        $b = NDArray::array([[0, 1]], DType::Int64);

        $result = $a->or($b);

        $this->assertSame(DType::Bool, $result->dtype());
        $this->assertSame([[true, true], [false, true]], $result->toArray());
    }
}
