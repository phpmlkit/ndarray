<?php

declare(strict_types=1);

namespace NDArray\Tests\Unit;

use NDArray\DType;
use NDArray\NDArray;
use PHPUnit\Framework\TestCase;

final class CreationTest extends TestCase
{
    public function testZeros(): void
    {
        $arr = NDArray::zeros([2, 3], DType::Float64);

        $this->assertSame([2, 3], $arr->shape());
        $this->assertSame(DType::Float64, $arr->dtype());
        $this->assertEquals([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ], $arr->toArray());
    }

    public function testZerosInt(): void
    {
        $arr = NDArray::zeros([2], DType::Int32);
        
        $this->assertSame(DType::Int32, $arr->dtype());
        $this->assertSame([0, 0], $arr->toArray());
    }

    public function testOnes(): void
    {
        $arr = NDArray::ones([2, 2], DType::Float32);

        $this->assertSame([2, 2], $arr->shape());
        $this->assertSame(DType::Float32, $arr->dtype());
        $this->assertEqualsWithDelta([
            [1.0, 1.0],
            [1.0, 1.0],
        ], $arr->toArray(), 0.0001);
    }

    public function testFull(): void
    {
        $arr = NDArray::full([3], 42.5, DType::Float64);

        $this->assertSame([3], $arr->shape());
        $this->assertSame(DType::Float64, $arr->dtype());
        $this->assertEquals([42.5, 42.5, 42.5], $arr->toArray());
    }

    public function testFullInferred(): void
    {
        $arr = NDArray::full([2], 100);
        $this->assertSame(DType::Int64, $arr->dtype());
        $this->assertSame([100, 100], $arr->toArray());

        $arrBool = NDArray::full([2], true);
        $this->assertSame(DType::Bool, $arrBool->dtype());
        $this->assertSame([true, true], $arrBool->toArray());
    }

    public function testEye(): void
    {
        $arr = NDArray::eye(3);

        $this->assertSame([3, 3], $arr->shape());
        $this->assertEquals([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], $arr->toArray());
    }

    public function testEyeRectangle(): void
    {
        $arr = NDArray::eye(2, 3);
        
        $this->assertSame([2, 3], $arr->shape());
        $this->assertEquals([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], $arr->toArray());
    }

    public function testEyeOffset(): void
    {
        // Upper diagonal (k=1)
        $arr = NDArray::eye(3, 3, 1);
        $this->assertEquals([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ], $arr->toArray());

        // Lower diagonal (k=-1)
        $arr = NDArray::eye(3, 3, -1);
        $this->assertEquals([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], $arr->toArray());
    }
}
