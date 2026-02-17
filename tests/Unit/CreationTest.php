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

    // =========================================================================
    // arange Tests
    // =========================================================================

    public function testArangeBasic(): void
    {
        $arr = NDArray::arange(0, 5);

        $this->assertSame([5], $arr->shape());
        $this->assertSame(DType::Int64, $arr->dtype());
        $this->assertSame([0, 1, 2, 3, 4], $arr->toArray());
    }

    public function testArangeWithStart(): void
    {
        $arr = NDArray::arange(2, 6);

        $this->assertSame([4], $arr->shape());
        $this->assertSame([2, 3, 4, 5], $arr->toArray());
    }

    public function testArangeWithStep(): void
    {
        $arr = NDArray::arange(0, 10, 2);

        $this->assertSame([5], $arr->shape());
        $this->assertSame([0, 2, 4, 6, 8], $arr->toArray());
    }

    public function testArangeNegativeStep(): void
    {
        $arr = NDArray::arange(10, 0, -2);

        $this->assertSame([5], $arr->shape());
        $this->assertSame([10, 8, 6, 4, 2], $arr->toArray());
    }

    public function testArangeFloat(): void
    {
        $arr = NDArray::arange(0.0, 1.0, 0.25, DType::Float64);

        $this->assertSame([4], $arr->shape());
        $this->assertSame(DType::Float64, $arr->dtype());
        $this->assertEqualsWithDelta([0.0, 0.25, 0.5, 0.75], $arr->toArray(), 0.0001);
    }

    public function testArangeEmpty(): void
    {
        // Start >= stop with positive step
        $arr = NDArray::arange(5, 0);
        $this->assertSame([0], $arr->shape());
        $this->assertSame([], $arr->toArray());

        // Start <= stop with negative step
        $arr = NDArray::arange(0, 5, -1);
        $this->assertSame([0], $arr->shape());
    }

    public function testArangeInt32(): void
    {
        $arr = NDArray::arange(0, 5, 1, DType::Int32);

        $this->assertSame(DType::Int32, $arr->dtype());
        $this->assertSame([0, 1, 2, 3, 4], $arr->toArray());
    }

    public function testArangeFloat32(): void
    {
        $arr = NDArray::arange(0.0, 2.0, 0.5, DType::Float32);

        $this->assertSame(DType::Float32, $arr->dtype());
        $this->assertEqualsWithDelta([0.0, 0.5, 1.0, 1.5], $arr->toArray(), 0.0001);
    }

    public function testArangeZeroStepThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        NDArray::arange(0, 5, 0);
    }

    public function testArangeBoolThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        NDArray::arange(0, 5, 1, DType::Bool);
    }

    // =========================================================================
    // linspace Tests
    // =========================================================================

    public function testLinspaceDefault(): void
    {
        $arr = NDArray::linspace(0.0, 1.0);

        $this->assertSame([50], $arr->shape());
        $this->assertSame(DType::Float64, $arr->dtype());

        // Check first and last values
        $data = $arr->toArray();
        $this->assertEqualsWithDelta(0.0, $data[0], 0.0001);
        $this->assertEqualsWithDelta(1.0, $data[49], 0.0001);
    }

    public function testLinspaceFivePoints(): void
    {
        $arr = NDArray::linspace(0.0, 1.0, 5);

        $this->assertSame([5], $arr->shape());
        $this->assertEqualsWithDelta([0.0, 0.25, 0.5, 0.75, 1.0], $arr->toArray(), 0.0001);
    }

    public function testLinspaceWithoutEndpoint(): void
    {
        $arr = NDArray::linspace(0.0, 1.0, 5, false);

        $this->assertSame([5], $arr->shape());
        $this->assertEqualsWithDelta([0.0, 0.2, 0.4, 0.6, 0.8], $arr->toArray(), 0.0001);
    }

    public function testLinspaceSinglePoint(): void
    {
        $arr = NDArray::linspace(0.0, 1.0, 1);

        $this->assertSame([1], $arr->shape());
        $this->assertEqualsWithDelta([0.0], $arr->toArray(), 0.0001);
    }

    public function testLinspaceFloat32(): void
    {
        $arr = NDArray::linspace(0.0, 1.0, 5, true, DType::Float32);

        $this->assertSame(DType::Float32, $arr->dtype());
        $this->assertEqualsWithDelta([0.0, 0.25, 0.5, 0.75, 1.0], $arr->toArray(), 0.0001);
    }

    public function testLinspaceNegativeThrows(): void
    {
        $this->expectException(\NDArray\Exceptions\ShapeException::class);
        NDArray::linspace(0.0, 1.0, 0);
    }

    public function testLinspaceIntThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        NDArray::linspace(0.0, 1.0, 5, true, DType::Int32);
    }

    // =========================================================================
    // logspace Tests
    // =========================================================================

    public function testLogspaceDefault(): void
    {
        $arr = NDArray::logspace(0.0, 2.0);

        $this->assertSame([50], $arr->shape());
        $this->assertSame(DType::Float64, $arr->dtype());

        // Default base is 10, so: 10^0=1 to 10^2=100
        $data = $arr->toArray();
        $this->assertEqualsWithDelta(1.0, $data[0], 0.0001);
        $this->assertEqualsWithDelta(100.0, $data[49], 0.0001);
    }

    public function testLogspaceFivePoints(): void
    {
        $arr = NDArray::logspace(0.0, 2.0, 5);

        $this->assertSame([5], $arr->shape());
        // base=10, exponents: 0, 0.5, 1, 1.5, 2
        // values: 10^0=1, 10^0.5≈3.16, 10^1=10, 10^1.5≈31.6, 10^2=100
        $data = $arr->toArray();
        $this->assertEqualsWithDelta(1.0, $data[0], 0.01);
        $this->assertEqualsWithDelta(10.0, $data[2], 0.01);
        $this->assertEqualsWithDelta(100.0, $data[4], 0.01);
    }

    public function testLogspaceBase2(): void
    {
        $arr = NDArray::logspace(0.0, 3.0, 4, 2.0);

        $this->assertSame([4], $arr->shape());
        // base=2, exponents: 0, 1, 2, 3
        // values: 2^0=1, 2^1=2, 2^2=4, 2^3=8
        $this->assertEqualsWithDelta([1.0, 2.0, 4.0, 8.0], $arr->toArray(), 0.0001);
    }

    public function testLogspaceSinglePoint(): void
    {
        $arr = NDArray::logspace(1.0, 2.0, 1);

        $this->assertSame([1], $arr->shape());
        $this->assertEqualsWithDelta([10.0], $arr->toArray(), 0.0001);
    }

    public function testLogspaceFloat32(): void
    {
        $arr = NDArray::logspace(0.0, 2.0, 5, 10.0, DType::Float32);

        $this->assertSame(DType::Float32, $arr->dtype());
        $data = $arr->toArray();
        $this->assertEqualsWithDelta(1.0, $data[0], 0.01);
        $this->assertEqualsWithDelta(100.0, $data[4], 0.01);
    }

    public function testLogspaceNegativeThrows(): void
    {
        $this->expectException(\NDArray\Exceptions\ShapeException::class);
        NDArray::logspace(0.0, 1.0, 0);
    }

    public function testLogspaceIntThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        NDArray::logspace(0.0, 1.0, 5, 10.0, DType::Int32);
    }

    // =========================================================================
    // geomspace Tests
    // =========================================================================

    public function testGeomspaceDefault(): void
    {
        $arr = NDArray::geomspace(1.0, 1000.0);

        $this->assertSame([50], $arr->shape());
        $this->assertSame(DType::Float64, $arr->dtype());

        $data = $arr->toArray();
        $this->assertEqualsWithDelta(1.0, $data[0], 0.0001);
        $this->assertEqualsWithDelta(1000.0, $data[49], 0.0001);
    }

    public function testGeomspaceFourPoints(): void
    {
        $arr = NDArray::geomspace(1.0, 1000.0, 4);

        $this->assertSame([4], $arr->shape());
        // ratio = (1000/1)^(1/3) = 10
        // values: 1, 10, 100, 1000
        $this->assertEqualsWithDelta([1.0, 10.0, 100.0, 1000.0], $arr->toArray(), 0.0001);
    }

    public function testGeomspaceNegativeValues(): void
    {
        $arr = NDArray::geomspace(-1000.0, -1.0, 4);

        $this->assertSame([4], $arr->shape());
        // Both negative, ratio = (-1/-1000)^(1/3) = (0.001)^(1/3) = 0.1
        // values: -1000, -100, -10, -1
        $this->assertEqualsWithDelta([-1000.0, -100.0, -10.0, -1.0], $arr->toArray(), 0.0001);
    }

    public function testGeomspaceSinglePoint(): void
    {
        $arr = NDArray::geomspace(1.0, 100.0, 1);

        $this->assertSame([1], $arr->shape());
        $this->assertEqualsWithDelta([1.0], $arr->toArray(), 0.0001);
    }

    public function testGeomspaceFloat32(): void
    {
        $arr = NDArray::geomspace(1.0, 1000.0, 4, DType::Float32);

        $this->assertSame(DType::Float32, $arr->dtype());
        $this->assertEqualsWithDelta([1.0, 10.0, 100.0, 1000.0], $arr->toArray(), 0.01);
    }

    public function testGeomspaceZeroStartThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        NDArray::geomspace(0.0, 100.0, 5);
    }

    public function testGeomspaceDifferentSignsThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        NDArray::geomspace(-1.0, 100.0, 5);
    }

    public function testGeomspaceNegativeThrows(): void
    {
        $this->expectException(\NDArray\Exceptions\ShapeException::class);
        NDArray::geomspace(1.0, 100.0, 0);
    }

    public function testGeomspaceIntThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        NDArray::geomspace(1.0, 100.0, 5, DType::Int32);
    }

    // =========================================================================
    // Random Array Creation Tests (REQ-3.3)
    // =========================================================================

    public function testRandomBasicAndRange(): void
    {
        $arr = NDArray::random([2, 3], DType::Float64, seed: 1234);
        $this->assertSame([2, 3], $arr->shape());
        $this->assertSame(DType::Float64, $arr->dtype());

        foreach ($arr->toFlatArray() as $v) {
            $this->assertGreaterThanOrEqual(0.0, $v);
            $this->assertLessThan(1.0, $v);
        }
    }

    public function testRandomSeedDeterministic(): void
    {
        $a = NDArray::random([8], DType::Float32, seed: 7)->toArray();
        $b = NDArray::random([8], DType::Float32, seed: 7)->toArray();
        $c = NDArray::random([8], DType::Float32, seed: 8)->toArray();

        $this->assertSame($a, $b);
        $this->assertNotSame($a, $c);
    }

    public function testRandomFloatOnlyThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        NDArray::random([4], DType::Int32);
    }

    public function testRandomIntRangeAndDtype(): void
    {
        $arr = NDArray::randomInt(10, 20, [100], DType::Int32, seed: 42);
        $this->assertSame([100], $arr->shape());
        $this->assertSame(DType::Int32, $arr->dtype());

        foreach ($arr->toFlatArray() as $v) {
            $this->assertGreaterThanOrEqual(10, $v);
            $this->assertLessThan(20, $v);
        }
    }

    public function testRandomIntSeedDeterministic(): void
    {
        $a = NDArray::randomInt(0, 100, [12], DType::Int64, seed: 11)->toArray();
        $b = NDArray::randomInt(0, 100, [12], DType::Int64, seed: 11)->toArray();
        $this->assertSame($a, $b);
    }

    public function testRandomIntRejectsInvalidArgs(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        NDArray::randomInt(5, 5, [2], DType::Int64);
    }

    public function testRandnStatisticalSanity(): void
    {
        $arr = NDArray::randn([20000], DType::Float64, seed: 77);
        $flat = $arr->toFlatArray();

        $mean = array_sum($flat) / count($flat);
        $variance = array_sum(array_map(
            static fn (float $x): float => ($x - $mean) ** 2,
            $flat
        )) / count($flat);
        $std = sqrt($variance);

        $this->assertEqualsWithDelta(0.0, $mean, 0.05);
        $this->assertEqualsWithDelta(1.0, $std, 0.08);
    }

    public function testNormalStatisticalSanity(): void
    {
        $meanTarget = 3.5;
        $stdTarget = 2.0;
        $arr = NDArray::normal($meanTarget, $stdTarget, [20000], DType::Float64, seed: 91);
        $flat = $arr->toFlatArray();

        $mean = array_sum($flat) / count($flat);
        $variance = array_sum(array_map(
            static fn (float $x): float => ($x - $mean) ** 2,
            $flat
        )) / count($flat);
        $std = sqrt($variance);

        $this->assertEqualsWithDelta($meanTarget, $mean, 0.1);
        $this->assertEqualsWithDelta($stdTarget, $std, 0.1);
    }

    public function testNormalRequiresPositiveStd(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        NDArray::normal(0.0, 0.0, [4]);
    }

    public function testUniformRangeAndMeanSanity(): void
    {
        $low = -2.0;
        $high = 6.0;
        $arr = NDArray::uniform($low, $high, [20000], DType::Float64, seed: 123);
        $flat = $arr->toFlatArray();

        foreach ($flat as $v) {
            $this->assertGreaterThanOrEqual($low, $v);
            $this->assertLessThan($high, $v);
        }

        $mean = array_sum($flat) / count($flat);
        $this->assertEqualsWithDelta(($low + $high) / 2.0, $mean, 0.08);
    }

    public function testUniformRequiresHighGreaterThanLow(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        NDArray::uniform(1.0, 1.0, [4]);
    }
}
