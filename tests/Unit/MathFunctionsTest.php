<?php

declare(strict_types=1);

namespace NDArray\Tests\Unit;

use NDArray\DType;
use NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Tests for element-wise math operations
 */
final class MathFunctionsTest extends TestCase
{
    public function testAbs(): void
    {
        $a = NDArray::array([[-1, 2], [-3, 4]], DType::Float64);
        $result = $a->abs();
        $this->assertEqualsWithDelta([[1, 2], [3, 4]], $result->toArray(), 0.0001);
    }

    public function testSqrt(): void
    {
        $a = NDArray::array([[1, 4], [9, 16]], DType::Float64);
        $result = $a->sqrt();
        $this->assertEqualsWithDelta([[1, 2], [3, 4]], $result->toArray(), 0.0001);
    }

    public function testExp(): void
    {
        $a = NDArray::array([0, 1], DType::Float64);
        $result = $a->exp();
        $this->assertEqualsWithDelta([1, M_E], $result->toArray(), 0.0001);
    }

    public function testLog(): void
    {
        $a = NDArray::array([1, M_E], DType::Float64);
        $result = $a->log();
        $this->assertEqualsWithDelta([0, 1], $result->toArray(), 0.0001);
    }

    public function testSin(): void
    {
        $a = NDArray::array([0, M_PI / 2], DType::Float64);
        $result = $a->sin();
        $this->assertEqualsWithDelta([0, 1], $result->toArray(), 0.0001);
    }

    public function testCos(): void
    {
        $a = NDArray::array([0, M_PI], DType::Float64);
        $result = $a->cos();
        $this->assertEqualsWithDelta([1, -1], $result->toArray(), 0.0001);
    }

    public function testTan(): void
    {
        $a = NDArray::array([0, M_PI / 4], DType::Float64);
        $result = $a->tan();
        $this->assertEqualsWithDelta([0, 1], $result->toArray(), 0.0001);
    }

    public function testSinh(): void
    {
        $a = NDArray::array([0, 1], DType::Float64);
        $result = $a->sinh();
        $this->assertEqualsWithDelta([0, sinh(1)], $result->toArray(), 0.0001);
    }

    public function testCosh(): void
    {
        $a = NDArray::array([0, 1], DType::Float64);
        $result = $a->cosh();
        $this->assertEqualsWithDelta([1, cosh(1)], $result->toArray(), 0.0001);
    }

    public function testTanh(): void
    {
        $a = NDArray::array([0, 1], DType::Float64);
        $result = $a->tanh();
        $this->assertEqualsWithDelta([0, tanh(1)], $result->toArray(), 0.0001);
    }

    public function testAsin(): void
    {
        $a = NDArray::array([0, 1], DType::Float64);
        $result = $a->asin();
        $this->assertEqualsWithDelta([0, M_PI / 2], $result->toArray(), 0.0001);
    }

    public function testAcos(): void
    {
        $a = NDArray::array([1, 0], DType::Float64);
        $result = $a->acos();
        $this->assertEqualsWithDelta([0, M_PI / 2], $result->toArray(), 0.0001);
    }

    public function testAtan(): void
    {
        $a = NDArray::array([0, 1], DType::Float64);
        $result = $a->atan();
        $this->assertEqualsWithDelta([0, M_PI / 4], $result->toArray(), 0.0001);
    }

    public function testCbrt(): void
    {
        $a = NDArray::array([1, 8, 27], DType::Float64);
        $result = $a->cbrt();
        $this->assertEqualsWithDelta([1, 2, 3], $result->toArray(), 0.0001);
    }

    public function testCeil(): void
    {
        $a = NDArray::array([1.1, 2.9, -1.1], DType::Float64);
        $result = $a->ceil();
        $this->assertEqualsWithDelta([2, 3, -1], $result->toArray(), 0.0001);
    }

    public function testExp2(): void
    {
        $a = NDArray::array([0, 1, 2, 3], DType::Float64);
        $result = $a->exp2();
        $this->assertEqualsWithDelta([1, 2, 4, 8], $result->toArray(), 0.0001);
    }

    public function testFloor(): void
    {
        $a = NDArray::array([1.9, 2.1, -1.1], DType::Float64);
        $result = $a->floor();
        $this->assertEqualsWithDelta([1, 2, -2], $result->toArray(), 0.0001);
    }

    public function testLog2(): void
    {
        $a = NDArray::array([1, 2, 4, 8], DType::Float64);
        $result = $a->log2();
        $this->assertEqualsWithDelta([0, 1, 2, 3], $result->toArray(), 0.0001);
    }

    public function testLog10(): void
    {
        $a = NDArray::array([1, 10, 100], DType::Float64);
        $result = $a->log10();
        $this->assertEqualsWithDelta([0, 1, 2], $result->toArray(), 0.0001);
    }

    public function testPow2(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $result = $a->pow2();
        $this->assertEqualsWithDelta([1, 4, 9], $result->toArray(), 0.0001);
    }

    public function testRound(): void
    {
        $a = NDArray::array([1.4, 1.6, -1.4, -1.6], DType::Float64);
        $result = $a->round();
        $this->assertEqualsWithDelta([1, 2, -1, -2], $result->toArray(), 0.0001);
    }

    public function testSignum(): void
    {
        $a = NDArray::array([-5, 5], DType::Float64);
        $result = $a->signum();
        $this->assertEqualsWithDelta([-1, 1], $result->toArray(), 0.0001);
    }

    public function testRecip(): void
    {
        $a = NDArray::array([1, 2, 0.5], DType::Float64);
        $result = $a->recip();
        $this->assertEqualsWithDelta([1, 0.5, 2], $result->toArray(), 0.0001);
    }

    public function testHypot(): void
    {
        $a = NDArray::array([3, 5], DType::Float64);
        $b = NDArray::array([4, 12], DType::Float64);
        $result = $a->hypot($b);
        $this->assertEqualsWithDelta([5, 13], $result->toArray(), 0.0001);
    }

    public function testMathOperationsPreserveShape(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->sqrt();
        $this->assertSame([2, 3], $result->shape());
    }

    public function testMathOperationsOnFloat32(): void
    {
        $a = NDArray::array([1, 4, 9], DType::Float32);
        $result = $a->sqrt();
        $this->assertSame(DType::Float32, $result->dtype());
        $this->assertEqualsWithDelta([1, 2, 3], $result->toArray(), 0.0001);
    }

    public function testChainedMathOperations(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $result = $a->pow2()->sqrt();
        $this->assertEqualsWithDelta([1, 2, 3], $result->toArray(), 0.0001);
    }
}
