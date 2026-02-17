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

    public function testLn(): void
    {
        $a = NDArray::array([1, M_E], DType::Float64);
        $result = $a->ln();
        $this->assertEqualsWithDelta([0, 1], $result->toArray(), 0.0001);
    }

    public function testLn1p(): void
    {
        $a = NDArray::array([0, M_E - 1], DType::Float64);
        $result = $a->ln1p();
        $this->assertEqualsWithDelta([0, 1], $result->toArray(), 0.0001);
    }

    public function testToDegrees(): void
    {
        $a = NDArray::array([0, M_PI / 2, M_PI], DType::Float64);
        $result = $a->toDegrees();
        $this->assertEqualsWithDelta([0, 90, 180], $result->toArray(), 0.0001);
    }

    public function testToRadians(): void
    {
        $a = NDArray::array([0, 90, 180], DType::Float64);
        $result = $a->toRadians();
        $this->assertEqualsWithDelta([0, M_PI / 2, M_PI], $result->toArray(), 0.0001);
    }

    public function testPowi(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $result = $a->powi(3);
        $this->assertEqualsWithDelta([1, 8, 27], $result->toArray(), 0.0001);
    }

    public function testPowf(): void
    {
        $a = NDArray::array([1, 4, 9], DType::Float64);
        $result = $a->powf(0.5);
        $this->assertEqualsWithDelta([1, 2, 3], $result->toArray(), 0.0001);
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

    public function testSigmoid(): void
    {
        $a = NDArray::array([0, 1], DType::Float64);
        $result = $a->sigmoid();
        $this->assertEqualsWithDelta([0.5, 1 / (1 + exp(-1))], $result->toArray(), 0.0001);
    }

    public function testSoftmax(): void
    {
        $a = NDArray::array([1.0, 2.0, 3.0], DType::Float64);
        $result = $a->softmax();
        $expSum = exp(1) + exp(2) + exp(3);
        $expected = [exp(1) / $expSum, exp(2) / $expSum, exp(3) / $expSum];
        $this->assertEqualsWithDelta($expected, $result->toArray(), 0.0001);
        $this->assertEqualsWithDelta(1.0, array_sum($result->toArray()), 0.0001);
    }

    public function testSoftmaxAlongAxis(): void
    {
        $a = NDArray::array([[1.0, 2.0], [3.0, 4.0]], DType::Float64);
        $result = $a->softmax(axis: -1);
        $this->assertSame([2, 2], $result->shape());
        // axis -1 = last axis (rows); each row sums to 1 (NumPy semantics)
        foreach ($result->toArray() as $row) {
            $this->assertEqualsWithDelta(1.0, array_sum($row), 0.0001);
        }
    }

    public function testSoftmax3D(): void
    {
        $a = NDArray::array([
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.0, 1.0], [2.0, 3.0]],
        ], DType::Float64);
        $result = $a->softmax(axis: -1);
        $this->assertSame([2, 2, 2], $result->shape());
        $arr = $result->toArray();
        foreach ($arr as $batch) {
            foreach ($batch as $row) {
                $this->assertEqualsWithDelta(1.0, array_sum($row), 0.0001);
            }
        }
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
        $a = NDArray::array([3.0, 5.0], DType::Float64);
        $result = $a->hypot(4.0);
        $this->assertEqualsWithDelta([5.0, 6.4031], $result->toArray(), 0.0001);
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

    public function testClamp(): void
    {
        $a = NDArray::array([0, 5, 10], DType::Float64);
        $result = $a->clamp(2, 8);
        $this->assertEqualsWithDelta([2, 5, 8], $result->toArray(), 0.0001);
    }

    public function testClampValuesBelowMin(): void
    {
        $a = NDArray::array([0, 1, 2], DType::Float64);
        $result = $a->clamp(5, 10);
        $this->assertEqualsWithDelta([5, 5, 5], $result->toArray(), 0.0001);
    }

    public function testClampValuesAboveMax(): void
    {
        $a = NDArray::array([8, 9, 10], DType::Float64);
        $result = $a->clamp(0, 5);
        $this->assertEqualsWithDelta([5, 5, 5], $result->toArray(), 0.0001);
    }

    public function testClampInvalidRange(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);

        $this->expectException(\InvalidArgumentException::class);
        $a->clamp(10, 5);
    }

    public function testClampPreservesDtype(): void
    {
        $a = NDArray::array([0, 5, 10], DType::Int32);
        $result = $a->clamp(2, 8);
        $this->assertSame(DType::Int32, $result->dtype());
        $this->assertEquals([2, 5, 8], $result->toArray());
    }
}
