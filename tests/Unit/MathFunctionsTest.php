<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Unit;

use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Tests for element-wise math operations.
 *
 * @internal
 *
 * @coversNothing
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
        $this->assertEqualsWithDelta([1, \M_E], $result->toArray(), 0.0001);
    }

    public function testLog(): void
    {
        $a = NDArray::array([1, \M_E], DType::Float64);
        $result = $a->log();
        $this->assertEqualsWithDelta([0, 1], $result->toArray(), 0.0001);
    }

    public function testLn(): void
    {
        $a = NDArray::array([1, \M_E], DType::Float64);
        $result = $a->ln();
        $this->assertEqualsWithDelta([0, 1], $result->toArray(), 0.0001);
    }

    public function testLn1p(): void
    {
        $a = NDArray::array([0, \M_E - 1], DType::Float64);
        $result = $a->ln1p();
        $this->assertEqualsWithDelta([0, 1], $result->toArray(), 0.0001);
    }

    public function testToDegrees(): void
    {
        $a = NDArray::array([0, \M_PI / 2, \M_PI], DType::Float64);
        $result = $a->toDegrees();
        $this->assertEqualsWithDelta([0, 90, 180], $result->toArray(), 0.0001);
    }

    public function testToRadians(): void
    {
        $a = NDArray::array([0, 90, 180], DType::Float64);
        $result = $a->toRadians();
        $this->assertEqualsWithDelta([0, \M_PI / 2, \M_PI], $result->toArray(), 0.0001);
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
        $a = NDArray::array([0, \M_PI / 2], DType::Float64);
        $result = $a->sin();
        $this->assertEqualsWithDelta([0, 1], $result->toArray(), 0.0001);
    }

    public function testCos(): void
    {
        $a = NDArray::array([0, \M_PI], DType::Float64);
        $result = $a->cos();
        $this->assertEqualsWithDelta([1, -1], $result->toArray(), 0.0001);
    }

    public function testTan(): void
    {
        $a = NDArray::array([0, \M_PI / 4], DType::Float64);
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
        $this->assertEqualsWithDelta([0, \M_PI / 2], $result->toArray(), 0.0001);
    }

    public function testAcos(): void
    {
        $a = NDArray::array([1, 0], DType::Float64);
        $result = $a->acos();
        $this->assertEqualsWithDelta([0, \M_PI / 2], $result->toArray(), 0.0001);
    }

    public function testAtan(): void
    {
        $a = NDArray::array([0, 1], DType::Float64);
        $result = $a->atan();
        $this->assertEqualsWithDelta([0, \M_PI / 4], $result->toArray(), 0.0001);
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

    public function testMinimum(): void
    {
        $a = NDArray::array([1, 5, 3, 8], DType::Float64);
        $b = NDArray::array([2, 4, 6, 7], DType::Float64);
        $result = $a->minimum($b);
        $this->assertEqualsWithDelta([1, 4, 3, 7], $result->toArray(), 0.0001);
    }

    public function testMaximum(): void
    {
        $a = NDArray::array([1, 5, 3, 8], DType::Float64);
        $b = NDArray::array([2, 4, 6, 7], DType::Float64);
        $result = $a->maximum($b);
        $this->assertEqualsWithDelta([2, 5, 6, 8], $result->toArray(), 0.0001);
    }

    public function testMinimumPreservesDtype(): void
    {
        $a = NDArray::array([1, 5, 3], DType::Int32);
        $b = NDArray::array([2, 4, 6], DType::Int32);
        $result = $a->minimum($b);
        $this->assertSame(DType::Int32, $result->dtype());
        $this->assertEquals([1, 4, 3], $result->toArray());
    }

    public function testMaximumPreservesDtype(): void
    {
        $a = NDArray::array([1, 5, 3], DType::Int32);
        $b = NDArray::array([2, 4, 6], DType::Int32);
        $result = $a->maximum($b);
        $this->assertSame(DType::Int32, $result->dtype());
        $this->assertEquals([2, 5, 6], $result->toArray());
    }

    public function testMinimumWithBroadcasting(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $b = NDArray::array([2, 2, 2], DType::Float64);
        $result = $a->minimum($b);
        $this->assertSame([2, 3], $result->shape());
        $this->assertEqualsWithDelta([[1, 2, 2], [2, 2, 2]], $result->toArray(), 0.0001);
    }

    public function testMaximumWithBroadcasting(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $b = NDArray::array([2, 2, 2], DType::Float64);
        $result = $a->maximum($b);
        $this->assertSame([2, 3], $result->shape());
        $this->assertEqualsWithDelta([[2, 2, 3], [4, 5, 6]], $result->toArray(), 0.0001);
    }

    public function testMinimumWithNegativeNumbers(): void
    {
        $a = NDArray::array([-5, -2, 3, 0], DType::Float64);
        $b = NDArray::array([-3, -4, 1, 2], DType::Float64);
        $result = $a->minimum($b);
        $this->assertEqualsWithDelta([-5, -4, 1, 0], $result->toArray(), 0.0001);
    }

    public function testMaximumWithNegativeNumbers(): void
    {
        $a = NDArray::array([-5, -2, 3, 0], DType::Float64);
        $b = NDArray::array([-3, -4, 1, 2], DType::Float64);
        $result = $a->maximum($b);
        $this->assertEqualsWithDelta([-3, -2, 3, 2], $result->toArray(), 0.0001);
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

    // VIEW/SUBSET TESTS (from MathFunctionsViewTest)

    // ========================================================================
    // Generic Math Operations (abs, signum, pow2) on Views
    // ========================================================================

    public function testAbsOn1DSlice(): void
    {
        $a = NDArray::array([-1, 2, -3, 4, -5, 6], DType::Float64);
        $slice = $a->slice(['1:4']); // [2, -3, 4]

        $result = $slice->abs();

        $this->assertSame([3], $result->shape());
        $this->assertEqualsWithDelta([2, 3, 4], $result->toArray(), 0.0001);
    }

    public function testAbsOn2DView(): void
    {
        $a = NDArray::array([[-1, 2], [-3, 4], [-5, 6]], DType::Float64);
        $view = $a->slice(['0:2', ':']); // [[-1, 2], [-3, 4]]

        $result = $view->abs();

        $this->assertSame([2, 2], $result->shape());
        $this->assertEqualsWithDelta([[1, 2], [3, 4]], $result->toArray(), 0.0001);
    }

    public function testSignumOnView(): void
    {
        $a = NDArray::array([-5, 10, -15, 20], DType::Float64);
        $view = $a->slice(['1:3']); // [10, -15]

        $result = $view->signum();

        $this->assertSame([2], $result->shape());
        $this->assertEqualsWithDelta([1, -1], $result->toArray(), 0.0001);
    }

    public function testPow2OnView(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5], DType::Float64);
        $view = $a->slice(['1:4']); // [2, 3, 4]

        $result = $view->pow2();

        $this->assertSame([3], $result->shape());
        $this->assertEqualsWithDelta([4, 9, 16], $result->toArray(), 0.0001);
    }

    public function testPow2On2DView(): void
    {
        $a = NDArray::array([[1, 2], [3, 4], [5, 6]], DType::Float64);
        $view = $a->slice(['1:3', '0:1']); // [[3], [5]]

        $result = $view->pow2();

        $this->assertSame([2, 1], $result->shape());
        $this->assertEqualsWithDelta([[9], [25]], $result->toArray(), 0.0001);
    }

    public function testAbsOnIntegerView(): void
    {
        $a = NDArray::array([[-1, 2], [-3, 4], [-5, 6]], DType::Int32);
        $view = $a->slice(['0:2', ':']); // [[-1, 2], [-3, 4]]

        $result = $view->abs();

        $this->assertSame(DType::Int32, $result->dtype());
        $this->assertEquals([[1, 2], [3, 4]], $result->toArray());
    }

    // ========================================================================
    // Float-Only Math Operations on Views
    // ========================================================================

    public function testSqrtOnView(): void
    {
        $a = NDArray::array([1, 4, 9, 16, 25], DType::Float64);
        $view = $a->slice(['1:4']); // [4, 9, 16]

        $result = $view->sqrt();

        $this->assertSame([3], $result->shape());
        $this->assertEqualsWithDelta([2, 3, 4], $result->toArray(), 0.0001);
    }

    public function testExpOnView(): void
    {
        $a = NDArray::array([0, 1, 2, 3], DType::Float64);
        $view = $a->slice(['0:2']); // [0, 1]

        $result = $view->exp();

        $this->assertSame([2], $result->shape());
        $this->assertEqualsWithDelta([1, \M_E], $result->toArray(), 0.0001);
    }

    public function testLogOnView(): void
    {
        $a = NDArray::array([1, \M_E, \M_E ** 2], DType::Float64);
        $view = $a->slice(['1:3']); // [M_E, M_E^2]

        $result = $view->log();

        $this->assertSame([2], $result->shape());
        $this->assertEqualsWithDelta([1, 2], $result->toArray(), 0.0001);
    }

    public function testSinOnView(): void
    {
        $a = NDArray::array([0, \M_PI / 2, \M_PI], DType::Float64);
        $view = $a->slice(['0:2']); // [0, M_PI/2]

        $result = $view->sin();

        $this->assertSame([2], $result->shape());
        $this->assertEqualsWithDelta([0, 1], $result->toArray(), 0.0001);
    }

    public function testCosOnView(): void
    {
        $a = NDArray::array([0, \M_PI / 2, \M_PI], DType::Float64);
        $view = $a->slice(['1:3']); // [M_PI/2, M_PI]

        $result = $view->cos();

        $this->assertSame([2], $result->shape());
        $this->assertEqualsWithDelta([0, -1], $result->toArray(), 0.0001);
    }

    public function testTanOnView(): void
    {
        $a = NDArray::array([0, \M_PI / 4, \M_PI / 2], DType::Float64);
        $view = $a->slice(['0:2']); // [0, M_PI/4]

        $result = $view->tan();

        $this->assertSame([2], $result->shape());
        $this->assertEqualsWithDelta([0, 1], $result->toArray(), 0.0001);
    }

    public function testSinhOnView(): void
    {
        $a = NDArray::array([0, 1, 2], DType::Float64);
        $view = $a->slice(['1:3']); // [1, 2]

        $result = $view->sinh();

        $this->assertSame([2], $result->shape());
        $this->assertEqualsWithDelta([sinh(1), sinh(2)], $result->toArray(), 0.0001);
    }

    public function testCoshOnView(): void
    {
        $a = NDArray::array([0, 1, 2], DType::Float64);
        $view = $a->slice(['1:3']); // [1, 2]

        $result = $view->cosh();

        $this->assertSame([2], $result->shape());
        $this->assertEqualsWithDelta([cosh(1), cosh(2)], $result->toArray(), 0.0001);
    }

    public function testTanhOnView(): void
    {
        $a = NDArray::array([0, 1, 2], DType::Float64);
        $view = $a->slice(['1:3']); // [1, 2]

        $result = $view->tanh();

        $this->assertSame([2], $result->shape());
        $this->assertEqualsWithDelta([tanh(1), tanh(2)], $result->toArray(), 0.0001);
    }

    public function testSigmoidOnView(): void
    {
        $a = NDArray::array([0, 1, 2], DType::Float64);
        $view = $a->slice(['1:3']); // [1, 2]

        $result = $view->sigmoid();

        $this->assertSame([2], $result->shape());
        $this->assertEqualsWithDelta([1 / (1 + exp(-1)), 1 / (1 + exp(-2))], $result->toArray(), 0.0001);
    }

    public function testSoftmaxOnView(): void
    {
        $a = NDArray::array([1.0, 2.0, 3.0, 4.0], DType::Float64);
        $view = $a->slice(['1:4']); // [2, 3, 4]

        $result = $view->softmax();

        $this->assertSame([3], $result->shape());
        $expSum = exp(2) + exp(3) + exp(4);
        $expected = [exp(2) / $expSum, exp(3) / $expSum, exp(4) / $expSum];
        $this->assertEqualsWithDelta($expected, $result->toArray(), 0.0001);
        $this->assertEqualsWithDelta(1.0, array_sum($result->toArray()), 0.0001);
    }

    public function testSoftmaxOn2DView(): void
    {
        $a = NDArray::array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], DType::Float64);
        $view = $a->slice(['1:3', ':']); // [[3, 4], [5, 6]]

        $result = $view->softmax(axis: -1);

        $this->assertSame([2, 2], $result->shape());
        foreach ($result->toArray() as $row) {
            $this->assertEqualsWithDelta(1.0, array_sum($row), 0.0001);
        }
    }

    public function testAsinOnView(): void
    {
        $a = NDArray::array([0, 0.5, 1], DType::Float64);
        $view = $a->slice(['0:2']); // [0, 0.5]

        $result = $view->asin();

        $this->assertSame([2], $result->shape());
        $this->assertEqualsWithDelta([0, asin(0.5)], $result->toArray(), 0.0001);
    }

    public function testAcosOnView(): void
    {
        $a = NDArray::array([1, 0.5, 0], DType::Float64);
        $view = $a->slice(['1:3']); // [0.5, 0]

        $result = $view->acos();

        $this->assertSame([2], $result->shape());
        $this->assertEqualsWithDelta([acos(0.5), \M_PI / 2], $result->toArray(), 0.0001);
    }

    public function testAtanOnView(): void
    {
        $a = NDArray::array([0, 1, sqrt(3)], DType::Float64);
        $view = $a->slice(['1:3']); // [1, sqrt(3)]

        $result = $view->atan();

        $this->assertSame([2], $result->shape());
        $this->assertEqualsWithDelta([\M_PI / 4, \M_PI / 3], $result->toArray(), 0.0001);
    }

    public function testCbrtOnView(): void
    {
        $a = NDArray::array([1, 8, 27, 64], DType::Float64);
        $view = $a->slice(['1:3']); // [8, 27]

        $result = $view->cbrt();

        $this->assertSame([2], $result->shape());
        $this->assertEqualsWithDelta([2, 3], $result->toArray(), 0.0001);
    }

    public function testCeilOnView(): void
    {
        $a = NDArray::array([1.1, 2.9, -1.1, -2.9], DType::Float64);
        $view = $a->slice(['1:3']); // [2.9, -1.1]

        $result = $view->ceil();

        $this->assertSame([2], $result->shape());
        $this->assertEqualsWithDelta([3, -1], $result->toArray(), 0.0001);
    }

    public function testExp2OnView(): void
    {
        $a = NDArray::array([0, 1, 2, 3, 4], DType::Float64);
        $view = $a->slice(['1:4']); // [1, 2, 3]

        $result = $view->exp2();

        $this->assertSame([3], $result->shape());
        $this->assertEqualsWithDelta([2, 4, 8], $result->toArray(), 0.0001);
    }

    public function testFloorOnView(): void
    {
        $a = NDArray::array([1.9, 2.1, -1.1, -2.9], DType::Float64);
        $view = $a->slice(['0:2']); // [1.9, 2.1]

        $result = $view->floor();

        $this->assertSame([2], $result->shape());
        $this->assertEqualsWithDelta([1, 2], $result->toArray(), 0.0001);
    }

    public function testLog2OnView(): void
    {
        $a = NDArray::array([1, 2, 4, 8, 16], DType::Float64);
        $view = $a->slice(['1:4']); // [2, 4, 8]

        $result = $view->log2();

        $this->assertSame([3], $result->shape());
        $this->assertEqualsWithDelta([1, 2, 3], $result->toArray(), 0.0001);
    }

    public function testLog10OnView(): void
    {
        $a = NDArray::array([1, 10, 100, 1000], DType::Float64);
        $view = $a->slice(['1:3']); // [10, 100]

        $result = $view->log10();

        $this->assertSame([2], $result->shape());
        $this->assertEqualsWithDelta([1, 2], $result->toArray(), 0.0001);
    }

    public function testRoundOnView(): void
    {
        $a = NDArray::array([1.4, 1.6, -1.4, -1.6], DType::Float64);
        $view = $a->slice(['0:2']); // [1.4, 1.6]

        $result = $view->round();

        $this->assertSame([2], $result->shape());
        $this->assertEqualsWithDelta([1, 2], $result->toArray(), 0.0001);
    }

    public function testRecipOnView(): void
    {
        $a = NDArray::array([1, 2, 0.5, 4], DType::Float64);
        $view = $a->slice(['1:3']); // [2, 0.5]

        $result = $view->recip();

        $this->assertSame([2], $result->shape());
        $this->assertEqualsWithDelta([0.5, 2], $result->toArray(), 0.0001);
    }

    public function testHypotOnViews(): void
    {
        $a = NDArray::array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0], DType::Float64);
        $view_a = $a->slice(['0:3']); // [3, 4, 5]

        $result = $view_a->hypot(4.0);

        $this->assertSame([3], $result->shape());
        $this->assertEqualsWithDelta([5.0, 5.657, 6.403], $result->toArray(), 0.001);
    }

    // ========================================================================
    // 2D View Tests
    // ========================================================================

    public function testSqrtOn2DView(): void
    {
        $a = NDArray::array([
            [1, 4, 9],
            [16, 25, 36],
            [49, 64, 81],
        ], DType::Float64);
        $view = $a->slice(['0:2', '1:3']); // [[4, 9], [25, 36]]

        $result = $view->sqrt();

        $this->assertSame([2, 2], $result->shape());
        $this->assertEqualsWithDelta([[2, 3], [5, 6]], $result->toArray(), 0.0001);
    }

    public function testExpOn2DView(): void
    {
        $a = NDArray::array([
            [0, 1],
            [2, 3],
            [4, 5],
        ], DType::Float64);
        $view = $a->slice(['1:3', ':']); // [[2, 3], [4, 5]]

        $result = $view->exp();

        $this->assertSame([2, 2], $result->shape());
        $this->assertEqualsWithDelta([[exp(2), exp(3)], [exp(4), exp(5)]], $result->toArray(), 0.0001);
    }

    // ========================================================================
    // Shape Preservation Tests
    // ========================================================================

    public function testMathOperationsPreserveShapeOnView(): void
    {
        $a = NDArray::array([
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ], DType::Float64);
        $view = $a->slice(['0:1']); // [[[1, 2, 3], [4, 5, 6]]]

        $result = $view->sqrt();

        $this->assertSame([1, 2, 3], $result->shape());
    }

    public function testMathOperationsOnFloat32View(): void
    {
        $a = NDArray::array([1, 4, 9, 16], DType::Float32);
        $view = $a->slice(['1:3']); // [4, 9]

        $result = $view->sqrt();

        $this->assertSame(DType::Float32, $result->dtype());
        $this->assertEqualsWithDelta([2, 3], $result->toArray(), 0.0001);
    }

    public function testChainedMathOperationsOnView(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5], DType::Float64);
        $view = $a->slice(['1:4']); // [2, 3, 4]

        $result = $view->pow2()->sqrt();

        $this->assertSame([3], $result->shape());
        $this->assertEqualsWithDelta([2, 3, 4], $result->toArray(), 0.0001);
    }

    // ========================================================================
    // Edge Cases
    // ========================================================================

    public function testSingleElementViewMath(): void
    {
        $a = NDArray::array([[1, 4], [9, 16]], DType::Float64);
        // String indices '1' and '1' are converted to integers, extracting single element
        $single = $a->slice(['1', '1']);

        $result = $single->sqrt();

        // Scalar result from single element extraction
        $this->assertSame([], $result->shape());
        $this->assertEqualsWithDelta(4, $result->toArray(), 0.0001);
    }

    public function testEmptyViewMath(): void
    {
        $a = NDArray::array([1, 4, 9], DType::Float64);
        $empty = $a->slice(['1:1']); // Empty

        $result = $empty->sqrt();

        $this->assertSame([0], $result->shape());
    }

    public function testStridedViewMath(): void
    {
        $a = NDArray::array([1, 4, 9, 16, 25, 36, 49, 64], DType::Float64);
        $strided = $a->slice(['::2']); // [1, 9, 25, 49]

        $result = $strided->sqrt();

        $this->assertSame([4], $result->shape());
        $this->assertEqualsWithDelta([1, 3, 5, 7], $result->toArray(), 0.0001);
    }
}
