<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Unit;

use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\Exceptions\ShapeException;
use PhpMlKit\NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Comprehensive tests for NumPy-compatible broadcasting.
 *
 * Broadcasting allows operations between arrays of different shapes by
 * virtually replicating the smaller array to match the larger one.
 *
 * @internal
 *
 * @coversNothing
 */
class BroadcastingTest extends TestCase
{
    public function testAddScalarTo1DArray(): void
    {
        $arr = NDArray::array([1, 2, 3]);
        $result = $arr->add(10);

        $this->assertEquals([11, 12, 13], $result->toArray());
    }

    public function testMultiplyScalarTo2DArray(): void
    {
        $arr = NDArray::array([[1, 2], [3, 4]]);
        $result = $arr->multiply(2);

        $this->assertEquals([[2, 4], [6, 8]], $result->toArray());
    }

    public function testSubtractScalar(): void
    {
        $arr = NDArray::array([10, 20, 30]);
        $result = $arr->subtract(5);

        $this->assertEquals([5, 15, 25], $result->toArray());
    }

    public function testDivideByScalar(): void
    {
        $arr = NDArray::array([10, 20, 30]);
        $result = $arr->divide(10);

        $this->assertEquals([1, 2, 3], $result->toArray());
    }

    public function testScalarBroadcastWithDifferentDTypes(): void
    {
        $arr = NDArray::array([1, 2, 3], DType::Float64);
        $result = $arr->add(1.5);  // Float scalar with Float64 array

        $this->assertSame(DType::Float64, $result->dtype());
        $this->assertEqualsWithDelta([2.5, 3.5, 4.5], $result->toArray(), 0.001);
    }

    public function testBroadcast1DTo2DRowWise(): void
    {
        // Shape (2, 3) + Shape (3,) = Shape (2, 3)
        $matrix = NDArray::array([[1, 2, 3], [4, 5, 6]]);  // Shape [2, 3]
        $row = NDArray::array([10, 20, 30]);  // Shape [3]

        $result = $matrix->add($row);

        $this->assertEquals([[11, 22, 33], [14, 25, 36]], $result->toArray());
    }

    public function testBroadcast1DTo2DColumnWise(): void
    {
        // Shape (2, 3) + Shape (2, 1) = Shape (2, 3)
        $matrix = NDArray::array([[1, 2, 3], [4, 5, 6]]);  // Shape [2, 3]
        $col = NDArray::array([[10], [20]]);  // Shape [2, 1]

        $result = $matrix->add($col);

        $this->assertEquals([[11, 12, 13], [24, 25, 26]], $result->toArray());
    }

    public function testBroadcast1DTo2DMultiply(): void
    {
        $matrix = NDArray::array([[1, 2, 3], [4, 5, 6]]);
        $factor = NDArray::array([2, 3, 4]);

        $result = $matrix->multiply($factor);

        $this->assertEquals([[2, 6, 12], [8, 15, 24]], $result->toArray());
    }

    public function testBroadcast3x1With1x3(): void
    {
        // Shape (3, 1) + Shape (1, 3) = Shape (3, 3)
        $a = NDArray::array([[1], [2], [3]]);  // Shape [3, 1]
        $b = NDArray::array([[10, 20, 30]]);  // Shape [1, 3]

        $result = $a->add($b);

        $expected = [
            [11, 21, 31],
            [12, 22, 32],
            [13, 23, 33],
        ];
        $this->assertEquals($expected, $result->toArray());
    }

    public function testBroadcast1x3With3x1(): void
    {
        // Shape (1, 3) + Shape (3, 1) = Shape (3, 3)
        $a = NDArray::array([[1, 2, 3]]);  // Shape [1, 3]
        $b = NDArray::array([[10], [20], [30]]);  // Shape [3, 1]

        $result = $a->multiply($b);

        $expected = [
            [10, 20, 30],
            [20, 40, 60],
            [30, 60, 90],
        ];
        $this->assertEquals($expected, $result->toArray());
    }

    public function testBroadcast4x1x3With1x5x3(): void
    {
        // Shape (4, 1, 3) + Shape (1, 5, 3) = Shape (4, 5, 3)
        $a = NDArray::array([
            [[1, 2, 3]],
            [[4, 5, 6]],
            [[7, 8, 9]],
            [[10, 11, 12]],
        ]);  // Shape [4, 1, 3]
        $b = NDArray::array([
            [[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120], [130, 140, 150]],
        ]);  // Shape [1, 5, 3]

        $result = $a->add($b);

        $this->assertEquals([4, 5, 3], $result->shape());
    }

    public function testBroadcast3DArrayWith1D(): void
    {
        // Shape (2, 3, 4) + Shape (4,) = Shape (2, 3, 4)
        $tensor = NDArray::zeros([2, 3, 4]);
        $vec = NDArray::array([1, 2, 3, 4]);

        $result = $tensor->add($vec);

        $this->assertEquals([2, 3, 4], $result->shape());
        $this->assertEquals([1, 2, 3, 4], $result->slice([0, 0])->toArray());
    }

    public function testBroadcast3DArrayWith2D(): void
    {
        // Shape (2, 3, 4) + Shape (3, 4) = Shape (2, 3, 4)
        $tensor = NDArray::ones([2, 3, 4]);
        $matrix = NDArray::array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ]);

        $result = $tensor->add($matrix);

        $this->assertEquals([2, 3, 4], $result->shape());
        $this->assertEquals([2, 3, 4, 5], $result->slice([0, 0])->toArray());
    }

    public function testBroadcastIncompatibleShapesThrows(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]]);  // Shape [2, 3]
        $b = NDArray::array([[1, 2], [3, 4], [5, 6]]);  // Shape [3, 2]

        $this->expectException(ShapeException::class);
        $a->add($b);
    }

    public function testBroadcast3DIncompatibleShapesThrows(): void
    {
        $a = NDArray::zeros([2, 3, 4]);
        $b = NDArray::zeros([2, 4, 3]);

        $this->expectException(ShapeException::class);
        $a->add($b);
    }

    public function testBroadcastMismatched1DThrows(): void
    {
        $matrix = NDArray::array([[1, 2, 3], [4, 5, 6]]);  // Shape [2, 3]
        $vec = NDArray::array([1, 2]);  // Shape [2] - can't broadcast to [2, 3]

        $this->expectException(ShapeException::class);
        $matrix->add($vec);
    }

    public function testBroadcastWithViewRow(): void
    {
        $matrix = NDArray::array([[1, 2, 3], [4, 5, 6]]);
        $row = $matrix->slice(['0:1']);  // Shape [1, 3]
        $scalar = 10;

        $result = $row->add($scalar);

        $this->assertEquals([[11, 12, 13]], $result->toArray());
    }

    public function testBroadcastWithViewColumn(): void
    {
        $matrix = NDArray::array([[1, 2, 3], [4, 5, 6]]);
        $col = $matrix->slice([':', '0:1']);  // Shape [2, 1]
        $row = NDArray::array([[10, 20, 30]]);  // Shape [1, 3]

        $result = $col->add($row);

        $expected = [[11, 21, 31], [14, 24, 34]];
        $this->assertEquals($expected, $result->toArray());
    }

    public function testBroadcastTransposedArray(): void
    {
        $matrix = NDArray::array([[1, 2, 3], [4, 5, 6]]);  // Shape [2, 3]
        $transposed = $matrix->transpose();  // Shape [3, 2]
        $vec = NDArray::array([10, 20]);  // Shape [2]

        $result = $transposed->add($vec);

        $this->assertEquals([3, 2], $result->shape());
    }

    public function testBroadcastInComparison(): void
    {
        $matrix = NDArray::array([[1, 2, 3], [4, 5, 6]]);
        $threshold = NDArray::array([[3], [6]]);  // Shape [2, 1]

        $result = $matrix->gt($threshold);

        $expected = [[false, false, false], [false, false, false]];
        $this->assertEquals($expected, $result->toArray());
    }

    public function testBroadcastInLogical(): void
    {
        $a = NDArray::array([[true, false], [true, true]]);
        $b = NDArray::array([[true, true]]);  // Shape [1, 2]

        $result = $a->and($b);

        $expected = [[true, false], [true, true]];
        $this->assertEquals($expected, $result->toArray());
    }

    public function testBroadcastInRemainder(): void
    {
        $matrix = NDArray::array([[10, 20, 30], [40, 50, 60]]);
        $divisors = NDArray::array([[3], [7]]);  // Shape [2, 1]

        $result = $matrix->rem($divisors);

        $expected = [[1, 2, 0], [5, 1, 4]];
        $this->assertEquals($expected, $result->toArray());
    }

    public function testBroadcastEmptyArrayThrows(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5]);
        $b = $a->slice(['2:2']); // Empty slice with shape [0]

        $this->expectException(ShapeException::class);
        $a->add($b);
    }

    public function testBroadcastWithDifferentDTypes(): void
    {
        $intArr = NDArray::array([1, 2, 3], DType::Int32);
        $floatArr = NDArray::array([1.5, 2.5, 3.5], DType::Float64);

        $result = $intArr->add($floatArr);

        $this->assertEquals(DType::Float64, $result->dtype());
        $this->assertEqualsWithDelta([2.5, 4.5, 6.5], $result->toArray(), 0.001);
    }

    public function testBroadcastingRule1SmallerDimensionPrepended(): void
    {
        // When dimensions differ, 1s are prepended to the smaller shape
        // Shape (3,) and Shape (2, 3) -> treated as (1, 3) and (2, 3)
        $a = NDArray::array([1, 2, 3]);  // Shape [3]
        $b = NDArray::array([[10, 20, 30], [40, 50, 60]]);  // Shape [2, 3]

        $result = $a->add($b);

        $expected = [[11, 22, 33], [41, 52, 63]];
        $this->assertEquals($expected, $result->toArray());
    }

    public function testBroadcastingRule2DimensionOfSize1CanStretch(): void
    {
        // Dimensions of size 1 can be stretched to match the other array
        // Shape (3, 1) and Shape (1, 3) -> both stretch to (3, 3)
        $a = NDArray::array([[1], [2], [3]]);  // Shape [3, 1]
        $b = NDArray::array([[10, 20, 30]]);  // Shape [1, 3]

        $result = $a->add($b);

        $expected = [
            [11, 21, 31],
            [12, 22, 32],
            [13, 23, 33],
        ];
        $this->assertEquals($expected, $result->toArray());
    }

    public function testBroadcastingRule3DimensionsMustMatchOrBe1(): void
    {
        // If dimensions don't match and neither is 1, broadcasting fails
        $a = NDArray::array([[1, 2, 3]]);  // Shape [1, 3]
        $b = NDArray::array([[1, 2]]);  // Shape [1, 2]

        $this->expectException(ShapeException::class);
        $a->add($b);
    }
}
