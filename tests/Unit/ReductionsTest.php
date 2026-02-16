<?php

declare(strict_types=1);

namespace NDArray\Tests\Unit;

use NDArray\DType;
use NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Tests for reduction and aggregation operations
 */
final class ReductionsTest extends TestCase
{
    // =========================================================================
    // Scalar Reduction Tests
    // =========================================================================

    public function testSumScalar(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->sum();
        $this->assertEquals(21.0, $result);
    }

    public function testSumScalar1D(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5], DType::Float64);
        $result = $a->sum();
        $this->assertEquals(15.0, $result);
    }

    public function testMeanScalar(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->mean();
        $this->assertEqualsWithDelta(3.5, $result, 0.0001);
    }

    public function testMinScalar(): void
    {
        $a = NDArray::array([[5, 2, 8], [1, 9, 3]], DType::Float64);
        $result = $a->min();
        $this->assertEquals(1.0, $result);
    }

    public function testMaxScalar(): void
    {
        $a = NDArray::array([[5, 2, 8], [1, 9, 3]], DType::Float64);
        $result = $a->max();
        $this->assertEquals(9.0, $result);
    }

    public function testArgminScalar(): void
    {
        $a = NDArray::array([[5, 2, 8], [1, 9, 3]], DType::Float64);
        $result = $a->argmin();
        $this->assertEquals(3, $result); // Index of 1 in flattened array
    }

    public function testArgmaxScalar(): void
    {
        $a = NDArray::array([[5, 2, 8], [1, 9, 3]], DType::Float64);
        $result = $a->argmax();
        $this->assertEquals(4, $result); // Index of 9 in flattened array
    }

    public function testProductScalar(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->product();
        $this->assertEquals(720.0, $result); // 1*2*3*4*5*6 = 720
    }

    public function testVarScalar(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5], DType::Float64);
        $result = $a->var();
        $this->assertEqualsWithDelta(2.0, $result, 0.0001); // Population variance
    }

    public function testVarSample(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5], DType::Float64);
        $result = $a->var(ddof: 1);
        $this->assertEqualsWithDelta(2.5, $result, 0.0001); // Sample variance
    }

    public function testStdScalar(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5], DType::Float64);
        $result = $a->std();
        $this->assertEqualsWithDelta(sqrt(2.0), $result, 0.0001);
    }

    public function testStdSample(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5], DType::Float64);
        $result = $a->std(ddof: 1);
        $this->assertEqualsWithDelta(sqrt(2.5), $result, 0.0001);
    }

    // =========================================================================
    // Axis Reduction Tests
    // =========================================================================

    public function testSumAxis0(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->sum(axis: 0);
        $this->assertEqualsWithDelta([5, 7, 9], $result->toArray(), 0.0001);
        $this->assertSame([3], $result->shape());
    }

    public function testSumAxis1(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->sum(axis: 1);
        $this->assertEqualsWithDelta([6, 15], $result->toArray(), 0.0001);
        $this->assertSame([2], $result->shape());
    }

    public function testMeanAxis0(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->mean(axis: 0);
        $this->assertEqualsWithDelta([2.5, 3.5, 4.5], $result->toArray(), 0.0001);
    }

    public function testMeanAxis1(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->mean(axis: 1);
        $this->assertEqualsWithDelta([2.0, 5.0], $result->toArray(), 0.0001);
    }

    public function testMinAxis0(): void
    {
        $a = NDArray::array([[5, 2, 8], [1, 9, 3]], DType::Float64);
        $result = $a->min(axis: 0);
        $this->assertEqualsWithDelta([1, 2, 3], $result->toArray(), 0.0001);
    }

    public function testMinAxis1(): void
    {
        $a = NDArray::array([[5, 2, 8], [1, 9, 3]], DType::Float64);
        $result = $a->min(axis: 1);
        $this->assertEqualsWithDelta([2, 1], $result->toArray(), 0.0001);
    }

    public function testMaxAxis0(): void
    {
        $a = NDArray::array([[5, 2, 8], [1, 9, 3]], DType::Float64);
        $result = $a->max(axis: 0);
        $this->assertEqualsWithDelta([5, 9, 8], $result->toArray(), 0.0001);
    }

    public function testMaxAxis1(): void
    {
        $a = NDArray::array([[5, 2, 8], [1, 9, 3]], DType::Float64);
        $result = $a->max(axis: 1);
        $this->assertEqualsWithDelta([8, 9], $result->toArray(), 0.0001);
    }

    public function testArgminAxis0(): void
    {
        $a = NDArray::array([[5, 2, 8], [1, 9, 3]], DType::Float64);
        $result = $a->argmin(axis: 0);
        $this->assertEquals([1, 0, 1], $result->toArray());
        $this->assertSame(DType::Int64, $result->dtype());
    }

    public function testArgminAxis1(): void
    {
        $a = NDArray::array([[5, 2, 8], [1, 9, 3]], DType::Float64);
        $result = $a->argmin(axis: 1);
        $this->assertEquals([1, 0], $result->toArray());
    }

    public function testArgmaxAxis0(): void
    {
        $a = NDArray::array([[5, 2, 8], [1, 9, 3]], DType::Float64);
        $result = $a->argmax(axis: 0);
        $this->assertEquals([0, 1, 0], $result->toArray());
        $this->assertSame(DType::Int64, $result->dtype());
    }

    public function testArgmaxAxis1(): void
    {
        $a = NDArray::array([[5, 2, 8], [1, 9, 3]], DType::Float64);
        $result = $a->argmax(axis: 1);
        $this->assertEquals([2, 1], $result->toArray());
    }

    public function testProductAxis0(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->product(axis: 0);
        $this->assertEqualsWithDelta([4, 10, 18], $result->toArray(), 0.0001);
    }

    public function testProductAxis1(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->product(axis: 1);
        $this->assertEqualsWithDelta([6, 120], $result->toArray(), 0.0001);
    }

    public function testCumsumScalar(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5], DType::Float64);
        $result = $a->cumsum();
        $this->assertSame([5], $result->shape());
        $this->assertEqualsWithDelta([1, 3, 6, 10, 15], $result->toArray(), 0.0001);
    }

    public function testCumsumScalar2D(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->cumsum();
        $this->assertSame([6], $result->shape());
        $this->assertEqualsWithDelta([1, 3, 6, 10, 15, 21], $result->toArray(), 0.0001);
    }

    public function testCumsumAxis0(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->cumsum(axis: 0);
        $this->assertSame([2, 3], $result->shape());
        $this->assertEqualsWithDelta([[1, 2, 3], [5, 7, 9]], $result->toArray(), 0.0001);
    }

    public function testCumsumAxis1(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->cumsum(axis: 1);
        $this->assertSame([2, 3], $result->shape());
        $this->assertEqualsWithDelta([[1, 3, 6], [4, 9, 15]], $result->toArray(), 0.0001);
    }

    public function testCumprodScalar(): void
    {
        $a = NDArray::array([1, 2, 3, 4], DType::Float64);
        $result = $a->cumprod();
        $this->assertSame([4], $result->shape());
        $this->assertEqualsWithDelta([1, 2, 6, 24], $result->toArray(), 0.0001);
    }

    public function testCumprodScalar2D(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);
        $result = $a->cumprod();
        $this->assertSame([4], $result->shape());
        $this->assertEqualsWithDelta([1, 2, 6, 24], $result->toArray(), 0.0001);
    }

    public function testCumprodAxis0(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->cumprod(axis: 0);
        $this->assertSame([2, 3], $result->shape());
        $this->assertEqualsWithDelta([[1, 2, 3], [4, 10, 18]], $result->toArray(), 0.0001);
    }

    public function testCumprodAxis1(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->cumprod(axis: 1);
        $this->assertSame([2, 3], $result->shape());
        $this->assertEqualsWithDelta([[1, 2, 6], [4, 20, 120]], $result->toArray(), 0.0001);
    }

    public function testVarAxis0(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->var(axis: 0);
        $this->assertEqualsWithDelta([2.25, 2.25, 2.25], $result->toArray(), 0.0001);
        $this->assertSame(DType::Float64, $result->dtype());
    }

    public function testStdAxis1(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->std(axis: 1);
        $this->assertEqualsWithDelta([sqrt(2/3), sqrt(2/3)], $result->toArray(), 0.0001);
    }

    // =========================================================================
    // Keepdims Tests
    // =========================================================================

    public function testSumAxis0Keepdims(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->sum(axis: 0, keepdims: true);
        $this->assertSame([1, 3], $result->shape());
        $this->assertEqualsWithDelta([[5, 7, 9]], $result->toArray(), 0.0001);
    }

    public function testSumAxis1Keepdims(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->sum(axis: 1, keepdims: true);
        $this->assertSame([2, 1], $result->shape());
        $this->assertEqualsWithDelta([[6], [15]], $result->toArray(), 0.0001);
    }

    public function testMeanAxisKeepdims(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->mean(axis: 0, keepdims: true);
        $this->assertSame([1, 3], $result->shape());
    }

    public function testMinAxisKeepdims(): void
    {
        $a = NDArray::array([[5, 2, 8], [1, 9, 3]], DType::Float64);
        $result = $a->min(axis: 1, keepdims: true);
        $this->assertSame([2, 1], $result->shape());
    }

    // =========================================================================
    // Negative Axis Tests
    // =========================================================================

    public function testSumNegativeAxis(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->sum(axis: -1);
        $this->assertEqualsWithDelta([6, 15], $result->toArray(), 0.0001);
    }

    public function testMeanNegativeAxis(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->mean(axis: -2);
        $this->assertEqualsWithDelta([2.5, 3.5, 4.5], $result->toArray(), 0.0001);
    }

    // =========================================================================
    // Dtype Tests
    // =========================================================================

    public function testSumPreservesInt64Dtype(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Int64);
        $result = $a->sum(axis: 0);
        $this->assertSame(DType::Int64, $result->dtype());
    }

    public function testMeanReturnsFloat64(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Int64);
        $result = $a->mean(axis: 0);
        $this->assertSame(DType::Float64, $result->dtype());
    }

    public function testMinPreservesDtype(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Int32);
        $result = $a->min(axis: 0);
        $this->assertSame(DType::Int32, $result->dtype());
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    public function testSumSingleElement(): void
    {
        $a = NDArray::array([42], DType::Float64);
        $result = $a->sum();
        $this->assertEquals(42.0, $result);
    }

    public function testMeanSingleElement(): void
    {
        $a = NDArray::array([42], DType::Float64);
        $result = $a->mean();
        $this->assertEquals(42.0, $result);
    }

    public function testArgminOnFloat32(): void
    {
        $a = NDArray::array([3.0, 1.0, 2.0], DType::Float32);
        $result = $a->argmin();
        $this->assertEquals(1, $result);
    }

    public function test3DArrayReductions(): void
    {
        $a = NDArray::array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ], DType::Float64);
        
        // Sum along axis 0
        $result = $a->sum(axis: 0);
        $this->assertSame([2, 2], $result->shape());
        $this->assertEqualsWithDelta([[6, 8], [10, 12]], $result->toArray(), 0.0001);
        
        // Sum along axis 1
        $result = $a->sum(axis: 1);
        $this->assertSame([2, 2], $result->shape());
        $this->assertEqualsWithDelta([[4, 6], [12, 14]], $result->toArray(), 0.0001);
        
        // Sum along axis 2
        $result = $a->sum(axis: 2);
        $this->assertSame([2, 2], $result->shape());
        $this->assertEqualsWithDelta([[3, 7], [11, 15]], $result->toArray(), 0.0001);
    }

}
