<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Unit;

use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\Exceptions\IndexException;
use PhpMlKit\NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Edge case tests for NDArray operations.
 *
 * Tests boundary conditions, special values, and unusual scenarios.
 *
 * @internal
 *
 * @coversNothing
 */
class EdgeCasesTest extends TestCase
{
    public function testNaNPropagationInArithmetic(): void
    {
        $a = NDArray::array([1.0, \NAN, 3.0]);
        $b = NDArray::array([2.0, 2.0, 2.0]);

        $result = $a->add($b);

        $this->assertEqualsWithDelta(3.0, $result->getAt(0), 0.001);
        $this->assertNan($result->getAt(1));
        $this->assertEqualsWithDelta(5.0, $result->getAt(2), 0.001);
    }

    public function testNaNSorting(): void
    {
        $a = NDArray::array([3.0, \NAN, 1.0, 2.0]);
        $result = $a->sort();

        // NaN should be placed at the end
        $this->assertEqualsWithDelta(1.0, $result->getAt(0), 0.001);
        $this->assertEqualsWithDelta(2.0, $result->getAt(1), 0.001);
        $this->assertEqualsWithDelta(3.0, $result->getAt(2), 0.001);
        $this->assertNan($result->getAt(3));
    }

    public function testNaNInReductions(): void
    {
        $a = NDArray::array([1.0, \NAN, 3.0]);

        // sum with NaN should be NaN
        $sum = $a->sum();
        $this->assertNan($sum);

        // mean with NaN should be NaN
        $mean = $a->mean();
        $this->assertNan($mean);
    }

    public function testInfinityArithmetic(): void
    {
        $a = NDArray::array([1.0, \INF, 3.0]);
        $b = NDArray::array([2.0, 2.0, 2.0]);

        $result = $a->add($b);

        $this->assertEqualsWithDelta(3.0, $result->getAt(0), 0.001);
        $this->assertInfinite($result->getAt(1));
        $this->assertEqualsWithDelta(5.0, $result->getAt(2), 0.001);
    }

    public function testInfinityMultiplyByZero(): void
    {
        $a = NDArray::array([\INF]);
        $result = $a->multiply(0);

        // INF * 0 should be NaN
        $this->assertNan($result->getAt(0));
    }

    public function testFloatSubnormalNumbers(): void
    {
        $minPositive = 5e-324; // Smallest positive subnormal float64
        $a = NDArray::array([$minPositive]);
        $result = $a->multiply(2);

        $this->assertGreaterThan(0, $result->getAt(0));
    }

    public function testFloatMaxValue(): void
    {
        $maxFloat = 1.7976931348623157e+308;
        $a = NDArray::array([$maxFloat]);
        $result = $a->multiply(2);

        // Should become INF
        $this->assertInfinite($result->getAt(0));
    }

    public function testPrecisionLossWithLargeNumbers(): void
    {
        $a = NDArray::array([1e16, 1.0]);
        $result = $a->sum();

        // 1e16 + 1 may lose precision
        $this->assertEqualsWithDelta(1e16, $result, 1);
    }

    public function testSliceAtExactBoundaries(): void
    {
        $a = NDArray::array([0, 1, 2, 3, 4, 5]);

        // Slice at exact start
        $start = $a->slice(['0:1']);
        $this->assertEquals([0], $start->toArray());

        // Slice at exact end
        $end = $a->slice(['5:6']);
        $this->assertEquals([5], $end->toArray());
    }

    public function testNegativeIndicesAtBoundaries(): void
    {
        $a = NDArray::array([0, 1, 2, 3, 4, 5]);

        // -1 should be last element
        $last = $a->slice(['-1:']);
        $this->assertEquals([5], $last->toArray());

        // -6 should be first element
        $first = $a->slice(['-6:-5']);
        $this->assertEquals([0], $first->toArray());
    }

    public function testZeroStepSlice(): void
    {
        $a = NDArray::array([0, 1, 2, 3, 4, 5]);
        $result = $a->slice(['0:6:1']);

        $this->assertEquals([0, 1, 2, 3, 4, 5], $result->toArray());
    }

    public function testNegativeStepSliceNotYetSupported(): void
    {
        // NOTE: Negative slice steps are not yet supported
        // This test documents the expected behavior once implemented
        $this->expectException(IndexException::class);
        $this->expectExceptionMessage('Negative slice steps are not yet supported');

        $a = NDArray::array([0, 1, 2, 3, 4, 5]);
        $a->slice(['5:0:-1']);
    }

    public function testOperationsOnEmptySlice(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5]);
        $empty = $a->slice(['2:2']);

        $this->assertSame([0], $empty->shape());
        $this->assertEquals(0, $empty->size());

        // Sum of empty should be 0
        $this->assertEquals(0, $empty->sum());
    }

    public function testAdditionIdentity(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5]);
        $zero = NDArray::zeros([5]);

        $result = $a->add($zero);
        $this->assertEquals([1, 2, 3, 4, 5], $result->toArray());
    }

    public function testMultiplicationIdentity(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5]);
        $one = NDArray::ones([5]);

        $result = $a->multiply($one);
        $this->assertEquals([1, 2, 3, 4, 5], $result->toArray());
    }

    public function testDivisionByZero(): void
    {
        $a = NDArray::array([1.0, 2.0, 3.0]);
        $result = $a->divide(0);

        // Should produce INF
        $this->assertInfinite($result->getAt(0));
    }

    public function testZeroDividedByZero(): void
    {
        $a = NDArray::array([0.0]);
        $result = $a->divide(0);

        $this->assertNan($result->getAt(0));
    }

    public function testRemainderByZero(): void
    {
        $a = NDArray::array([5.0, 10.0]);
        $result = $a->rem(0);

        // Should produce NaN
        $this->assertTrue(is_nan($result->getAt(0)));
    }

    // ============================================================================
    // Array with All Same Values
    // ============================================================================

    public function testAllSameValuesStd(): void
    {
        $a = NDArray::array([5, 5, 5, 5, 5]);

        $this->assertEquals(0, $a->std());
        $this->assertEquals(0, $a->var());
    }

    public function testAllSameValuesSort(): void
    {
        $a = NDArray::array([5, 5, 5, 5, 5]);
        $result = $a->sort();

        $this->assertEquals([5, 5, 5, 5, 5], $result->toArray());
    }

    // ============================================================================
    // Maximum Dimension Tests
    // ============================================================================

    public function testLargeArrayCreation(): void
    {
        // Test creating a reasonably large array
        $large = NDArray::arange(0, 10000);
        $this->assertEquals(10000, $large->size());
        $this->assertEquals([10000], $large->shape());
    }

    public function testLarge2DArray(): void
    {
        $large = NDArray::zeros([1000, 100]);
        $this->assertEquals(100000, $large->size());
        $this->assertEquals([1000, 100], $large->shape());
    }

    // ============================================================================
    // View Chain Tests
    // ============================================================================

    public function testNestedViews(): void
    {
        $a = NDArray::array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]);

        // Create nested views
        $view1 = $a->slice(['1:3']); // Rows 1-2
        $view2 = $view1->slice([':', '1:3']); // Columns 1-2 of view1

        $this->assertEquals([[6, 7], [10, 11]], $view2->toArray());
    }

    public function testViewOfTransposedView(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        $transposed = $a->transpose();
        $view = $transposed->slice(['0:2', '0:2']);

        $this->assertEquals([[1, 4], [2, 5]], $view->toArray());
    }

    // ============================================================================
    // Memory and Resource Tests
    // ============================================================================

    public function testMultipleReferencesToSameView(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5]);
        $view = $a->slice(['1:4']);

        // Modify through one reference
        $view->setAt(0, 999);

        // Check original is modified
        $this->assertEquals(999, $a->getAt(1));
    }

    public function testViewAfterOriginalModification(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5]);
        $view = $a->slice(['1:4']); // [2, 3, 4]

        // Modify original
        $a->setAt(2, 999);

        // View should see the change
        $this->assertEquals(999, $view->getAt(1));
    }

    // ============================================================================
    // Type Conversion Edge Cases
    // ============================================================================

    public function testFloatToIntTruncation(): void
    {
        $a = NDArray::array([1.9, 2.5, 3.1], DType::Float64);
        $result = $a->astype(DType::Int32);

        // Should truncate toward zero
        $this->assertEquals([1, 2, 3], $result->toArray());
    }

    public function testNegativeFloatToInt(): void
    {
        $a = NDArray::array([-1.9, -2.5, -3.1], DType::Float64);
        $result = $a->astype(DType::Int32);

        // Should truncate toward zero
        $this->assertEquals([-1, -2, -3], $result->toArray());
    }

    public function testLargeFloatToInt(): void
    {
        $a = NDArray::array([1e20], DType::Float64);
        $result = $a->astype(DType::Int64);

        // Large values may saturate or wrap
        $this->assertIsInt($result->getAt(0));
    }

    // ============================================================================
    // Comparison Edge Cases
    // ============================================================================

    public function testComparisonWithNaN(): void
    {
        $a = NDArray::array([1.0, \NAN, 3.0]);
        $b = NDArray::array([1.0, 2.0, 3.0]);

        $eq = $a->eq($b);

        // NaN == anything should be false
        $this->assertTrue($eq->getAt(0));
        $this->assertFalse($eq->getAt(1));
        $this->assertTrue($eq->getAt(2));
    }

    public function testComparisonWithInfinity(): void
    {
        $a = NDArray::array([1.0, \INF, -\INF]);
        $b = NDArray::array([1.0, \INF, -\INF]);

        $eq = $a->eq($b);

        $this->assertTrue($eq->getAt(0));
        $this->assertTrue($eq->getAt(1));
        $this->assertTrue($eq->getAt(2));
    }

    // ============================================================================
    // Logical Operation Edge Cases
    // ============================================================================

    public function testLogicalOperationsWithAllTrue(): void
    {
        $a = NDArray::array([true, true, true], DType::Bool);
        $b = NDArray::array([true, true, true], DType::Bool);

        $and = $a->and($b);
        $or = $a->or($b);

        $this->assertEquals([true, true, true], $and->toArray());
        $this->assertEquals([true, true, true], $or->toArray());
    }

    public function testLogicalOperationsWithAllFalse(): void
    {
        $a = NDArray::array([false, false, false], DType::Bool);
        $b = NDArray::array([false, false, false], DType::Bool);

        $and = $a->and($b);
        $or = $a->or($b);

        $this->assertEquals([false, false, false], $and->toArray());
        $this->assertEquals([false, false, false], $or->toArray());
    }

    // ============================================================================
    // Shape Manipulation Edge Cases
    // ============================================================================

    public function testReshapeToSameShape(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]]);
        $result = $a->reshape([2, 3]);

        $this->assertEquals([[1, 2, 3], [4, 5, 6]], $result->toArray());
    }

    public function testFlattenAlreadyFlat(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5]);
        $result = $a->flatten();

        $this->assertEquals([1, 2, 3, 4, 5], $result->toArray());
    }

    public function testTranspose1DArray(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5]);
        $result = $a->transpose();

        // 1D transpose should return same shape
        $this->assertEquals([5], $result->shape());
    }
}
