<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Unit;

use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\FlatIterator;
use PhpMlKit\NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Tests for NDArray iterator functionality.
 *
 * @internal
 *
 * @coversNothing
 */
class IteratorTest extends TestCase
{
    // ============================================================================
    // 1D Array Iteration Tests
    // ============================================================================

    public function test1DArrayIteration(): void
    {
        $arr = NDArray::array([1, 2, 3, 4, 5]);
        $values = [];

        foreach ($arr as $value) {
            $values[] = $value;
        }

        $this->assertEquals([1, 2, 3, 4, 5], $values);
    }

    public function test1DArrayIterationKeys(): void
    {
        $arr = NDArray::array([10, 20, 30]);
        $keys = [];

        foreach ($arr as $key => $value) {
            $keys[] = $key;
        }

        $this->assertEquals([0, 1, 2], $keys);
    }

    public function test1DArrayWithDifferentDtypes(): void
    {
        // Int32
        $arr = NDArray::array([1, 2, 3], DType::Int32);
        $values = iterator_to_array($arr);
        $this->assertEquals([1, 2, 3], $values);

        // Float64
        $arr = NDArray::array([1.1, 2.2, 3.3], DType::Float64);
        $values = iterator_to_array($arr);
        $this->assertEqualsWithDelta([1.1, 2.2, 3.3], $values, 0.001);
    }

    // ============================================================================
    // 2D Array Iteration Tests
    // ============================================================================

    public function test2DArrayIterationYieldsViews(): void
    {
        $matrix = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]);

        $rows = [];
        foreach ($matrix as $row) {
            $this->assertInstanceOf(NDArray::class, $row);
            $this->assertEquals([3], $row->shape());
            $rows[] = $row->toArray();
        }

        $this->assertCount(3, $rows);
        $this->assertEquals([1, 2, 3], $rows[0]);
        $this->assertEquals([4, 5, 6], $rows[1]);
        $this->assertEquals([7, 8, 9], $rows[2]);
    }

    public function test2DArrayIterationViewsAreModifiable(): void
    {
        $matrix = NDArray::array([
            [1, 2],
            [3, 4],
        ]);

        foreach ($matrix as $row) {
            $row->set([0], 999);
        }

        // Original should be modified
        $this->assertEquals(999, $matrix->get(0, 0));
        $this->assertEquals(999, $matrix->get(1, 0));
    }

    public function test2DArrayIterationWithKeys(): void
    {
        $matrix = NDArray::array([[1, 2], [3, 4]]);
        $keys = [];

        foreach ($matrix as $key => $row) {
            $keys[] = $key;
        }

        $this->assertEquals([0, 1], $keys);
    }

    // ============================================================================
    // 3D Array Iteration Tests
    // ============================================================================

    public function test3DArrayIteration(): void
    {
        $tensor = NDArray::array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ]);

        $matrices = [];
        foreach ($tensor as $matrix) {
            $this->assertEquals([2, 2], $matrix->shape());
            $matrices[] = $matrix->toArray();
        }

        $this->assertCount(2, $matrices);
        $this->assertEquals([[1, 2], [3, 4]], $matrices[0]);
        $this->assertEquals([[5, 6], [7, 8]], $matrices[1]);
    }

    // ============================================================================
    // FlatIterator Tests - Basic Functionality
    // ============================================================================

    public function testFlatIterator1D(): void
    {
        $arr = NDArray::array([1, 2, 3, 4, 5]);
        $flat = $arr->flat();

        $values = iterator_to_array($flat);
        $this->assertEquals([1, 2, 3, 4, 5], $values);
    }

    public function testFlatIterator2D(): void
    {
        $matrix = NDArray::array([[1, 2], [3, 4]]);
        $flat = $matrix->flat();

        $values = iterator_to_array($flat);
        $this->assertEquals([1, 2, 3, 4], $values);
    }

    public function testFlatIterator3D(): void
    {
        $tensor = NDArray::array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ]);
        $flat = $tensor->flat();

        $values = iterator_to_array($flat);
        $this->assertEquals([1, 2, 3, 4, 5, 6, 7, 8], $values);
    }

    public function testFlatIteratorOrderIsCContiguous(): void
    {
        // 2D array - should iterate row by row
        $matrix = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
        ]);

        $values = iterator_to_array($matrix->flat());
        $this->assertEquals([1, 2, 3, 4, 5, 6], $values);
    }

    // ============================================================================
    // FlatIterator Tests - ArrayAccess
    // ============================================================================

    public function testFlatIteratorArrayAccess(): void
    {
        $matrix = NDArray::array([[1, 2], [3, 4]]);
        $flat = $matrix->flat();

        $this->assertEquals(1, $flat[0]);
        $this->assertEquals(2, $flat[1]);
        $this->assertEquals(3, $flat[2]);
        $this->assertEquals(4, $flat[3]);
    }

    public function testFlatIteratorArrayAccessOutOfBounds(): void
    {
        $arr = NDArray::array([1, 2, 3]);
        $flat = $arr->flat();

        $this->expectException(\OutOfBoundsException::class);
        $_unused = $flat[10];
    }

    public function testFlatIteratorArrayAccessNegativeIndex(): void
    {
        $arr = NDArray::array([1, 2, 3, 4, 5]);
        $flat = $arr->flat();

        // Negative indices should work like regular NDArray
        $this->assertTrue(isset($flat[-1]));
        $this->assertTrue(isset($flat[-5]));
        $this->assertFalse(isset($flat[-6])); // Out of bounds

        // Test accessing with negative indices
        $this->assertEquals(5, $flat[-1]); // Last element
        $this->assertEquals(4, $flat[-2]); // Second to last
        $this->assertEquals(1, $flat[-5]); // First element
    }

    public function testFlatIteratorArrayAccessAssignment(): void
    {
        $arr = NDArray::array([1, 2, 3]);
        $flat = $arr->flat();

        // Assignment should work
        $flat[0] = 999;
        $this->assertEquals(999, $arr->get(0));
        $this->assertEquals(999, $flat[0]);

        // Test negative index assignment
        $flat[-1] = 777;
        $this->assertEquals(777, $arr->get(2));
        $this->assertEquals(777, $flat[2]);
    }

    // ============================================================================
    // FlatIterator Tests - Countable
    // ============================================================================

    public function testFlatIteratorCount(): void
    {
        $arr = NDArray::array([1, 2, 3, 4, 5]);
        $flat = $arr->flat();

        $this->assertEquals(5, \count($flat));
    }

    public function testFlatIteratorCount2D(): void
    {
        $matrix = NDArray::array([[1, 2, 3], [4, 5, 6]]);
        $flat = $matrix->flat();

        $this->assertEquals(6, \count($flat));
    }

    // ============================================================================
    // FlatIterator Tests - toArray()
    // ============================================================================

    public function testFlatIteratorToArray(): void
    {
        $matrix = NDArray::array([[1, 2], [3, 4]]);
        $flat = $matrix->flat();

        $this->assertEquals([1, 2, 3, 4], $flat->toArray());
    }

    public function testFlatIteratorToArrayAfterIteration(): void
    {
        $arr = NDArray::array([1, 2, 3, 4, 5]);
        $flat = $arr->flat();

        // Iterate partially
        $flat->next();
        $flat->next();

        // toArray should still return all elements
        $this->assertEquals([1, 2, 3, 4, 5], $flat->toArray());

        // Position should be restored
        $this->assertEquals(2, $flat->key());
    }

    // ============================================================================
    // FlatIterator Tests - Rewind
    // ============================================================================

    public function testFlatIteratorRewind(): void
    {
        $arr = NDArray::array([1, 2, 3, 4, 5]);
        $flat = $arr->flat();

        // Iterate forward
        $flat->next();
        $flat->next();
        $this->assertEquals(3, $flat->current());

        // Rewind
        $flat->rewind();
        $this->assertEquals(1, $flat->current());
        $this->assertEquals(0, $flat->key());
    }

    public function testFlatIteratorRewindAndReiterate(): void
    {
        $arr = NDArray::array([1, 2, 3]);
        $flat = $arr->flat();

        // First iteration
        $values1 = iterator_to_array($flat);

        // Rewind and iterate again
        $flat->rewind();
        $values2 = iterator_to_array($flat);

        $this->assertEquals($values1, $values2);
    }

    // ============================================================================
    // FlatIterator Tests - Hybrid Mode
    // ============================================================================

    public function testFlatIteratorSmallArrayUsesBatchMode(): void
    {
        // Array with less than 100k elements should use batch mode
        $arr = NDArray::random([100, 100]); // 10,000 elements
        $flat = $arr->flat();

        // Should be able to iterate without issues
        $count = 0;
        foreach ($flat as $value) {
            ++$count;
        }

        $this->assertEquals(10000, $count);
    }

    public function testFlatIteratorLargeArrayUsesChunkedMode(): void
    {
        // Save original threshold
        $originalThreshold = FlatIterator::$chunkThreshold;
        $originalChunkSize = FlatIterator::$chunkSize;

        // Temporarily lower threshold for testing
        FlatIterator::$chunkThreshold = 100;
        FlatIterator::$chunkSize = 10;

        try {
            $arr = NDArray::random([20, 10]); // 200 elements > threshold
            $flat = $arr->flat();

            $count = 0;
            foreach ($flat as $value) {
                ++$count;
            }

            $this->assertEquals(200, $count);
        } finally {
            // Restore original values
            FlatIterator::$chunkThreshold = $originalThreshold;
            FlatIterator::$chunkSize = $originalChunkSize;
        }
    }

    // ============================================================================
    // FlatIterator Tests - Views
    // ============================================================================

    public function testFlatIteratorWithView(): void
    {
        $arr = NDArray::array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        $view = $arr->slice(['0:2', '1:3']); // Shape [2, 2]

        $flat = $view->flat();
        $values = iterator_to_array($flat);

        // Should be [2, 3, 5, 6] (C-order of the view)
        $this->assertEquals([2, 3, 5, 6], $values);
    }

    public function testFlatIteratorWithTransposedView(): void
    {
        $arr = NDArray::array([[1, 2], [3, 4]]);
        $transposed = $arr->transpose(); // [[1, 3], [2, 4]]

        $flat = $transposed->flat();
        $values = iterator_to_array($flat);

        // Should iterate in C-order of transposed array
        $this->assertEquals([1, 3, 2, 4], $values);
    }

    // ============================================================================
    // Empty Array Tests
    // ============================================================================

    public function testEmpty1DArrayIteration(): void
    {
        // Create a 1D array with 0 elements using empty
        $arr = NDArray::zeros([0]);
        $count = 0;

        foreach ($arr as $value) {
            ++$count;
        }

        $this->assertEquals(0, $count);
    }

    // ============================================================================
    // Multiple Iterations
    // ============================================================================

    public function testMultipleForeachOnSameArray(): void
    {
        $arr = NDArray::array([1, 2, 3]);

        $sum1 = 0;
        foreach ($arr as $value) {
            $sum1 += $value;
        }

        $sum2 = 0;
        foreach ($arr as $value) {
            $sum2 += $value;
        }

        $this->assertEquals($sum1, $sum2);
        $this->assertEquals(6, $sum1);
    }
}
