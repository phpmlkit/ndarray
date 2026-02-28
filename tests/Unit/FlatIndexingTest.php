<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Unit;

use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\Exceptions\IndexException;
use PhpMlKit\NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Tests for flat indexing methods: setAt() and getAt().
 *
 * These methods provide C-order flat indexing for arrays and views.
 *
 * @internal
 *
 * @coversNothing
 */
class FlatIndexingTest extends TestCase
{
    public function testGetAtBasic(): void
    {
        $arr = NDArray::array([[1, 2, 3], [4, 5, 6]]);

        $this->assertEquals(1, $arr->getAt(0));  // First element
        $this->assertEquals(2, $arr->getAt(1));  // Second element
        $this->assertEquals(4, $arr->getAt(3));  // First element of second row
        $this->assertEquals(6, $arr->getAt(5));  // Last element
    }

    public function testGetAt1DArray(): void
    {
        $arr = NDArray::array([10, 20, 30, 40, 50]);

        $this->assertEquals(10, $arr->getAt(0));
        $this->assertEquals(30, $arr->getAt(2));
        $this->assertEquals(50, $arr->getAt(4));
    }

    public function testGetAtNegativeIndex(): void
    {
        $arr = NDArray::array([[1, 2, 3], [4, 5, 6]]);

        $this->assertEquals(6, $arr->getAt(-1));  // Last element
        $this->assertEquals(5, $arr->getAt(-2));  // Second to last
        $this->assertEquals(1, $arr->getAt(-6));  // First element
    }

    public function testGetAt3DArray(): void
    {
        $arr = NDArray::array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ]);

        // C-order: [1, 2, 3, 4, 5, 6, 7, 8]
        $this->assertEquals(1, $arr->getAt(0));
        $this->assertEquals(4, $arr->getAt(3));
        $this->assertEquals(8, $arr->getAt(7));
    }

    public function testGetAtOnView(): void
    {
        $arr = NDArray::array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ]);
        $view = $arr->slice(['1:3', '1:3']);  // [[6, 7], [10, 11]]

        // Flat indexing on view
        $this->assertEquals(6, $view->getAt(0));
        $this->assertEquals(7, $view->getAt(1));
        $this->assertEquals(10, $view->getAt(2));
        $this->assertEquals(11, $view->getAt(3));
    }

    public function testGetAtOnTransposedView(): void
    {
        $arr = NDArray::array([[1, 2, 3], [4, 5, 6]]);  // Shape [2, 3]
        $transposed = $arr->transpose();  // Shape [3, 2], data [[1, 4], [2, 5], [3, 6]]

        $this->assertEquals(1, $transposed->getAt(0));
        $this->assertEquals(4, $transposed->getAt(1));
        $this->assertEquals(2, $transposed->getAt(2));
    }

    public function testGetAtOutOfBoundsThrows(): void
    {
        $arr = NDArray::array([1, 2, 3]);

        $this->expectException(IndexException::class);
        $arr->getAt(10);
    }

    public function testGetAtNegativeOutOfBoundsThrows(): void
    {
        $arr = NDArray::array([1, 2, 3]);

        $this->expectException(IndexException::class);
        $arr->getAt(-4);
    }

    public function testGetAtDifferentDTypes(): void
    {
        $dtypes = [DType::Int8, DType::Int32, DType::Float32, DType::Float64, DType::Bool];

        foreach ($dtypes as $dtype) {
            $arr = NDArray::array([1, 2, 3, 4, 5], $dtype);
            $value = $arr->getAt(2);
            $this->assertEquals(3, $value, "getAt failed for dtype: {$dtype->name}");
        }
    }

    public function testSetAtBasic(): void
    {
        $arr = NDArray::array([[1, 2, 3], [4, 5, 6]]);

        $arr->setAt(0, 100);
        $this->assertEquals(100, $arr->getAt(0));

        $arr->setAt(5, 999);
        $this->assertEquals(999, $arr->getAt(5));
    }

    public function testSetAt1DArray(): void
    {
        $arr = NDArray::array([10, 20, 30, 40]);

        $arr->setAt(1, 99);
        $this->assertEquals(99, $arr->getAt(1));
        $this->assertEquals([10, 99, 30, 40], $arr->toArray());
    }

    public function testSetAtNegativeIndex(): void
    {
        $arr = NDArray::array([[1, 2, 3], [4, 5, 6]]);

        $arr->setAt(-1, 999);  // Last element
        $this->assertEquals(999, $arr->getAt(-1));
        $this->assertEquals(999, $arr->get(1, 2));

        $arr->setAt(-3, 888);  // Third from last
        $this->assertEquals(888, $arr->getAt(-3));
    }

    public function testSetAtOnView(): void
    {
        $arr = NDArray::array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ]);
        $view = $arr->slice(['1:3', '1:3']);  // [[6, 7], [10, 11]]

        $view->setAt(0, 999);
        $this->assertEquals(999, $view->getAt(0));
        $this->assertEquals(999, $arr->get(1, 1));  // Original is modified
    }

    public function testSetAtMultipleValues(): void
    {
        $arr = NDArray::array([0, 0, 0, 0, 0]);

        $arr->setAt(0, 1);
        $arr->setAt(2, 2);
        $arr->setAt(4, 3);

        $this->assertEquals([1, 0, 2, 0, 3], $arr->toArray());
    }

    public function testSetAtOutOfBoundsThrows(): void
    {
        $arr = NDArray::array([1, 2, 3]);

        $this->expectException(IndexException::class);
        $arr->setAt(10, 999);
    }

    public function testSetAtDifferentDTypes(): void
    {
        // Int32
        $arr = NDArray::array([1, 2, 3], DType::Int32);
        $arr->setAt(0, 999);
        $this->assertEquals(999, $arr->getAt(0));

        // Float64
        $arr = NDArray::array([1.0, 2.0, 3.0], DType::Float64);
        $arr->setAt(1, 3.14);
        $this->assertEqualsWithDelta(3.14, $arr->getAt(1), 0.001);

        // Bool
        $arr = NDArray::array([true, false, true], DType::Bool);
        $arr->setAt(1, true);
        $this->assertTrue($arr->getAt(1));
    }

    public function testGetAtSingleElement(): void
    {
        $arr = NDArray::array([42]);
        $this->assertEquals(42, $arr->getAt(0));
        $this->assertEquals(42, $arr->getAt(-1));
    }

    public function testSetAtSingleElement(): void
    {
        $arr = NDArray::array([42]);
        $arr->setAt(0, 999);
        $this->assertEquals(999, $arr->getAt(0));
    }

    public function testGetAtZeroDimArray(): void
    {
        $arr = NDArray::array([42])->reshape([]);
        $this->assertEquals(42, $arr->getAt(0));
    }

    public function testSetAtZeroDimArray(): void
    {
        $arr = NDArray::array([42])->reshape([]);
        $arr->setAt(0, 999);
        $this->assertEquals(999, $arr->getAt(0));
    }

    public function testGetAtEmptySliceThrows(): void
    {
        $arr = NDArray::array([1, 2, 3, 4, 5]);
        $empty = $arr->slice(['2:2']); // Empty slice with shape [0]

        $this->expectException(IndexException::class);
        $empty->getAt(0);
    }

    public function testSetAtEmptySliceThrows(): void
    {
        $arr = NDArray::array([1, 2, 3, 4, 5]);
        $empty = $arr->slice(['2:2']); // Empty slice with shape [0]

        $this->expectException(IndexException::class);
        $empty->setAt(0, 1);
    }
}
