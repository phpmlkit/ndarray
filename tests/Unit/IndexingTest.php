<?php

declare(strict_types=1);

namespace NDArray\Tests\Unit;

use NDArray\DType;
use NDArray\Exceptions\IndexException;
use NDArray\NDArray;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;

/**
 * Tests for NDArray indexing: get(), set(), ArrayAccess, and view behavior.
 */
final class IndexingTest extends TestCase
{
    // =========================================================================
    // Scalar Access (get) — Full Indexing
    // =========================================================================

    public function testGet1DScalar(): void
    {
        $arr = NDArray::array([10, 20, 30, 40], DType::Int64);

        $this->assertSame(10, $arr->get(0));
        $this->assertSame(20, $arr->get(1));
        $this->assertSame(30, $arr->get(2));
        $this->assertSame(40, $arr->get(3));
    }

    public function testGet2DScalar(): void
    {
        $arr = NDArray::array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ], DType::Float64);

        $this->assertEqualsWithDelta(1.0, $arr->get(0, 0), 0.0001);
        $this->assertEqualsWithDelta(3.0, $arr->get(0, 2), 0.0001);
        $this->assertEqualsWithDelta(4.0, $arr->get(1, 0), 0.0001);
        $this->assertEqualsWithDelta(6.0, $arr->get(1, 2), 0.0001);
    }

    public function testGet3DScalar(): void
    {
        $arr = NDArray::array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ], DType::Int32);

        $this->assertSame(1, $arr->get(0, 0, 0));
        $this->assertSame(4, $arr->get(0, 1, 1));
        $this->assertSame(5, $arr->get(1, 0, 0));
        $this->assertSame(8, $arr->get(1, 1, 1));
    }

    // =========================================================================
    // Scalar Access — All DTypes
    // =========================================================================

    #[DataProvider('allDTypesProvider')]
    public function testGetScalarAllDTypes(DType $dtype, array $data, mixed $expected): void
    {
        $arr = NDArray::array($data, $dtype);
        $result = $arr->get(0);

        if ($dtype === DType::Bool) {
            $this->assertSame($expected, $result);
        } elseif ($dtype->isFloat()) {
            $this->assertEqualsWithDelta($expected, $result, 0.001);
        } else {
            $this->assertSame($expected, $result);
        }
    }

    public static function allDTypesProvider(): array
    {
        return [
            'Int8' => [DType::Int8, [42], 42],
            'Int16' => [DType::Int16, [1000], 1000],
            'Int32' => [DType::Int32, [100000], 100000],
            'Int64' => [DType::Int64, [9999999], 9999999],
            'Uint8' => [DType::Uint8, [200], 200],
            'Uint16' => [DType::Uint16, [50000], 50000],
            'Uint32' => [DType::Uint32, [3000000], 3000000],
            'Uint64' => [DType::Uint64, [123456789], 123456789],
            'Float32' => [DType::Float32, [3.14], 3.14],
            'Float64' => [DType::Float64, [2.718281828], 2.718281828],
            'Bool' => [DType::Bool, [true, false], true],
        ];
    }

    // =========================================================================
    // Partial Indexing — View Creation
    // =========================================================================

    public function testPartialIndex2DReturnsView(): void
    {
        $arr = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
        ], DType::Int64);

        $row = $arr->get(0);

        $this->assertInstanceOf(NDArray::class, $row);
        $this->assertTrue($row->isView());
        $this->assertSame([3], $row->shape());
        $this->assertSame(1, $row->ndim());
        $this->assertSame(3, $row->size());
        $this->assertSame([1, 2, 3], $row->toArray());

        $row2 = $arr->get(1);
        $this->assertSame([4, 5, 6], $row2->toArray());
    }

    public function testPartialIndex3DReturnsView(): void
    {
        $arr = NDArray::array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ], DType::Int32);

        // Single index on 3D -> 2D view
        $matrix = $arr->get(0);
        $this->assertSame([2, 2], $matrix->shape());
        $this->assertSame([[1, 2], [3, 4]], $matrix->toArray());

        // Two indices on 3D -> 1D view
        $row = $arr->get(1, 0);
        $this->assertSame([2], $row->shape());
        $this->assertSame([5, 6], $row->toArray());
    }

    public function testViewScalarAccess(): void
    {
        $arr = NDArray::array([
            [10, 20, 30],
            [40, 50, 60],
        ], DType::Int64);

        $row = $arr->get(1);
        $this->assertSame(40, $row->get(0));
        $this->assertSame(50, $row->get(1));
        $this->assertSame(60, $row->get(2));
    }

    // =========================================================================
    // View-of-View Chaining
    // =========================================================================

    public function testViewOfViewChaining(): void
    {
        $arr = NDArray::array([
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ], DType::Int32);

        // 3D -> get(1) -> 2D view -> get(0) -> 1D view -> get(2) -> scalar
        $matrix = $arr->get(1);          // [[7,8,9],[10,11,12]]
        $this->assertTrue($matrix->isView());
        $this->assertSame([2, 3], $matrix->shape());

        $row = $matrix->get(0);          // [7,8,9]
        $this->assertTrue($row->isView());
        $this->assertSame([3], $row->shape());

        $scalar = $row->get(2);          // 9
        $this->assertSame(9, $scalar);
    }

    // =========================================================================
    // set() Tests
    // =========================================================================

    public function testSetScalar1D(): void
    {
        $arr = NDArray::array([1, 2, 3], DType::Int64);

        $arr->set([1], 99);

        $this->assertSame(99, $arr->get(1));
        $this->assertSame([1, 99, 3], $arr->toArray());
    }

    public function testSetAtScalar1D(): void
    {
        $arr = NDArray::array([10, 20, 30, 40], DType::Int64);
        $arr->setAt(2, 99);

        $this->assertSame([10, 20, 99, 40], $arr->toArray());
    }

    public function testSetAtNegativeIndex(): void
    {
        $arr = NDArray::array([10, 20, 30, 40], DType::Int64);
        $arr->setAt(-1, 77);

        $this->assertSame([10, 20, 30, 77], $arr->toArray());
    }

    public function testSetAtOnViewUsesLogicalOrder(): void
    {
        $arr = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Int64);

        $view = $arr->slice(['::2', ':']); // rows 0 and 2, shape [2, 3]
        $view->setAt(4, 99); // logical flat index 4 => row 1 col 1 inside view => arr[2,1]

        $this->assertSame(99, $arr->get(2, 1));
        $this->assertSame([
            [1, 2, 3],
            [4, 5, 6],
            [7, 99, 9],
        ], $arr->toArray());
    }

    public function testSetScalar2D(): void
    {
        $arr = NDArray::array([
            [1.0, 2.0],
            [3.0, 4.0],
        ], DType::Float64);

        $arr->set([0, 1], 99.5);

        $this->assertEqualsWithDelta(99.5, $arr->get(0, 1), 0.0001);
    }

    public function testSetBool(): void
    {
        $arr = NDArray::array([true, false, true], DType::Bool);

        $arr->set([1], true);

        $this->assertSame(true, $arr->get(1));
    }

    // =========================================================================
    // set() Through Views — Shared Memory
    // =========================================================================

    public function testSetThroughViewMutatesRoot(): void
    {
        $arr = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
        ], DType::Int64);

        // Get a view of the second row
        $row = $arr->get(1);

        // Mutate through the view
        $row->set([0], 99);

        // Verify the root array is also mutated (shared memory)
        $this->assertSame(99, $arr->get(1, 0));
        $this->assertSame([
            [1, 2, 3],
            [99, 5, 6],
        ], $arr->toArray());
    }

    // =========================================================================
    // ArrayAccess — Integer Keys
    // =========================================================================

    public function testArrayAccessIntGet(): void
    {
        $arr = NDArray::array([
            [10, 20],
            [30, 40],
        ], DType::Int64);

        $row = $arr[0];
        $this->assertInstanceOf(NDArray::class, $row);
        $this->assertSame([10, 20], $row->toArray());
    }

    public function testArrayAccessIntScalar(): void
    {
        $arr = NDArray::array([100, 200, 300], DType::Int64);

        $this->assertSame(100, $arr[0]);
        $this->assertSame(200, $arr[1]);
        $this->assertSame(300, $arr[2]);
    }

    // =========================================================================
    // ArrayAccess — String Keys (Comma-Separated)
    // =========================================================================

    public function testArrayAccessStringMultiIndex(): void
    {
        $arr = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
        ], DType::Int64);

        $this->assertSame(5, $arr['1,1']);
        $this->assertSame(3, $arr['0,2']);
        $this->assertSame(6, $arr['1, 2']); // with space
    }

    public function testArrayAccessStringSet(): void
    {
        $arr = NDArray::array([
            [1, 2],
            [3, 4],
        ], DType::Int64);

        $arr['0,1'] = 99;

        $this->assertSame(99, $arr->get(0, 1));
    }

    // =========================================================================
    // ArrayAccess — offsetExists
    // =========================================================================

    public function testOffsetExists(): void
    {
        $arr = NDArray::array([1, 2, 3], DType::Int64);

        $this->assertTrue(isset($arr[0]));
        $this->assertTrue(isset($arr[2]));
        $this->assertFalse(isset($arr[3]));
        // Negative indices are now valid!
        $this->assertTrue(isset($arr[-1]));
        $this->assertTrue(isset($arr[-2]));
        $this->assertTrue(isset($arr[-3]));
        $this->assertFalse(isset($arr[-4]));
    }

    // =========================================================================
    // ArrayAccess — Slice Syntax Works
    // =========================================================================

    public function testSliceSyntaxWorks(): void
    {
        $arr = NDArray::array([1, 2, 3, 4], DType::Int64);

        $slice = $arr['0:2'];
        $this->assertSame([1, 2], $slice->toArray());
    }

    // =========================================================================
    // Error Cases
    // =========================================================================

    public function testGetNoIndicesThrows(): void
    {
        $arr = NDArray::array([1, 2, 3], DType::Int64);

        $this->expectException(IndexException::class);
        $this->expectExceptionMessage('At least one index is required');

        $arr->get();
    }

    public function testGetTooManyIndicesThrows(): void
    {
        $arr = NDArray::array([1, 2, 3], DType::Int64);

        $this->expectException(IndexException::class);
        $this->expectExceptionMessage('Too many indices');

        $arr->get(0, 0);
    }

    public function testGetOutOfBoundsThrows(): void
    {
        $arr = NDArray::array([1, 2, 3], DType::Int64);

        $this->expectException(IndexException::class);
        $this->expectExceptionMessage('out of bounds');

        $arr->get(5);
    }

    // =========================================================================
    // Negative Indexing
    // =========================================================================

    public function testGetNegativeIndex1D(): void
    {
        $arr = NDArray::array([10, 20, 30, 40, 50], DType::Int64);

        // -1 should return last element
        $this->assertSame(50, $arr->get(-1));
        // -2 should return second-to-last
        $this->assertSame(40, $arr->get(-2));
        // -5 should return first element
        $this->assertSame(10, $arr->get(-5));
    }

    public function testGetNegativeIndex2D(): void
    {
        $arr = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Int64);

        // -1, -1 should be bottom-right corner
        $this->assertSame(9, $arr->get(-1, -1));
        // -1, 0 should be first element of last row
        $this->assertSame(7, $arr->get(-1, 0));
        // 0, -1 should be last element of first row
        $this->assertSame(3, $arr->get(0, -1));
        // -2, -2 should be middle
        $this->assertSame(5, $arr->get(-2, -2));
    }

    public function testGetNegativeIndex3D(): void
    {
        $arr = NDArray::array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ], DType::Int32);

        // -1, -1, -1 should be last element
        $this->assertSame(8, $arr->get(-1, -1, -1));
        // -1, 0, 0 should be first element of last matrix
        $this->assertSame(5, $arr->get(-1, 0, 0));
    }

    public function testGetNegativeIndexOutOfBoundsThrows(): void
    {
        $arr = NDArray::array([1, 2, 3], DType::Int64);

        $this->expectException(IndexException::class);
        $this->expectExceptionMessage('out of bounds');

        // -4 is out of bounds for array of size 3 (would be index -1)
        $arr->get(-4);
    }

    public function testSetNegativeIndex(): void
    {
        $arr = NDArray::array([1, 2, 3, 4, 5], DType::Int64);

        // Set last element using negative index
        $arr->set([-1], 99);
        $this->assertSame(99, $arr->get(-1));
        $this->assertSame([1, 2, 3, 4, 99], $arr->toArray());

        // Set first element using negative index
        $arr->set([-5], 11);
        $this->assertSame(11, $arr->get(0));
        $this->assertSame([11, 2, 3, 4, 99], $arr->toArray());
    }

    public function testSetNegativeIndex2D(): void
    {
        $arr = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Int64);

        // Set bottom-right corner
        $arr->set([-1, -1], 99);
        $this->assertSame(99, $arr->get(2, 2));

        // Set first element of last row
        $arr->set([-1, 0], 77);
        $this->assertSame(77, $arr->get(2, 0));
    }

    public function testSetAtOutOfBoundsThrows(): void
    {
        $arr = NDArray::array([1, 2, 3], DType::Int64);

        $this->expectException(IndexException::class);
        $this->expectExceptionMessage('out of bounds');

        $arr->setAt(3, 10);
    }

    // =========================================================================
    // take / takeAlongAxis
    // =========================================================================

    public function testTakeFlatArrayIndices(): void
    {
        $arr = NDArray::array([[10, 20, 30], [40, 50, 60]], DType::Int64);
        $result = $arr->take([0, 2, -1]);

        $this->assertSame([3], $result->shape());
        $this->assertSame([10, 30, 60], $result->toArray());
    }

    public function testTakeFlatNestedIndicesPreservesShape(): void
    {
        $arr = NDArray::array([1, 2, 3, 4, 5, 6], DType::Int64);
        $result = $arr->take([[0, 1], [4, 5]]);

        $this->assertSame([2, 2], $result->shape());
        $this->assertSame([[1, 2], [5, 6]], $result->toArray());
    }

    public function testTakeAlongAxis(): void
    {
        $arr = NDArray::array([[10, 20, 30], [40, 50, 60]], DType::Int64);
        $indices = NDArray::array([[2, 1], [0, 2]], DType::Int64);
        $result = $arr->takeAlongAxis($indices, 1);

        $this->assertSame([2, 2], $result->shape());
        $this->assertSame([[30, 20], [40, 60]], $result->toArray());
    }

    public function testTakeOnView(): void
    {
        $arr = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Int64);
        $view = $arr->slice(['::2', ':']); // [[1,2,3],[7,8,9]]
        $result = $view->take([1, 4, 5]);

        $this->assertSame([3], $result->shape());
        $this->assertSame([2, 8, 9], $result->toArray());
    }

    // =========================================================================
    // put / putAlongAxis / scatterAdd
    // =========================================================================

    public function testPutFlatScalar(): void
    {
        $arr = NDArray::array([1, 2, 3, 4], DType::Int64);
        $result = $arr->put([1, -1], 9);

        $this->assertSame([1, 9, 3, 9], $result->toArray());
        $this->assertSame([1, 2, 3, 4], $arr->toArray());
    }

    public function testPutFlatCyclesValues(): void
    {
        $arr = NDArray::array([10, 20, 30, 40], DType::Int64);
        $vals = NDArray::array([1, 2], DType::Int64);
        $result = $arr->put([0, 1, 2, 3], $vals);

        $this->assertSame([1, 2, 1, 2], $result->toArray());
    }

    public function testPutAlongAxis(): void
    {
        $arr = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Int64);
        $indices = NDArray::array([[0, 2], [1, 0]], DType::Int64);
        $values = NDArray::array([[9, 8], [7, 6]], DType::Int64);
        $result = $arr->putAlongAxis($indices, $values, 1);

        $this->assertSame([[9, 2, 8], [6, 7, 6]], $result->toArray());
    }

    public function testScatterAddFlat(): void
    {
        $arr = NDArray::array([1, 2, 3, 4], DType::Int64);
        $result = $arr->scatterAdd([1, 1, 3], NDArray::array([5, 6, 7], DType::Int64));

        $this->assertSame([1, 13, 3, 11], $result->toArray());
    }

    // =========================================================================
    // Static where
    // =========================================================================

    public function testStaticWhereArrayCondition(): void
    {
        $cond = NDArray::array([true, false, true], DType::Bool);
        $x = NDArray::array([1, 2, 3], DType::Int64);
        $y = NDArray::array([10, 20, 30], DType::Int64);

        $result = NDArray::where($cond, $x, $y);
        $this->assertSame([1, 20, 3], $result->toArray());
    }

    public function testStaticWhereScalarConditionBroadcasts(): void
    {
        $x = NDArray::array([[1, 2], [3, 4]], DType::Int64);
        $y = NDArray::array([[10, 20], [30, 40]], DType::Int64);

        $resultTrue = NDArray::where(true, $x, $y);
        $this->assertSame([[1, 2], [3, 4]], $resultTrue->toArray());

        $resultFalse = NDArray::where(false, $x, $y);
        $this->assertSame([[10, 20], [30, 40]], $resultFalse->toArray());
    }

    public function testArrayAccessNegativeIndex(): void
    {
        $arr = NDArray::array([10, 20, 30, 40], DType::Int64);

        // Test ArrayAccess with negative string indices
        $this->assertSame(40, $arr['-1']);
        $this->assertSame(30, $arr['-2']);
        $this->assertSame(10, $arr['-4']);
    }

    public function testArrayAccessNegativeMultiIndex(): void
    {
        $arr = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Int64);

        // Test multi-dimensional negative indexing via ArrayAccess
        $this->assertSame(9, $arr['-1,-1']);
        $this->assertSame(7, $arr['-1,0']);
        $this->assertSame(3, $arr['0,-1']);
        $this->assertSame(5, $arr['-2,-2']);
    }

    public function testArrayAccessNegativeIndexSet(): void
    {
        $arr = NDArray::array([1, 2, 3, 4, 5], DType::Int64);

        // Set using negative index
        $arr['-1'] = 99;
        $this->assertSame(99, $arr[4]);
        $this->assertSame([1, 2, 3, 4, 99], $arr->toArray());
    }

    public function testArrayAccessNegativeMultiIndexSet(): void
    {
        $arr = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Int64);

        // Set using negative multi-index
        $arr['-1,-1'] = 99;
        $this->assertSame(99, $arr->get(2, 2));
    }

    public function testOffsetExistsWithNegativeIndex(): void
    {
        $arr = NDArray::array([1, 2, 3], DType::Int64);

        // These should exist
        $this->assertTrue(isset($arr[-1]));
        $this->assertTrue(isset($arr[-2]));
        $this->assertTrue(isset($arr[-3]));

        // These should not exist
        $this->assertFalse(isset($arr[-4]));
        $this->assertFalse(isset($arr[3]));
    }

    public function testPartialIndexWithNegativeIndex(): void
    {
        $arr = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Int64);

        // Get last row using negative index
        $lastRow = $arr->get(-1);
        $this->assertInstanceOf(NDArray::class, $lastRow);
        $this->assertTrue($lastRow->isView());
        $this->assertSame([7, 8, 9], $lastRow->toArray());

        // Get first row using negative index (size is 3, so -3 = 0)
        $firstRow = $arr->get(-3);
        $this->assertSame([1, 2, 3], $firstRow->toArray());
    }

    public function testViewOfViewWithNegativeIndex(): void
    {
        $arr = NDArray::array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ], DType::Int32);

        // Get last matrix
        $lastMatrix = $arr->get(-1);
        $this->assertSame([[5, 6], [7, 8]], $lastMatrix->toArray());

        // Get last row of last matrix
        $lastRow = $lastMatrix->get(-1);
        $this->assertSame([7, 8], $lastRow->toArray());

        // Get last element
        $this->assertSame(8, $lastRow->get(-1));
    }

    public function testNegativeIndexAllDTypes(): void
    {
        // Test negative indexing works with all dtypes
        $dtypes = [
            DType::Int8,
            DType::Int16,
            DType::Int32,
            DType::Int64,
            DType::Uint8,
            DType::Uint16,
            DType::Uint32,
            DType::Uint64,
            DType::Float32,
            DType::Float64,
        ];

        foreach ($dtypes as $dtype) {
            $arr = NDArray::array([10, 20, 30, 40, 50], $dtype);
            $this->assertEqualsWithDelta(50, $arr->get(-1), 0.001, "Failed for $dtype->name");
            $this->assertEqualsWithDelta(40, $arr->get(-2), 0.001, "Failed for $dtype->name");
            $this->assertEqualsWithDelta(10, $arr->get(-5), 0.001, "Failed for $dtype->name");
        }

        // Test Bool separately since values are different
        $boolArr = NDArray::array([true, false, true, false], DType::Bool);
        $this->assertSame(false, $boolArr->get(-1));
        $this->assertSame(true, $boolArr->get(-2));
        $this->assertSame(true, $boolArr->get(-4));
    }

    public function testSetPartialIndicesThrows(): void
    {
        $arr = NDArray::array([
            [1, 2],
            [3, 4],
        ], DType::Int64);

        $this->expectException(IndexException::class);
        $this->expectExceptionMessage('requires exactly 2 indices');

        $arr->set([0], 99);
    }

    public function testOffsetUnsetThrows(): void
    {
        $arr = NDArray::array([1, 2, 3], DType::Int64);

        $this->expectException(IndexException::class);
        $this->expectExceptionMessage('Cannot unset');

        unset($arr[0]);
    }

    public function testArrayAccessInvalidStringThrows(): void
    {
        $arr = NDArray::array([1, 2, 3], DType::Int64);

        $this->expectException(IndexException::class);
        $this->expectExceptionMessage('Invalid index component');

        /** @noinspection PhpUnusedLocalVariableInspection */
        $_ = $arr['abc'];
    }

    // =========================================================================
    // isView Property
    // =========================================================================

    public function testRootArrayIsNotView(): void
    {
        $arr = NDArray::array([1, 2, 3], DType::Int64);
        $this->assertFalse($arr->isView());
    }

    public function testViewIsView(): void
    {
        $arr = NDArray::array([[1, 2], [3, 4]], DType::Int64);
        $view = $arr->get(0);
        $this->assertTrue($view->isView());
    }

    // =========================================================================
    // View toArray()
    // =========================================================================

    public function testViewToArray(): void
    {
        $arr = NDArray::array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], DType::Float64);

        $row0 = $arr->get(0);
        $this->assertEqualsWithDelta([1.0, 2.0, 3.0], $row0->toArray(), 0.0001);

        $row2 = $arr->get(2);
        $this->assertEqualsWithDelta([7.0, 8.0, 9.0], $row2->toArray(), 0.0001);
    }

    public function testViewToArrayBool(): void
    {
        $arr = NDArray::array([
            [true, false],
            [false, true],
        ], DType::Bool);

        $row = $arr->get(0);
        $this->assertSame([true, false], $row->toArray());
    }

    // =========================================================================
    // View Keeps Root Alive (Destructor Safety)
    // =========================================================================

    public function testViewKeepsRootAlive(): void
    {
        // Create view, then release the root reference
        $view = (function () {
            $arr = NDArray::array([[10, 20], [30, 40]], DType::Int64);
            return $arr->get(1); // view of row [30, 40]
        })();

        // Root's PHP object is out of scope, but view holds $base reference
        // This should NOT segfault — the Rust handle is still alive
        $this->assertSame([30, 40], $view->toArray());
        $this->assertSame(30, $view->get(0));
        $this->assertSame(40, $view->get(1));
    }

    // =========================================================================
    // ArrayAccess Partial Set Broadcasts
    // =========================================================================

    public function testArrayAccessPartialSetBroadcasts(): void
    {
        $arr = NDArray::array([
            [1, 2],
            [3, 4],
        ], DType::Int64);

        // Assign scalar to row (broadcast)
        $arr[0] = 99;
        
        $this->assertSame([
            [99, 99],
            [3, 4],
        ], $arr->toArray());
    }

    // =========================================================================
    // 4D View Access
    // =========================================================================

    public function test4DPartialIndexing(): void
    {
        // Shape [2, 2, 2, 2] = 16 elements
        $data = [
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            [[[9, 10], [11, 12]], [[13, 14], [15, 16]]],
        ];
        $arr = NDArray::array($data, DType::Int32);

        // 4D -> get(1) -> 3D view
        $cube = $arr->get(1);
        $this->assertSame([2, 2, 2], $cube->shape());
        $this->assertSame([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], $cube->toArray());

        // 4D -> get(1,0) -> 2D view
        $matrix = $arr->get(1, 0);
        $this->assertSame([2, 2], $matrix->shape());
        $this->assertSame([[9, 10], [11, 12]], $matrix->toArray());

        // 4D -> get(1,0,1) -> 1D view
        $row = $arr->get(1, 0, 1);
        $this->assertSame([2], $row->shape());
        $this->assertSame([11, 12], $row->toArray());

        // 4D -> get(1,0,1,0) -> scalar
        $this->assertSame(11, $arr->get(1, 0, 1, 0));
    }
}
