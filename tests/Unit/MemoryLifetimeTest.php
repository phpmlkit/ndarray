<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Unit;

use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Comprehensive memory lifetime tests for view chains and nested views.
 *
 * These tests verify that the PHP reference counting and Rust memory management
 * work correctly together, especially for complex view chains.
 *
 * @internal
 *
 * @coversNothing
 */
final class MemoryLifetimeTest extends TestCase
{
    // =========================================================================
    // Basic Root-View Lifetime Tests
    // =========================================================================

    public function testViewSurvivesRootRelease(): void
    {
        $view = (static function () {
            $arr = NDArray::array([[1, 2], [3, 4], [5, 6]], DType::Int64);

            return $arr->get(1); // view of row [3, 4]
        })();

        // Root's PHP object is out of scope, but view holds $base reference
        $this->assertSame([3, 4], $view->toArray());
        $this->assertSame(3, $view->get(0));
        $this->assertSame(4, $view->get(1));
    }

    public function testMultipleViewsFromSameRoot(): void
    {
        $arr = NDArray::array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], DType::Int64);

        $view1 = $arr->get(0); // [1, 2, 3]
        $view2 = $arr->get(1); // [4, 5, 6]
        $view3 = $arr->get(2); // [7, 8, 9]

        // All views should be independent but share the same root
        $this->assertSame([1, 2, 3], $view1->toArray());
        $this->assertSame([4, 5, 6], $view2->toArray());
        $this->assertSame([7, 8, 9], $view3->toArray());

        // Modify through one view
        $view2[0] = 100;

        // Verify the change is visible in the root
        $this->assertSame(100, $arr->get(1, 0));
    }

    // =========================================================================
    // Nested View Chain Tests
    // =========================================================================

    public function testNestedViewChain2D(): void
    {
        $arr = NDArray::array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]],
        ], DType::Int64);

        // 3D -> 2D view
        $view2d = $arr->get(1); // [[5, 6], [7, 8]]
        $this->assertSame([[5, 6], [7, 8]], $view2d->toArray());

        // 2D view -> 1D view
        $view1d = $view2d->get(0); // [5, 6]
        $this->assertSame([5, 6], $view1d->toArray());

        // 1D view -> scalar
        $scalar = $view1d->get(1); // 6
        $this->assertSame(6, $scalar);
    }

    public function testDeepNestedViewChain(): void
    {
        $arr = NDArray::array([
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            [[[9, 10], [11, 12]], [[13, 14], [15, 16]]],
        ], DType::Int64);

        // 4D -> 3D -> 2D -> 1D -> scalar
        $view3d = $arr->get(0);
        $view2d = $view3d->get(1);
        $view1d = $view2d->get(0);
        $scalar = $view1d->get(1);

        $this->assertSame(6, $scalar);
    }

    public function testNestedViewSurvivesIntermediateRelease(): void
    {
        $deepView = (static function () {
            $arr = NDArray::array([
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
            ], DType::Int64);

            $view2d = $arr->get(1); // Get 2D view

            return $view2d->get(0); // Get 1D view from it
            // Return only the deepest view
              // [5, 6]
        })();

        // All intermediate views and root are out of scope
        // But deepView should still work due to reference chain
        $this->assertSame([5, 6], $deepView->toArray());
        $this->assertSame(5, $deepView->get(0));
        $this->assertSame(6, $deepView->get(1));
    }

    public function testMultipleNestedViewsFromSameBranch(): void
    {
        $arr = NDArray::array([
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ], DType::Int64);

        // Get 2D view from first branch
        $branch1 = $arr->get(0);

        // Create multiple views from this branch
        $leaf1 = $branch1->get(0); // [1, 2, 3]
        $leaf2 = $branch1->get(1); // [4, 5, 6]

        $this->assertSame([1, 2, 3], $leaf1->toArray());
        $this->assertSame([4, 5, 6], $leaf2->toArray());

        // Release the branch reference
        unset($branch1);

        // Leaves should still work
        $this->assertSame([1, 2, 3], $leaf1->toArray());
        $this->assertSame([4, 5, 6], $leaf2->toArray());
    }

    // =========================================================================
    // Slice View Chain Tests
    // =========================================================================

    public function testSliceViewChain(): void
    {
        $arr = NDArray::array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ], DType::Int64);

        // Create slice view
        $slice = $arr->slice(['::2', ':']); // Rows 0 and 2
        $this->assertSame([[1, 2, 3, 4], [9, 10, 11, 12]], $slice->toArray());

        // Get view from slice
        $row = $slice->get(1); // [9, 10, 11, 12]
        $this->assertSame([9, 10, 11, 12], $row->toArray());

        // Get element from view of slice
        $elem = $row->get(2); // 11
        $this->assertSame(11, $elem);
    }

    public function testSliceViewSurvivesRootRelease(): void
    {
        $view = (static function () {
            $arr = NDArray::array([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ], DType::Int64);

            $slice = $arr->slice(['::2', ':']);

            return $slice->get(1); // Get row from slice
        })();

        $this->assertSame([7, 8, 9], $view->toArray());
    }

    // =========================================================================
    // Mixed View Operations Tests
    // =========================================================================

    public function testMixedGetAndSliceViews(): void
    {
        $arr = NDArray::array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]],
        ], DType::Int64);

        // Get a 2D view using get()
        $view2d = $arr->get(1); // [[5, 6], [7, 8]]

        // Slice that view
        $slice = $view2d->slice([':', '1:']); // [[6], [8]]
        $this->assertSame([[6], [8]], $slice->toArray());

        // Get from the slice
        $elem = $slice->get(0, 0); // 6
        $this->assertSame(6, $elem);
    }

    public function testArithmeticOnNestedViews(): void
    {
        $arr = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Int64);

        // Get views
        $row1 = $arr->get(0);
        $row2 = $arr->get(1);

        // Perform arithmetic on views
        $result = $row1->add($row2);
        $this->assertSame([5, 7, 9], $result->toArray());

        // Original arrays should be unchanged
        $this->assertSame([1, 2, 3], $row1->toArray());
        $this->assertSame([4, 5, 6], $row2->toArray());
    }

    // =========================================================================
    // Stress Tests
    // =========================================================================

    public function testRepeatedViewCreationAndDestruction(): void
    {
        $arr = NDArray::array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ], DType::Int64);

        // Create and destroy many views in a loop
        for ($i = 0; $i < 100; ++$i) {
            $view = $arr->get($i % 5);
            $this->assertNotNull($view->toArray());
            // View goes out of scope at end of iteration
        }

        // Original array should still be valid
        $this->assertSame([1, 2, 3, 4, 5], $arr->get(0)->toArray());
    }

    public function testNestedViewCreationStress(): void
    {
        $arr = NDArray::array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ], DType::Int64);

        // Create deeply nested views repeatedly
        for ($i = 0; $i < 50; ++$i) {
            $v1 = $arr->get(0);
            $v2 = $v1->get(1);
            $v3 = $v2->get(0);
            $this->assertSame(3, $v3);
        }

        // Array should still be intact
        $this->assertSame([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], $arr->toArray());
    }

    public function testMultipleViewChainsStress(): void
    {
        $arr = NDArray::array([
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
            [[13, 14, 15], [16, 17, 18]],
        ], DType::Int64);

        $views = [];

        // Create many view chains
        for ($i = 0; $i < 20; ++$i) {
            $branch = $arr->get($i % 3);
            $leaf = $branch->get($i % 2);
            $views[] = $leaf;
        }

        // Verify all views are still valid
        for ($i = 0; $i < 20; ++$i) {
            $this->assertNotNull($views[$i]->toArray());
        }

        // Release references gradually
        for ($i = 0; $i < 10; ++$i) {
            unset($views[$i]);
        }

        // Remaining views should still work
        for ($i = 10; $i < 20; ++$i) {
            $this->assertNotNull($views[$i]->toArray());
        }
    }

    public function testViewOperationsInLoop(): void
    {
        $arr = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Int64);

        $results = [];

        // Perform operations on views in a loop
        for ($i = 0; $i < 3; ++$i) {
            $row = $arr->get($i);
            $doubled = $row->multiply(2);
            $results[] = $doubled->sum();
        }

        $this->assertSame([12, 30, 48], $results);
    }

    // =========================================================================
    // Memory Mutation Through Views Tests
    // =========================================================================

    public function testMutationThroughNestedViews(): void
    {
        $arr = NDArray::array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ], DType::Int64);

        // Navigate to deeply nested element
        $branch = $arr->get(1);
        $leaf = $branch->get(0);

        // Mutate through nested view
        $leaf[0] = 999;

        // Verify mutation propagated to root
        $this->assertSame(999, $arr->get(1, 0, 0));
        $this->assertSame([[999, 6], [7, 8]], $branch->toArray());
    }

    public function testMutationThroughSliceViews(): void
    {
        $arr = NDArray::array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ], DType::Int64);

        // Get slice view
        $slice = $arr->slice(['1:', ':']); // Last two rows

        // Get view from slice
        $row = $slice->get(0);

        // Mutate
        $row[1] = 999;

        // Verify
        $this->assertSame(999, $arr->get(1, 1));
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    public function testEmptyViewChain(): void
    {
        $arr = NDArray::array([[[1]]], DType::Int64);

        $v1 = $arr->get(0);
        $v2 = $v1->get(0);
        $scalar = $v2->get(0);

        $this->assertSame(1, $scalar);
    }

    public function testViewOfScalarArray(): void
    {
        $arr = NDArray::array([42], DType::Int64);
        $view = $arr->get(0);

        $this->assertSame(42, $view);
    }

    public function testLargeArrayViewPerformance(): void
    {
        // Create a large array
        $data = [];
        for ($i = 0; $i < 100; ++$i) {
            $row = [];
            for ($j = 0; $j < 100; ++$j) {
                $row[] = $i * 100 + $j;
            }
            $data[] = $row;
        }

        $arr = NDArray::array($data, DType::Int64);

        // Create views at different depths
        $row50 = $arr->get(50);
        $this->assertSame(5050, $row50->get(50));

        $slice = $arr->slice(['50:60', '50:60']);
        $this->assertSame(5050, $slice->get(0, 0));
    }
}
