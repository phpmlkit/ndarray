<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Unit;

use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Tests for 3D array slicing and assignment.
 *
 * @internal
 *
 * @coversNothing
 */
final class Slicing3DTest extends TestCase
{
    public function testSlice3DBasic(): void
    {
        // Shape [2, 2, 2] = 8 elements
        // [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
        $arr = NDArray::arange(8)->reshape([2, 2, 2]);

        // Slice first plane: arr[0]
        $slice = $arr->slice([0]);
        $this->assertSame([2, 2], $slice->shape());
        $this->assertSame([[0, 1], [2, 3]], $slice->toArray());

        // Slice along last axis: arr[:, :, 0]
        $slice = $arr->slice([':', ':', 0]);
        $this->assertSame([2, 2], $slice->shape());
        $this->assertSame([[0, 2], [4, 6]], $slice->toArray());
    }

    public function testSlice3DStrided(): void
    {
        // Shape [3, 3, 3] = 27 elements
        $arr = NDArray::arange(27)->reshape([3, 3, 3]);

        // Slice with stride 2 on axis 1: arr[:, ::2, :]
        $slice = $arr->slice([':', '::2', ':']);

        // Original axis 1 has size 3. Step 2 selects indices 0, 2. So new size is 2.
        $this->assertSame([3, 2, 3], $slice->shape());

        // Check a value
        // Original [0, 2, 0] is 0*9 + 2*3 + 0 = 6
        // In slice [0, 1, 0] corresponds to original [0, 2, 0] -> 6
        $this->assertEquals(6, $slice->get(0, 1, 0));
    }

    public function testAssignScalarTo3DSlice(): void
    {
        // 3x3x3 zeros
        $arr = NDArray::zeros([3, 3, 3], DType::Int32);

        // Set middle block to 1: arr[1, :, :] = 1
        $arr->slice([1])->assign(1);

        // Check planes
        $data = $arr->toArray();

        // Plane 0: all zeros
        $this->assertEquals(0, $arr->get(0, 0, 0));

        // Plane 1: all ones
        $this->assertEquals(1, $arr->get(1, 0, 0));
        $this->assertEquals(1, $arr->get(1, 2, 2));

        // Plane 2: all zeros
        $this->assertEquals(0, $arr->get(2, 0, 0));
    }

    public function testAssignScalarToStrided3DSlice(): void
    {
        // 4x4x4 zeros
        $arr = NDArray::zeros([4, 4, 4], DType::Int32);

        // Set every other plane to 5: arr[::2] = 5
        $arr->slice(['::2'])->assign(5);

        // Plane 0 should be 5s
        $this->assertEquals(5, $arr->get(0, 0, 0));

        // Plane 1 should be 0s
        $this->assertEquals(0, $arr->get(1, 0, 0));

        // Plane 2 should be 5s
        $this->assertEquals(5, $arr->get(2, 0, 0));

        // Plane 3 should be 0s
        $this->assertEquals(0, $arr->get(3, 0, 0));
    }
}
