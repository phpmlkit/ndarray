<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Unit;

use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Tests for NDArray type casting (astype).
 *
 * @internal
 *
 * @coversNothing
 */
final class AstypeTest extends TestCase
{
    public function testAstypeIntToFloat(): void
    {
        $arr = NDArray::array([1, 2, 3, 4, 5], DType::Int32);

        $this->assertSame(DType::Int32, $arr->dtype());

        $floatArr = $arr->astype(DType::Float64);

        $this->assertSame(DType::Float64, $floatArr->dtype());
        $this->assertSame([1.0, 2.0, 3.0, 4.0, 5.0], $floatArr->toArray());
        $this->assertSame([5], $floatArr->shape());
    }

    public function testAstypeFloatToInt(): void
    {
        $arr = NDArray::array([1.7, 2.3, 3.9], DType::Float64);

        $intArr = $arr->astype(DType::Int32);

        $this->assertSame(DType::Int32, $intArr->dtype());
        // Truncation behavior: 1.7 -> 1, 2.3 -> 2, 3.9 -> 3
        $this->assertSame([1, 2, 3], $intArr->toArray());
    }

    public function testAstypeSameTypeReturnsCopy(): void
    {
        $arr = NDArray::array([1, 2, 3], DType::Int64);

        $copy = $arr->astype(DType::Int64);

        $this->assertSame(DType::Int64, $copy->dtype());
        $this->assertSame([1, 2, 3], $copy->toArray());

        // Should be a copy, not the same object
        $this->assertNotSame($arr, $copy);
    }

    public function testAstype2DArray(): void
    {
        $arr = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
        ], DType::Int32);

        $floatArr = $arr->astype(DType::Float32);

        $this->assertSame(DType::Float32, $floatArr->dtype());
        $this->assertSame([2, 3], $floatArr->shape());
        $this->assertEquals([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ], $floatArr->toArray());
    }

    public function testAstype3DArray(): void
    {
        $arr = NDArray::array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ], DType::Int64);

        $floatArr = $arr->astype(DType::Float64);

        $this->assertSame(DType::Float64, $floatArr->dtype());
        $this->assertSame([2, 2, 2], $floatArr->shape());
    }

    public function testAstypeUnsignedToSigned(): void
    {
        $arr = NDArray::array([0, 100, 255], DType::UInt8);

        $signedArr = $arr->astype(DType::Int16);

        $this->assertSame(DType::Int16, $signedArr->dtype());
        $this->assertSame([0, 100, 255], $signedArr->toArray());
    }

    public function testAstypeSignedToUnsigned(): void
    {
        $arr = NDArray::array([-10, 0, 10], DType::Int16);

        $unsignedArr = $arr->astype(DType::UInt16);

        $this->assertSame(DType::UInt16, $unsignedArr->dtype());
        // Note: Negative values will wrap due to Rust's casting behavior
        // -10 as u16 = 65526
        $result = $unsignedArr->toArray();
        $this->assertSame(65526, $result[0]);
        $this->assertSame(0, $result[1]);
        $this->assertSame(10, $result[2]);
    }

    public function testAstypeToBool(): void
    {
        $arr = NDArray::array([0, 1, 2, -1, 0.0, 3.14], DType::Float64);

        $boolArr = $arr->astype(DType::Bool);

        $this->assertSame(DType::Bool, $boolArr->dtype());
        // 0 and 0.0 become false, everything else becomes true
        $this->assertSame([false, true, true, true, false, true], $boolArr->toArray());
    }

    public function testAstypeFromBool(): void
    {
        $arr = NDArray::array([true, false, true], DType::Bool);

        $intArr = $arr->astype(DType::Int32);

        $this->assertSame(DType::Int32, $intArr->dtype());
        $this->assertSame([1, 0, 1], $intArr->toArray());
    }

    public function testAstypeAllCombinations(): void
    {
        $dtypes = [
            DType::Int8,
            DType::Int16,
            DType::Int32,
            DType::Int64,
            DType::UInt8,
            DType::UInt16,
            DType::UInt32,
            DType::UInt64,
            DType::Float32,
            DType::Float64,
            DType::Bool,
        ];

        $arr = NDArray::array([1, 2, 3], DType::Int32);

        foreach ($dtypes as $targetDtype) {
            $converted = $arr->astype($targetDtype);
            $this->assertSame($targetDtype, $converted->dtype(), "Failed converting Int32 to {$targetDtype->name}");
            $this->assertSame([3], $converted->shape());
        }
    }

    public function testAstypeOnView(): void
    {
        $arr = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
        ], DType::Int32);

        // Get a view (row 1)
        $view = $arr->get(1);
        $this->assertTrue($view->isView());

        // Cast the view
        $floatView = $view->astype(DType::Float64);

        $this->assertFalse($floatView->isView()); // Result is not a view
        $this->assertSame(DType::Float64, $floatView->dtype());
        $this->assertSame([4.0, 5.0, 6.0], $floatView->toArray());
    }

    public function testAstypeOnSlice(): void
    {
        $arr = NDArray::array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], DType::Int64);

        // Get a slice
        $slice = $arr->slice(['2:7']); // [3, 4, 5, 6, 7]
        $this->assertSame([3, 4, 5, 6, 7], $slice->toArray());

        // Cast the slice
        $floatSlice = $slice->astype(DType::Float32);

        $this->assertSame(DType::Float32, $floatSlice->dtype());
        $this->assertSame([3.0, 4.0, 5.0, 6.0, 7.0], $floatSlice->toArray());
    }

    public function testAstypeFloatPrecision(): void
    {
        $arr = NDArray::array([1.123456789, 2.987654321], DType::Float64);

        $float32Arr = $arr->astype(DType::Float32);
        $backTo64 = $float32Arr->astype(DType::Float64);

        $this->assertSame(DType::Float32, $float32Arr->dtype());
        $this->assertSame(DType::Float64, $backTo64->dtype());

        // Values should be close but not exactly equal due to precision loss
        $result = $backTo64->toArray();
        $this->assertEqualsWithDelta(1.123456789, $result[0], 0.0001);
        $this->assertEqualsWithDelta(2.987654321, $result[1], 0.0001);
    }

    public function testAstypeLargeArray(): void
    {
        $data = range(1, 1000);
        $arr = NDArray::array($data, DType::Int32);

        $floatArr = $arr->astype(DType::Float64);

        $this->assertSame(DType::Float64, $floatArr->dtype());
        $this->assertSame([1000], $floatArr->shape());

        // Check a few values
        $result = $floatArr->toArray();
        $this->assertEquals(1.0, $result[0]);
        $this->assertEquals(500.0, $result[499]);
        $this->assertEquals(1000.0, $result[999]);
    }

    public function testAstypeZeroDimensional(): void
    {
        $arr = NDArray::array([42], DType::Int32)->reshape([]);

        $this->assertSame([], $arr->shape());

        $floatArr = $arr->astype(DType::Float64);

        $this->assertSame(DType::Float64, $floatArr->dtype());
        $this->assertSame([], $floatArr->shape());
        $this->assertSame(42.0, $floatArr->toArray());
    }

    public function testAstypeEmptyArray(): void
    {
        $arr = NDArray::zeros([0], DType::Int32);

        $floatArr = $arr->astype(DType::Float64);

        $this->assertSame(DType::Float64, $floatArr->dtype());
        $this->assertSame([], $floatArr->toArray());
    }

    public function testAstypeChained(): void
    {
        $arr = NDArray::array([1, 2, 3], DType::Int32);

        // Chain multiple conversions
        $result = $arr
            ->astype(DType::Float64)
            ->astype(DType::Int16)
            ->astype(DType::Float32)
        ;

        $this->assertSame(DType::Float32, $result->dtype());
        $this->assertEquals([1.0, 2.0, 3.0], $result->toArray());
    }

    public function testAstypeResultIsContiguous(): void
    {
        // Create a strided view
        $arr = NDArray::arange(10)->reshape([2, 5]);
        $strided = $arr->slice([':', '::2']); // Every other column

        $this->assertFalse($strided->isContiguous());

        // Cast result should be contiguous
        $casted = $strided->astype(DType::Float64);
        $this->assertTrue($casted->isContiguous());
    }

    public function testAstypePreservesShape(): void
    {
        $shapes = [
            [5],
            [3, 4],
            [2, 3, 4],
            [2, 2, 2, 2],
        ];

        foreach ($shapes as $shape) {
            $arr = NDArray::zeros($shape, DType::Int32);
            $casted = $arr->astype(DType::Float64);

            $this->assertSame($shape, $casted->shape(), 'Failed for shape '.json_encode($shape));
        }
    }
}
