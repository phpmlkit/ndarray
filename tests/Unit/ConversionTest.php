<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Unit;

use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\Exceptions\ShapeException;
use PhpMlKit\NDArray\FFI\Lib;
use PhpMlKit\NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Tests for conversion operations (toArray, toScalar, copy, astype).
 *
 * @internal
 *
 * @coversNothing
 */
final class ConversionTest extends TestCase
{
    public function testItemsizeAndNbytes(): void
    {
        $a = NDArray::array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], DType::Float64);

        $this->assertSame(8, $a->itemsize());
        $this->assertSame(48, $a->nbytes());
    }

    public function testTobytesUint8(): void
    {
        $a = NDArray::array([1, 2, 255], DType::Uint8);
        $bytes = $a->tobytes();

        $this->assertSame(3, \strlen($bytes));
        $this->assertSame("\x01\x02\xff", $bytes);
    }

    public function testTobytesBool(): void
    {
        $a = NDArray::array([true, false, true], DType::Bool);
        $bytes = $a->tobytes();

        $this->assertSame(3, \strlen($bytes));
        $this->assertSame("\x01\x00\x01", $bytes);
    }

    public function testTobytesFloat64RoundTripWithUnpack(): void
    {
        $a = NDArray::array([1.5, -2.25, 3.75], DType::Float64);
        $bytes = $a->tobytes();

        $this->assertSame(24, \strlen($bytes)); // 3 * 8 bytes

        // Use machine-endian unpack for stable round-trip on this platform.
        $decoded = array_values(unpack('d*', $bytes));

        $this->assertCount(3, $decoded);
        $this->assertEqualsWithDelta(1.5, $decoded[0], 0.0000001);
        $this->assertEqualsWithDelta(-2.25, $decoded[1], 0.0000001);
        $this->assertEqualsWithDelta(3.75, $decoded[2], 0.0000001);
    }

    public function testCopyToBufferTypedInt32(): void
    {
        $a = NDArray::array([10, 20, 30, 40], DType::Int32);
        $ffi = Lib::get();
        $dst = $ffi->new('int32_t[4]');

        $copied = $a->copyToBuffer($dst);

        $this->assertSame(4, $copied);
        $this->assertSame(10, $dst[0]);
        $this->assertSame(20, $dst[1]);
        $this->assertSame(30, $dst[2]);
        $this->assertSame(40, $dst[3]);
    }

    public function testCopyToBufferOnView(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Int32);
        $view = $a->slice([':', '1:3']); // [[2,3],[5,6]]
        $ffi = Lib::get();
        $dst = $ffi->new('int32_t[4]');

        $copied = $view->copyToBuffer($dst);

        $this->assertSame(4, $copied);
        $this->assertSame(2, $dst[0]);
        $this->assertSame(3, $dst[1]);
        $this->assertSame(5, $dst[2]);
        $this->assertSame(6, $dst[3]);
    }

    public function testCopyToBufferTooSmallThrows(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Int32);
        $ffi = Lib::get();
        $dst = $ffi->new('int32_t[2]');

        $this->expectException(ShapeException::class);
        $this->expectExceptionMessage('Destination buffer too small');
        $a->copyToBuffer($dst, 2);
    }

    public function testToArrayForAllDTypes1D(): void
    {
        $cases = [
            [DType::Int8, [1, -2, 3]],
            [DType::Int16, [100, -200, 300]],
            [DType::Int32, [1000, -2000, 3000]],
            [DType::Int64, [10000, -20000, 30000]],
            [DType::Uint8, [1, 2, 3]],
            [DType::Uint16, [100, 200, 300]],
            [DType::Uint32, [1000, 2000, 3000]],
            [DType::Uint64, [10000, 20000, 30000]],
            [DType::Float32, [1.25, -2.5, 3.75]],
            [DType::Float64, [1.25, -2.5, 3.75]],
            [DType::Bool, [true, false, true]],
        ];

        foreach ($cases as [$dtype, $input]) {
            $a = NDArray::array($input, $dtype);
            $out = $a->toArray();

            $this->assertIsArray($out);
            $this->assertCount(\count($input), $out);

            if (DType::Float32 === $dtype || DType::Float64 === $dtype) {
                for ($i = 0; $i < \count($input); ++$i) {
                    $this->assertEqualsWithDelta((float) $input[$i], $out[$i], 0.0001, "Failed for {$dtype->name} at index {$i}");
                }
            } else {
                $this->assertSame($input, $out, "Failed for {$dtype->name}");
            }
        }
    }

    public function testToArrayForBool2D(): void
    {
        $a = NDArray::array([
            [true, false, true],
            [false, true, false],
        ], DType::Bool);

        $out = $a->toArray();

        $this->assertSame([
            [true, false, true],
            [false, true, false],
        ], $out);
    }

    public function testToArrayForViewAllDTypes(): void
    {
        $cases = [
            [DType::Int32, [[1, 2, 3], [4, 5, 6]], [[2, 3], [5, 6]]],
            [DType::Uint16, [[1, 2, 3], [4, 5, 6]], [[2, 3], [5, 6]]],
            [DType::Float64, [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], [[2.5, 3.5], [5.5, 6.5]]],
            [DType::Bool, [[true, false, true], [false, true, false]], [[false, true], [true, false]]],
        ];

        foreach ($cases as [$dtype, $input, $expected]) {
            $a = NDArray::array($input, $dtype);
            $view = $a->slice([':', '1:3']);
            $out = $view->toArray();

            if (DType::Float64 === $dtype) {
                $this->assertEqualsWithDelta($expected, $out, 0.0001, "Failed for {$dtype->name}");
            } else {
                $this->assertSame($expected, $out, "Failed for {$dtype->name}");
            }
        }
    }

    public function testToFlatArray1D(): void
    {
        $a = NDArray::array([1, 2, 3, 4], DType::Int32);

        $flat = $a->toFlatArray();

        $this->assertIsArray($flat);
        $this->assertSame([1, 2, 3, 4], $flat);
    }

    public function testToFlatArrayOnView(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Int32);
        $view = $a->slice([':', '1:3']); // [[2, 3], [5, 6]]

        $flat = $view->toFlatArray();

        $this->assertIsArray($flat);
        $this->assertSame([2, 3, 5, 6], $flat);
    }

    public function testToArrayPreservesNaN(): void
    {
        $a = NDArray::array([1.0, \NAN, 2.0], DType::Float64);

        $out = $a->toArray();

        $this->assertIsArray($out);
        $this->assertEqualsWithDelta(1.0, $out[0], 0.0001);
        $this->assertTrue(is_nan($out[1]));
        $this->assertEqualsWithDelta(2.0, $out[2], 0.0001);
    }

    public function testToArrayNestedFromFlat(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Int32);

        $out = $a->toArray();

        $this->assertSame([[1, 2, 3], [4, 5, 6]], $out);
    }

    public function testToFlatArrayScalarReturnsScalar(): void
    {
        $a = NDArray::full([], 7, DType::Int32);

        $flat = $a->toFlatArray();

        $this->assertIsInt($flat);
        $this->assertSame(7, $flat);
    }

    public function testToScalarFloat64(): void
    {
        $a = NDArray::array([3.14], DType::Float64)->reshape([]);

        $this->assertSame(0, $a->ndim());
        $this->assertSame([], $a->shape());

        $scalar = $a->toScalar();

        $this->assertIsFloat($scalar);
        $this->assertEqualsWithDelta(3.14, $scalar, 0.0001);
    }

    public function testToScalarFloat32(): void
    {
        $a = NDArray::array([2.5], DType::Float32)->reshape([]);

        $scalar = $a->toScalar();

        $this->assertIsFloat($scalar);
        $this->assertEqualsWithDelta(2.5, $scalar, 0.0001);
    }

    public function testToScalarInt64(): void
    {
        $a = NDArray::array([42], DType::Int64)->reshape([]);

        $scalar = $a->toScalar();

        $this->assertIsInt($scalar);
        $this->assertSame(42, $scalar);
    }

    public function testToScalarInt32(): void
    {
        $a = NDArray::array([-100], DType::Int32)->reshape([]);

        $scalar = $a->toScalar();

        $this->assertIsInt($scalar);
        $this->assertSame(-100, $scalar);
    }

    public function testToScalarBool(): void
    {
        $a = NDArray::full([], true, DType::Bool);

        $scalar = $a->toScalar();

        $this->assertIsBool($scalar);
        $this->assertTrue($scalar);
    }

    public function testToScalarBoolFalse(): void
    {
        $a = NDArray::full([], false, DType::Bool);

        $scalar = $a->toScalar();

        $this->assertIsBool($scalar);
        $this->assertFalse($scalar);
    }

    public function testToScalarFromFull(): void
    {
        $a = NDArray::full([], 7.5, DType::Float64);

        $scalar = $a->toScalar();

        $this->assertIsFloat($scalar);
        $this->assertEqualsWithDelta(7.5, $scalar, 0.0001);
    }

    public function testToScalarFromView(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5], DType::Int64);
        $view = $a->slice(['2:3']);  // [3]
        $scalarView = $view->reshape([]);  // 0-d view of element 3

        $scalar = $scalarView->toScalar();

        $this->assertIsInt($scalar);
        $this->assertSame(3, $scalar);
    }

    public function testToScalarNonZeroDimThrows(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);

        $this->expectException(ShapeException::class);
        $this->expectExceptionMessage('toScalar requires a 0-dimensional array');
        $a->toScalar();
    }

    public function testToScalar2DThrows(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);

        $this->expectException(ShapeException::class);
        $this->expectExceptionMessage('toScalar requires a 0-dimensional array');
        $a->toScalar();
    }
}
