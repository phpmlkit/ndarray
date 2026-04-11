<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Unit;

use FFI\CData;
use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\Exceptions\ShapeException;
use PhpMlKit\NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Tests for buffer interop operations (toBuffer, fromBytes).
 *
 * Tests FFI buffer allocation, byte string conversion, and data integrity.
 *
 * @internal
 *
 * @coversNothing
 */
final class BufferInteropTest extends TestCase
{
    // =========================================================================
    // toBuffer() Tests
    // =========================================================================

    public function testToBufferAllocatesNewBuffer(): void
    {
        $a = NDArray::array([1.0, 2.0, 3.0, 4.0, 5.0], DType::Float64);

        $buffer = $a->toBuffer();

        $this->assertInstanceOf(CData::class, $buffer);

        // Verify buffer contents
        $this->assertEqualsWithDelta(1.0, $buffer[0], 0.0001);
        $this->assertEqualsWithDelta(2.0, $buffer[1], 0.0001);
        $this->assertEqualsWithDelta(3.0, $buffer[2], 0.0001);
        $this->assertEqualsWithDelta(4.0, $buffer[3], 0.0001);
        $this->assertEqualsWithDelta(5.0, $buffer[4], 0.0001);
    }

    public function testToBufferUsesProvidedBuffer(): void
    {
        $a = NDArray::array([10, 20, 30, 40], DType::Int32);
        $ffi = \FFI::cdef();
        $buffer = $ffi->new('int32_t[4]');

        $result = $a->toBuffer($buffer);

        $this->assertSame($buffer, $result);
        $this->assertSame(10, $buffer[0]);
        $this->assertSame(20, $buffer[1]);
        $this->assertSame(30, $buffer[2]);
        $this->assertSame(40, $buffer[3]);
    }

    public function testToBufferWithStartParameter(): void
    {
        $a = NDArray::array([1.0, 2.0, 3.0, 4.0, 5.0], DType::Float64);

        $buffer = $a->toBuffer(null, 2);

        // Should contain elements from index 2 onwards
        $this->assertEqualsWithDelta(3.0, $buffer[0], 0.0001);
        $this->assertEqualsWithDelta(4.0, $buffer[1], 0.0001);
        $this->assertEqualsWithDelta(5.0, $buffer[2], 0.0001);
    }

    public function testToBufferWithStartAndLenParameters(): void
    {
        $a = NDArray::array([1.0, 2.0, 3.0, 4.0, 5.0], DType::Float64);

        $buffer = $a->toBuffer(null, 1, 2);

        // Should contain elements at indices 1 and 2
        $this->assertEqualsWithDelta(2.0, $buffer[0], 0.0001);
        $this->assertEqualsWithDelta(3.0, $buffer[1], 0.0001);
    }

    public function testToBufferWithLenParameter(): void
    {
        $a = NDArray::array([1.0, 2.0, 3.0, 4.0, 5.0], DType::Float64);

        $buffer = $a->toBuffer(null, 0, 3);

        // Should contain first 3 elements
        $this->assertEqualsWithDelta(1.0, $buffer[0], 0.0001);
        $this->assertEqualsWithDelta(2.0, $buffer[1], 0.0001);
        $this->assertEqualsWithDelta(3.0, $buffer[2], 0.0001);
    }

    public function testToBufferWithProvidedBufferAndStart(): void
    {
        $a = NDArray::array([10, 20, 30, 40], DType::Int32);
        $ffi = \FFI::cdef();
        $buffer = $ffi->new('int32_t[2]');

        $result = $a->toBuffer($buffer, 2);

        // Should write to indices 2 and 3 into the buffer
        $this->assertSame($buffer, $result);
        $this->assertSame(30, $buffer[0]);
        $this->assertSame(40, $buffer[1]);
    }

    public function testToBufferReturnsEmptyBufferForZeroLen(): void
    {
        $a = NDArray::array([1.0, 2.0, 3.0], DType::Float64);

        $buffer = $a->toBuffer(null, 0, 0);

        $this->assertInstanceOf(CData::class, $buffer);
    }

    public function testToBufferReturnsEmptyBufferForNegativeLen(): void
    {
        $a = NDArray::array([1.0, 2.0, 3.0], DType::Float64);

        $buffer = $a->toBuffer(null, 0, -1);

        $this->assertInstanceOf(CData::class, $buffer);
    }

    public function testToBufferWithStartBeyondSize(): void
    {
        $a = NDArray::array([1.0, 2.0, 3.0], DType::Float64);

        $buffer = $a->toBuffer(null, 10);

        // Should return an empty/minimal buffer since start > size
        $this->assertInstanceOf(CData::class, $buffer);
    }

    public function testToBufferWith2DArray(): void
    {
        $a = NDArray::array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ], DType::Float64);

        $buffer = $a->toBuffer();

        // 2D array should be flattened in C-order (row-major)
        $this->assertEqualsWithDelta(1.0, $buffer[0], 0.0001);
        $this->assertEqualsWithDelta(2.0, $buffer[1], 0.0001);
        $this->assertEqualsWithDelta(3.0, $buffer[2], 0.0001);
        $this->assertEqualsWithDelta(4.0, $buffer[3], 0.0001);
        $this->assertEqualsWithDelta(5.0, $buffer[4], 0.0001);
        $this->assertEqualsWithDelta(6.0, $buffer[5], 0.0001);
    }

    public function testToBufferWith3DArray(): void
    {
        $a = NDArray::array([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ], DType::Float64);

        $buffer = $a->toBuffer();

        // 3D array should be flattened in C-order
        $expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for ($i = 0; $i < 8; ++$i) {
            $this->assertEqualsWithDelta($expected[$i], $buffer[$i], 0.0001, "Failed at index {$i}");
        }
    }

    public function testToBufferWithView(): void
    {
        $a = NDArray::array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], DType::Int32);

        $view = $a->slice(['1:3', '1:3']); // [[5,6],[8,9]]

        $buffer = $view->toBuffer();

        // View should be converted to contiguous data
        $this->assertSame(5, $buffer[0]);
        $this->assertSame(6, $buffer[1]);
        $this->assertSame(8, $buffer[2]);
        $this->assertSame(9, $buffer[3]);
    }

    public function testToBufferFloat32(): void
    {
        $a = NDArray::array([1.5, 2.5, 3.5], DType::Float32);

        $buffer = $a->toBuffer();

        $this->assertEqualsWithDelta(1.5, $buffer[0], 0.0001);
        $this->assertEqualsWithDelta(2.5, $buffer[1], 0.0001);
        $this->assertEqualsWithDelta(3.5, $buffer[2], 0.0001);
    }

    public function testToBufferInt8(): void
    {
        $a = NDArray::array([1, -2, 3], DType::Int8);

        $buffer = $a->toBuffer();

        $this->assertSame(1, $buffer[0]);
        $this->assertSame(-2, $buffer[1]);
        $this->assertSame(3, $buffer[2]);
    }

    public function testToBufferInt16(): void
    {
        $a = NDArray::array([100, -200, 300], DType::Int16);

        $buffer = $a->toBuffer();

        $this->assertSame(100, $buffer[0]);
        $this->assertSame(-200, $buffer[1]);
        $this->assertSame(300, $buffer[2]);
    }

    public function testToBufferInt64(): void
    {
        $a = NDArray::array([10000, -20000, 30000], DType::Int64);

        $buffer = $a->toBuffer();

        $this->assertSame(10000, $buffer[0]);
        $this->assertSame(-20000, $buffer[1]);
        $this->assertSame(30000, $buffer[2]);
    }

    public function testToBufferUInt8(): void
    {
        $a = NDArray::array([0, 128, 255], DType::UInt8);

        $buffer = $a->toBuffer();

        $this->assertSame(0, $buffer[0]);
        $this->assertSame(128, $buffer[1]);
        $this->assertSame(255, $buffer[2]);
    }

    public function testToBufferUInt32(): void
    {
        $a = NDArray::array([1000, 2000, 3000], DType::UInt32);

        $buffer = $a->toBuffer();

        $this->assertSame(1000, $buffer[0]);
        $this->assertSame(2000, $buffer[1]);
        $this->assertSame(3000, $buffer[2]);
    }

    public function testToBufferBool(): void
    {
        $a = NDArray::array([true, false, true], DType::Bool);

        $buffer = $a->toBuffer();

        $this->assertSame(1, $buffer[0]);
        $this->assertSame(0, $buffer[1]);
        $this->assertSame(1, $buffer[2]);
    }

    public function testToBufferEmptyArray(): void
    {
        $a = NDArray::empty([0], DType::Float64);

        $buffer = $a->toBuffer();

        $this->assertInstanceOf(CData::class, $buffer);
    }

    // =========================================================================
    // fromBytes() Tests
    // =========================================================================

    public function testFromBytesFloat64(): void
    {
        // Pack 4 doubles: 1.0, 2.0, 3.0, 4.0
        $bytes = pack('d*', 1.0, 2.0, 3.0, 4.0);

        $arr = NDArray::fromBytes($bytes, [2, 2], DType::Float64);

        $this->assertSame([2, 2], $arr->shape());
        $this->assertSame(DType::Float64, $arr->dtype());
        $this->assertEqualsWithDelta([
            [1.0, 2.0],
            [3.0, 4.0],
        ], $arr->toArray(), 0.0001);
    }

    public function testFromBytesFloat32(): void
    {
        // Pack 6 floats: 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
        $bytes = pack('f*', 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);

        $arr = NDArray::fromBytes($bytes, [2, 3], DType::Float32);

        $this->assertSame([2, 3], $arr->shape());
        $this->assertSame(DType::Float32, $arr->dtype());
        $this->assertEqualsWithDelta([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ], $arr->toArray(), 0.0001);
    }

    public function testFromBytesInt32(): void
    {
        // Pack 4 int32 values: 10, 20, 30, 40
        $bytes = pack('l*', 10, 20, 30, 40);

        $arr = NDArray::fromBytes($bytes, [2, 2], DType::Int32);

        $this->assertSame([2, 2], $arr->shape());
        $this->assertSame(DType::Int32, $arr->dtype());
        $this->assertSame([
            [10, 20],
            [30, 40],
        ], $arr->toArray());
    }

    public function testFromBytesInt64(): void
    {
        // Pack 3 int64 values: 100, 200, 300
        $bytes = pack('q*', 100, 200, 300);

        $arr = NDArray::fromBytes($bytes, [3], DType::Int64);

        $this->assertSame([3], $arr->shape());
        $this->assertSame(DType::Int64, $arr->dtype());
        $this->assertSame([100, 200, 300], $arr->toArray());
    }

    public function testFromBytesInt8(): void
    {
        // Pack 4 int8 values: 1, -2, 3, -4
        $bytes = pack('c*', 1, -2, 3, -4);

        $arr = NDArray::fromBytes($bytes, [2, 2], DType::Int8);

        $this->assertSame([2, 2], $arr->shape());
        $this->assertSame(DType::Int8, $arr->dtype());
        $this->assertSame([
            [1, -2],
            [3, -4],
        ], $arr->toArray());
    }

    public function testFromBytesInt16(): void
    {
        // Pack 4 int16 values
        $bytes = pack('s*', 1000, -2000, 3000, -4000);

        $arr = NDArray::fromBytes($bytes, [4], DType::Int16);

        $this->assertSame([4], $arr->shape());
        $this->assertSame(DType::Int16, $arr->dtype());
        $this->assertSame([1000, -2000, 3000, -4000], $arr->toArray());
    }

    public function testFromBytesUInt8(): void
    {
        // Raw bytes: 0x01, 0x02, 0xff
        $bytes = "\x01\x02\xff";

        $arr = NDArray::fromBytes($bytes, [3], DType::UInt8);

        $this->assertSame([3], $arr->shape());
        $this->assertSame(DType::UInt8, $arr->dtype());
        $this->assertSame([1, 2, 255], $arr->toArray());
    }

    public function testFromBytesUInt16(): void
    {
        // Pack 3 uint16 values
        $bytes = pack('S*', 100, 200, 300);

        $arr = NDArray::fromBytes($bytes, [3], DType::UInt16);

        $this->assertSame([3], $arr->shape());
        $this->assertSame(DType::UInt16, $arr->dtype());
        $this->assertSame([100, 200, 300], $arr->toArray());
    }

    public function testFromBytesUInt32(): void
    {
        // Pack 3 uint32 values
        $bytes = pack('L*', 1000, 2000, 3000);

        $arr = NDArray::fromBytes($bytes, [3], DType::UInt32);

        $this->assertSame([3], $arr->shape());
        $this->assertSame(DType::UInt32, $arr->dtype());
        $this->assertSame([1000, 2000, 3000], $arr->toArray());
    }

    public function testFromBytesUInt64(): void
    {
        // Pack 2 uint64 values
        $bytes = pack('Q*', 10000, 20000);

        $arr = NDArray::fromBytes($bytes, [2], DType::UInt64);

        $this->assertSame([2], $arr->shape());
        $this->assertSame(DType::UInt64, $arr->dtype());
        $this->assertSame([10000, 20000], $arr->toArray());
    }

    public function testFromBytesBool(): void
    {
        // Raw bytes: 0x01 (true), 0x00 (false), 0x01 (true)
        $bytes = "\x01\x00\x01";

        $arr = NDArray::fromBytes($bytes, [3], DType::Bool);

        $this->assertSame([3], $arr->shape());
        $this->assertSame(DType::Bool, $arr->dtype());
        $this->assertSame([true, false, true], $arr->toArray());
    }

    public function testFromBytes1D(): void
    {
        $bytes = pack('f*', 1.0, 2.0, 3.0, 4.0, 5.0);

        $arr = NDArray::fromBytes($bytes, [5], DType::Float32);

        $this->assertSame([5], $arr->shape());
        $this->assertEqualsWithDelta([1.0, 2.0, 3.0, 4.0, 5.0], $arr->toArray(), 0.0001);
    }

    public function testFromBytes3D(): void
    {
        // Pack 8 floats for a 2x2x2 array
        $bytes = pack('f*', 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);

        $arr = NDArray::fromBytes($bytes, [2, 2, 2], DType::Float32);

        $this->assertSame([2, 2, 2], $arr->shape());
        $this->assertEqualsWithDelta([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ], $arr->toArray(), 0.0001);
    }

    public function testFromBytesInvalidSizeThrows(): void
    {
        $this->expectException(ShapeException::class);
        $this->expectExceptionMessage('Byte string length');

        // 4 bytes for Float32, but shape expects 3 elements (12 bytes)
        $bytes = pack('f*', 1.0, 2.0, 3.0, 4.0);

        NDArray::fromBytes($bytes, [3], DType::Float32); // Expects 12 bytes, got 16
    }

    public function testFromBytesTooShortThrows(): void
    {
        $this->expectException(ShapeException::class);
        $this->expectExceptionMessage('Byte string length');

        // Only 8 bytes for 2 Float64, but shape expects 3 elements (24 bytes)
        $bytes = pack('d*', 1.0, 2.0);

        NDArray::fromBytes($bytes, [3], DType::Float64); // Expects 24 bytes, got 16
    }

    public function testFromBytesTooLongThrows(): void
    {
        $this->expectException(ShapeException::class);
        $this->expectExceptionMessage('Byte string length');

        // 24 bytes for 3 Float64, but shape expects 2 elements (16 bytes)
        $bytes = pack('d*', 1.0, 2.0, 3.0);

        NDArray::fromBytes($bytes, [2], DType::Float64); // Expects 16 bytes, got 24
    }

    public function testFromBytesEmptyShapeThrows(): void
    {
        $this->expectException(ShapeException::class);
        $this->expectExceptionMessage('Shape must have positive size');

        $bytes = pack('f*', 1.0, 2.0);

        NDArray::fromBytes($bytes, [0], DType::Float32);
    }

    public function testFromBytesSingleElement(): void
    {
        $bytes = pack('d*', 3.14);

        $arr = NDArray::fromBytes($bytes, [1], DType::Float64);

        $this->assertSame([1], $arr->shape());
        $this->assertEqualsWithDelta([3.14], $arr->toArray(), 0.0001);
    }

    public function testFromBytesLargeShape(): void
    {
        // Create a 10x10 array = 100 float64 values
        $values = [];
        for ($i = 0; $i < 100; ++$i) {
            $values[] = (float) $i;
        }
        $bytes = pack('d*', ...$values);

        $arr = NDArray::fromBytes($bytes, [10, 10], DType::Float64);

        $this->assertSame([10, 10], $arr->shape());
        $this->assertSame(100, $arr->size());

        // Verify first and last elements
        $array = $arr->toArray();
        $this->assertEqualsWithDelta(0.0, $array[0][0], 0.0001);
        $this->assertEqualsWithDelta(99.0, $array[9][9], 0.0001);
    }

    public function testFromBytesRoundTripWithToBytes(): void
    {
        // Create an array, convert to bytes, then back
        $original = NDArray::array([
            [1.5, 2.5, 3.5],
            [4.5, 5.5, 6.5],
        ], DType::Float64);

        $bytes = $original->toBytes();
        $reconstructed = NDArray::fromBytes($bytes, [2, 3], DType::Float64);

        $this->assertEqualsWithDelta($original->toArray(), $reconstructed->toArray(), 0.0001);
    }

    public function testFromBytesRoundTripInt32(): void
    {
        $original = NDArray::array([100, 200, 300, 400], DType::Int32);

        $bytes = $original->toBytes();
        $reconstructed = NDArray::fromBytes($bytes, [4], DType::Int32);

        $this->assertSame($original->toArray(), $reconstructed->toArray());
    }

    public function testFromBytesDataIndependence(): void
    {
        // Ensure modifying original bytes doesn't affect array
        $bytes = pack('f*', 1.0, 2.0, 3.0, 4.0);
        $originalBytes = $bytes;

        $arr = NDArray::fromBytes($bytes, [2, 2], DType::Float32);

        // Modify the original bytes
        $bytes = pack('f*', 99.0, 99.0, 99.0, 99.0);

        // Array should still have original values
        $this->assertEqualsWithDelta([
            [1.0, 2.0],
            [3.0, 4.0],
        ], $arr->toArray(), 0.0001);
    }
}
