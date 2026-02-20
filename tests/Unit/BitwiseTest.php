<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Unit;

use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\Exceptions\ShapeException;
use PhpMlKit\NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Tests for bitwise operations (AND, OR, XOR, shifts).
 */
final class BitwiseTest extends TestCase
{
    // ========================================================================
    // Bitwise AND (&)
    // ========================================================================

    public function testBitandIntArrays(): void
    {
        $a = NDArray::array([0b1100, 0b1010], DType::Int32);
        $b = NDArray::array([0b1010, 0b1100], DType::Int32);
        $result = $a->bitand($b);

        $this->assertSame([8, 8], $result->toArray()); // 0b1000 = 8
    }

    public function testBitandUintArrays(): void
    {
        $a = NDArray::array([0xFF, 0xAA], DType::Uint32);
        $b = NDArray::array([0x0F, 0x55], DType::Uint32);
        $result = $a->bitand($b);

        $this->assertSame([0x0F, 0x00], $result->toArray());
    }

    public function testBitandBoolArrays(): void
    {
        $a = NDArray::array([true, false, true], DType::Bool);
        $b = NDArray::array([true, true, false], DType::Bool);
        $result = $a->bitand($b);

        $this->assertSame([true, false, false], $result->toArray());
    }

    public function testBitandScalar(): void
    {
        $a = NDArray::array([0b1100, 0b1010], DType::Int32);
        $result = $a->bitand(0b1010);

        $this->assertSame([8, 10], $result->toArray());
    }

    public function testBitandFloatShouldError(): void
    {
        $this->expectException(\Exception::class);
        $this->expectExceptionMessage('not supported for float');

        $a = NDArray::array([1.0, 2.0], DType::Float64);
        $b = NDArray::array([3.0, 4.0], DType::Float64);
        $result = $a->bitand($b);
    }

    // ========================================================================
    // Bitwise OR (|)
    // ========================================================================

    public function testBitorIntArrays(): void
    {
        $a = NDArray::array([0b1100, 0b1010], DType::Int32);
        $b = NDArray::array([0b1010, 0b1100], DType::Int32);
        $result = $a->bitor($b);

        $this->assertSame([14, 14], $result->toArray()); // 0b1110 = 14
    }

    public function testBitorBoolArrays(): void
    {
        $a = NDArray::array([true, false, true], DType::Bool);
        $b = NDArray::array([true, true, false], DType::Bool);
        $result = $a->bitor($b);

        $this->assertSame([true, true, true], $result->toArray());
    }

    public function testBitorScalar(): void
    {
        $a = NDArray::array([0b1100, 0b1010], DType::Int32);
        $result = $a->bitor(0b0001);

        $this->assertSame([13, 11], $result->toArray());
    }

    public function testBitorFloatShouldError(): void
    {
        $this->expectException(\Exception::class);
        $this->expectExceptionMessage('not supported for float');

        $a = NDArray::array([1.0, 2.0], DType::Float64);
        $result = $a->bitor(3);
    }

    // ========================================================================
    // Bitwise XOR (^)
    // ========================================================================

    public function testBitxorIntArrays(): void
    {
        $a = NDArray::array([0b1100, 0b1010], DType::Int32);
        $b = NDArray::array([0b1010, 0b1100], DType::Int32);
        $result = $a->bitxor($b);

        $this->assertSame([6, 6], $result->toArray()); // 0b0110 = 6
    }

    public function testBitxorBoolArrays(): void
    {
        $a = NDArray::array([true, false, true], DType::Bool);
        $b = NDArray::array([true, true, false], DType::Bool);
        $result = $a->bitxor($b);

        $this->assertSame([false, true, true], $result->toArray());
    }

    public function testBitxorScalar(): void
    {
        $a = NDArray::array([0b1100, 0b1010], DType::Int32);
        $result = $a->bitxor(0b1111);

        $this->assertSame([3, 5], $result->toArray());
    }

    public function testBitxorFloatShouldError(): void
    {
        $this->expectException(\Exception::class);
        $this->expectExceptionMessage('not supported for float');

        $a = NDArray::array([1.0, 2.0], DType::Float32);
        $result = $a->bitxor(3);
    }

    // ========================================================================
    // Left Shift (<<)
    // ========================================================================

    public function testLeftShiftIntArrays(): void
    {
        $a = NDArray::array([1, 2, 4], DType::Int32);
        $b = NDArray::array([1, 2, 3], DType::Int32);
        $result = $a->leftShift($b);

        $this->assertSame([2, 8, 32], $result->toArray());
    }

    public function testLeftShiftScalar(): void
    {
        $a = NDArray::array([1, 2, 4], DType::Int32);
        $result = $a->leftShift(2);

        $this->assertSame([4, 8, 16], $result->toArray());
    }

    public function testLeftShiftByZero(): void
    {
        $a = NDArray::array([1, 2, 4], DType::Int32);
        $result = $a->leftShift(0);

        $this->assertSame([1, 2, 4], $result->toArray());
    }

    public function testLeftShiftLarge(): void
    {
        $a = NDArray::array([1], DType::Int32);
        $result = $a->leftShift(31);

        $this->assertSame([-2147483648], $result->toArray());
    }

    public function testLeftShiftFloatShouldError(): void
    {
        $this->expectException(\Exception::class);
        $this->expectExceptionMessage('only supported for integer');

        $a = NDArray::array([1.0, 2.0], DType::Float64);
        $result = $a->leftShift(1);
    }

    public function testLeftShiftBoolShouldError(): void
    {
        $this->expectException(\Exception::class);
        $this->expectExceptionMessage('only supported for integer');

        $a = NDArray::array([true, false], DType::Bool);
        $result = $a->leftShift(1);
    }

    // ========================================================================
    // Right Shift (>>)
    // ========================================================================

    public function testRightShiftIntArrays(): void
    {
        $a = NDArray::array([8, 16, 32], DType::Int32);
        $b = NDArray::array([1, 2, 3], DType::Int32);
        $result = $a->rightShift($b);

        $this->assertSame([4, 4, 4], $result->toArray());
    }

    public function testRightShiftScalar(): void
    {
        $a = NDArray::array([32, 16, 8], DType::Int32);
        $result = $a->rightShift(2);

        $this->assertSame([8, 4, 2], $result->toArray());
    }

    public function testRightShiftSigned(): void
    {
        $a = NDArray::array([-8, -16], DType::Int32);
        $result = $a->rightShift(2);

        // Arithmetic right shift preserves sign
        $this->assertSame([-2, -4], $result->toArray());
    }

    public function testRightShiftUnsigned(): void
    {
        $a = NDArray::array([0x80000000, 0x40000000], DType::Uint32);
        $result = $a->rightShift(1);

        $this->assertSame([0x40000000, 0x20000000], $result->toArray());
    }

    public function testRightShiftFloatShouldError(): void
    {
        $this->expectException(\Exception::class);
        $this->expectExceptionMessage('only supported for integer');

        $a = NDArray::array([8.0, 16.0], DType::Float32);
        $result = $a->rightShift(1);
    }

    public function testRightShiftBoolShouldError(): void
    {
        $this->expectException(\Exception::class);
        $this->expectExceptionMessage('only supported for integer');

        $a = NDArray::array([true, false], DType::Bool);
        $result = $a->rightShift(1);
    }

    // ========================================================================
    // Mixed Type Operations
    // ========================================================================

    public function testBitandMixedIntTypes(): void
    {
        $a = NDArray::array([0b1100, 0b1010], DType::Int32);
        $b = NDArray::array([0b1010, 0b1100], DType::Int64);
        $result = $a->bitand($b);

        $this->assertSame(DType::Int64, $result->dtype());
        $this->assertSame([8, 8], $result->toArray());
    }

    public function testLeftShiftMixedIntTypes(): void
    {
        $a = NDArray::array([1, 2, 4], DType::Int32);
        $b = NDArray::array([1, 2, 3], DType::Int16);
        $result = $a->leftShift($b);

        $this->assertSame(DType::Int32, $result->dtype());
        $this->assertSame([2, 8, 32], $result->toArray());
    }

    // ========================================================================
    // All Integer Types
    // ========================================================================

    public function testBitwiseAllIntTypes(): void
    {
        $types = [
            DType::Int8,
            DType::Int16,
            DType::Int32,
            DType::Int64,
            DType::Uint8,
            DType::Uint16,
            DType::Uint32,
            DType::Uint64,
        ];

        foreach ($types as $dtype) {
            $a = NDArray::array([0b1100, 0b1010], $dtype);
            $b = NDArray::array([0b1010, 0b1100], $dtype);
            
            $result = $a->bitand($b);
            $this->assertSame($dtype, $result->dtype(), "bitand failed for {$dtype->name}");
            $this->assertSame([8, 8], $result->toArray());

            $result = $a->bitor($b);
            $this->assertSame($dtype, $result->dtype(), "bitor failed for {$dtype->name}");
            $this->assertSame([14, 14], $result->toArray());

            $result = $a->bitxor($b);
            $this->assertSame($dtype, $result->dtype(), "bitxor failed for {$dtype->name}");
            $this->assertSame([6, 6], $result->toArray());
        }
    }

    // ========================================================================
    // Broadcasting
    // ========================================================================

    public function testBitandBroadcasting(): void
    {
        $a = NDArray::array([[0b1100, 0b1010], [0b1111, 0b0000]], DType::Int32);
        $b = NDArray::array([0b1010], DType::Int32);
        $result = $a->bitand($b);

        $this->assertSame([2, 2], $result->shape());
        $this->assertSame([[8, 10], [10, 0]], $result->toArray());
    }

    public function testLeftShiftBroadcasting(): void
    {
        $a = NDArray::array([[1, 2], [4, 8]], DType::Int32);
        $b = NDArray::array([1], DType::Int32);
        $result = $a->leftShift($b);

        $this->assertSame([[2, 4], [8, 16]], $result->toArray());
    }

    // ========================================================================
    // Edge Cases
    // ========================================================================

    public function testBitwiseWithAllZeros(): void
    {
        $a = NDArray::array([0, 0, 0], DType::Int32);
        $b = NDArray::array([0, 0, 0], DType::Int32);

        $this->assertSame([0, 0, 0], $a->bitand($b)->toArray());
        $this->assertSame([0, 0, 0], $a->bitor($b)->toArray());
        $this->assertSame([0, 0, 0], $a->bitxor($b)->toArray());
    }

    public function testBitwiseWithAllOnes(): void
    {
        $a = NDArray::array([0xFF, 0xFF], DType::Uint8);
        $b = NDArray::array([0xFF, 0x00], DType::Uint8);

        $this->assertSame([0xFF, 0x00], $a->bitand($b)->toArray());
        $this->assertSame([0xFF, 0xFF], $a->bitor($b)->toArray());
        $this->assertSame([0x00, 0xFF], $a->bitxor($b)->toArray());
    }

    public function testBitwiseComplement(): void
    {
        $a = NDArray::array([0b11110000], DType::Uint8);
        $b = NDArray::array([0b00001111], DType::Uint8);

        $this->assertSame([0], $a->bitand($b)->toArray());
        $this->assertSame([255], $a->bitor($b)->toArray());
        $this->assertSame([255], $a->bitxor($b)->toArray());
    }

    public function testShiftByLargeAmount(): void
    {
        $a = NDArray::array([1, 2, 4], DType::Int32);
        $result = $a->leftShift(30);

        // Will overflow but should not panic
        $this->assertIsArray($result->toArray());
    }

    // ========================================================================
    // Incompatible Shapes
    // ========================================================================

    public function testBitandIncompatibleShapes(): void
    {
        $this->expectException(ShapeException::class);

        $a = NDArray::array([1, 2, 3], DType::Int32);
        $b = NDArray::array([1, 2], DType::Int32);
        $result = $a->bitand($b);
    }

    public function testLeftShiftIncompatibleShapes(): void
    {
        $this->expectException(ShapeException::class);

        $a = NDArray::array([1, 2, 3], DType::Int32);
        $b = NDArray::array([1, 2], DType::Int32);
        $result = $a->leftShift($b);
    }

    // ========================================================================
    // Empty Arrays
    // ========================================================================

    public function testBitwiseEmptyArray(): void
    {
        $a = NDArray::zeros([0], DType::Int32);
        $b = NDArray::zeros([0], DType::Int32);

        $this->assertSame([], $a->bitand($b)->toArray());
        $this->assertSame([], $a->bitor($b)->toArray());
        $this->assertSame([], $a->bitxor($b)->toArray());
    }

    public function testShiftEmptyArray(): void
    {
        $a = NDArray::zeros([0], DType::Int32);
        $result = $a->leftShift(1);

        $this->assertSame([], $result->toArray());
    }

    // ========================================================================
    // Zero-Dimensional Arrays
    // ========================================================================

    public function testBitwiseZeroDim(): void
    {
        $a = NDArray::array([0b1100], DType::Int32)->reshape([]);
        $b = NDArray::array([0b1010], DType::Int32)->reshape([]);

        $result = $a->bitand($b);
        $this->assertSame([], $result->shape());
        $this->assertSame(8, $result->toArray());
    }

    public function testShiftZeroDim(): void
    {
        $a = NDArray::array([8], DType::Int32)->reshape([]);
        $result = $a->rightShift(2);

        $this->assertSame(2, $result->toArray());
    }
}
