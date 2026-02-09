<?php

declare(strict_types=1);

namespace NDArray\Tests\Unit;

use NDArray\DType;
use NDArray\Exceptions\ShapeException;
use NDArray\NDArray;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;

/**
 * Tests for NDArray::array() factory method.
 */
final class NDArrayTest extends TestCase
{
    // =========================================================================
    // Basic Creation Tests
    // =========================================================================

    public function testCreateFrom1DArray(): void
    {
        $arr = NDArray::array([1.0, 2.0, 3.0, 4.0]);

        $this->assertSame([4], $arr->shape());
        $this->assertSame(1, $arr->ndim());
        $this->assertSame(4, $arr->size());
        $this->assertSame(DType::Float64, $arr->dtype());
    }

    public function testCreateFrom2DArray(): void
    {
        $arr = NDArray::array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]);

        $this->assertSame([2, 3], $arr->shape());
        $this->assertSame(2, $arr->ndim());
        $this->assertSame(6, $arr->size());
    }

    public function testCreateFrom3DArray(): void
    {
        $arr = NDArray::array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ]);

        $this->assertSame([2, 2, 2], $arr->shape());
        $this->assertSame(3, $arr->ndim());
        $this->assertSame(8, $arr->size());
    }

    // =========================================================================
    // DType Inference Tests
    // =========================================================================

    public function testInferFloat64FromFloatArray(): void
    {
        $arr = NDArray::array([1.5, 2.5, 3.5]);

        $this->assertSame(DType::Float64, $arr->dtype());
    }

    public function testInferInt64FromIntArray(): void
    {
        $arr = NDArray::array([1, 2, 3]);

        $this->assertSame(DType::Int64, $arr->dtype());
    }

    public function testInferBoolFromBoolArray(): void
    {
        $arr = NDArray::array([true, false, true]);

        $this->assertSame(DType::Bool, $arr->dtype());
    }

    public function testInferFloat64FromMixedIntFloat(): void
    {
        $arr = NDArray::array([1, 2.5, 3]);

        $this->assertSame(DType::Float64, $arr->dtype());
    }

    // =========================================================================
    // Explicit DType Tests
    // =========================================================================

    #[DataProvider('dtypeProvider')]
    public function testCreateWithExplicitDType(DType $dtype, array $data): void
    {
        $arr = NDArray::array($data, $dtype);

        $this->assertSame($dtype, $arr->dtype());
        $this->assertSame($dtype->itemSize(), $arr->itemsize());
    }

    public static function dtypeProvider(): array
    {
        return [
            'Float64' => [DType::Float64, [1.0, 2.0, 3.0]],
            'Float32' => [DType::Float32, [1.0, 2.0, 3.0]],
            'Int64' => [DType::Int64, [1, 2, 3]],
            'Int32' => [DType::Int32, [1, 2, 3]],
            'Int16' => [DType::Int16, [1, 2, 3]],
            'Int8' => [DType::Int8, [1, 2, 3]],
            'Uint64' => [DType::Uint64, [1, 2, 3]],
            'Uint32' => [DType::Uint32, [1, 2, 3]],
            'Uint16' => [DType::Uint16, [1, 2, 3]],
            'Uint8' => [DType::Uint8, [1, 2, 3]],
            'Bool' => [DType::Bool, [true, false, true]],
        ];
    }

    // =========================================================================
    // Properties Tests
    // =========================================================================

    public function testItemsize(): void
    {
        $f64 = NDArray::array([1.0], DType::Float64);
        $f32 = NDArray::array([1.0], DType::Float32);
        $i64 = NDArray::array([1], DType::Int64);
        $i32 = NDArray::array([1], DType::Int32);
        $i8 = NDArray::array([1], DType::Int8);
        $bool = NDArray::array([true], DType::Bool);

        $this->assertSame(8, $f64->itemsize());
        $this->assertSame(4, $f32->itemsize());
        $this->assertSame(8, $i64->itemsize());
        $this->assertSame(4, $i32->itemsize());
        $this->assertSame(1, $i8->itemsize());
        $this->assertSame(1, $bool->itemsize());
    }

    public function testNbytes(): void
    {
        $arr = NDArray::array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], DType::Float64);

        // 6 elements * 8 bytes = 48 bytes
        $this->assertSame(48, $arr->nbytes());
    }

    // =========================================================================
    // toArray Tests
    // =========================================================================

    public function testToArray1D(): void
    {
        $data = [1.0, 2.0, 3.0, 4.0];
        $arr = NDArray::array($data);
        $result = $arr->toArray();

        $this->assertEqualsWithDelta($data, $result, 0.0001);
    }

    public function testToArray2D(): void
    {
        $data = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ];
        $arr = NDArray::array($data);
        $result = $arr->toArray();

        $this->assertEqualsWithDelta($data, $result, 0.0001);
    }

    public function testToArrayInt(): void
    {
        $data = [1, 2, 3, 4];
        $arr = NDArray::array($data, DType::Int32);
        $result = $arr->toArray();

        $this->assertSame($data, $result);
    }

    public function testToArrayBool(): void
    {
        $data = [true, false, true, false];
        $arr = NDArray::array($data, DType::Bool);
        $result = $arr->toArray();

        $this->assertSame($data, $result);
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    public function testCreateSingleElement(): void
    {
        $arr = NDArray::array([42.0]);

        $this->assertSame([1], $arr->shape());
        $this->assertSame(1, $arr->size());
        $this->assertEqualsWithDelta([42.0], $arr->toArray(), 0.0001);
    }

    public function testCreateLargeArray(): void
    {
        $data = range(1, 1000);
        $arr = NDArray::array($data, DType::Int64);

        $this->assertSame([1000], $arr->shape());
        $this->assertSame(1000, $arr->size());
    }

    public function testEmptyArrayThrows(): void
    {
        $this->expectException(ShapeException::class);
        $this->expectExceptionMessage('Cannot create array from empty data');

        NDArray::array([]);
    }

    public function testJaggedArrayThrows(): void
    {
        $this->expectException(ShapeException::class);
        // Rust error message should be propagated
        $this->expectExceptionMessage('Data length 3 does not match shape [2, 2] (expected 4)');

        // [[1, 2], [3]] -> inferred shape [2, 2], flattened length 3
        NDArray::array([[1, 2], [3]]);
    }

    // =========================================================================
    // All DType Roundtrip Tests
    // =========================================================================

    public function testFloat64Roundtrip(): void
    {
        $data = [1.5, 2.5, 3.5, 4.5];
        $arr = NDArray::array($data, DType::Float64);
        $result = $arr->toArray();

        $this->assertEqualsWithDelta($data, $result, 0.0001);
    }

    public function testFloat32Roundtrip(): void
    {
        $data = [1.5, 2.5, 3.5, 4.5];
        $arr = NDArray::array($data, DType::Float32);
        $result = $arr->toArray();

        $this->assertEqualsWithDelta($data, $result, 0.001);
    }

    public function testInt64Roundtrip(): void
    {
        $data = [100, 200, 300, 400];
        $arr = NDArray::array($data, DType::Int64);
        $result = $arr->toArray();

        $this->assertSame($data, $result);
    }

    public function testInt32Roundtrip(): void
    {
        $data = [100, 200, 300, 400];
        $arr = NDArray::array($data, DType::Int32);
        $result = $arr->toArray();

        $this->assertSame($data, $result);
    }

    public function testInt16Roundtrip(): void
    {
        $data = [100, 200, 300, 400];
        $arr = NDArray::array($data, DType::Int16);
        $result = $arr->toArray();

        $this->assertSame($data, $result);
    }

    public function testInt8Roundtrip(): void
    {
        $data = [10, 20, 30, 40];
        $arr = NDArray::array($data, DType::Int8);
        $result = $arr->toArray();

        $this->assertSame($data, $result);
    }

    public function testUint64Roundtrip(): void
    {
        $data = [100, 200, 300, 400];
        $arr = NDArray::array($data, DType::Uint64);
        $result = $arr->toArray();

        $this->assertSame($data, $result);
    }

    public function testUint32Roundtrip(): void
    {
        $data = [100, 200, 300, 400];
        $arr = NDArray::array($data, DType::Uint32);
        $result = $arr->toArray();

        $this->assertSame($data, $result);
    }

    public function testUint16Roundtrip(): void
    {
        $data = [100, 200, 300, 400];
        $arr = NDArray::array($data, DType::Uint16);
        $result = $arr->toArray();

        $this->assertSame($data, $result);
    }

    public function testUint8Roundtrip(): void
    {
        $data = [10, 20, 30, 40];
        $arr = NDArray::array($data, DType::Uint8);
        $result = $arr->toArray();

        $this->assertSame($data, $result);
    }

    public function testBoolRoundtrip(): void
    {
        $data = [true, false, true, false];
        $arr = NDArray::array($data, DType::Bool);
        $result = $arr->toArray();

        $this->assertSame($data, $result);
    }

    // =========================================================================
    // 2D Roundtrip for Each DType
    // =========================================================================

    public function test2DFloat64Roundtrip(): void
    {
        $data = [[1.0, 2.0], [3.0, 4.0]];
        $arr = NDArray::array($data, DType::Float64);
        $result = $arr->toArray();

        $this->assertEqualsWithDelta($data, $result, 0.0001);
    }

    public function test2DInt32Roundtrip(): void
    {
        $data = [[1, 2], [3, 4]];
        $arr = NDArray::array($data, DType::Int32);
        $result = $arr->toArray();

        $this->assertSame($data, $result);
    }

    public function test2DBoolRoundtrip(): void
    {
        $data = [[true, false], [false, true]];
        $arr = NDArray::array($data, DType::Bool);
        $result = $arr->toArray();

        $this->assertSame($data, $result);
    }
}
