<?php

declare(strict_types=1);

namespace NDArray\Tests\Unit;

use NDArray\DType;
use NDArray\Exceptions\DTypeException;
use NDArray\Exceptions\IndexException;
use NDArray\Exceptions\MathException;
use NDArray\Exceptions\NDArrayException;
use NDArray\Exceptions\ShapeException;
use NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Comprehensive tests for panic-to-exception mapping.
 *
 * These tests intentionally trigger real error conditions to verify
 * that Rust panics/errors are properly mapped to PHP exceptions.
 *
 * @covers \NDArray\FFI\Lib::checkStatus
 */
final class PanicMappingTest extends TestCase
{
    // =========================================================================
    // ShapeException Tests
    // =========================================================================

    public function testConcatenateIncompatibleShapesThrowsShapeException(): void
    {
        $a = NDArray::array([[1, 2], [3, 4]], DType::Float64);  // 2x2
        $b = NDArray::array([[5, 6, 7]], DType::Float64);  // 1x3

        $this->expectException(ShapeException::class);
        NDArray::concatenate([$a, $b], 0);
    }

    public function testConcatenateDifferentDimensionsThrowsShapeException(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);  // 1D
        $b = NDArray::array([[4, 5, 6]], DType::Float64);  // 2D

        $this->expectException(ShapeException::class);
        NDArray::concatenate([$a, $b], 0);
    }

    public function testStackIncompatibleShapesThrowsShapeException(): void
    {
        $a = NDArray::array([1, 2], DType::Float64);  // length 2
        $b = NDArray::array([3, 4, 5], DType::Float64);  // length 3

        $this->expectException(ShapeException::class);
        NDArray::stack([$a, $b], 0);
    }

    public function testInvalidAxisForSumThrowsShapeException(): void
    {
        $arr = NDArray::array([[1, 2], [3, 4]], DType::Float64);

        $this->expectException(ShapeException::class);
        $arr->sum(5);  // Axis 5 doesn't exist (only 0 and 1)
    }

    public function testInvalidAxisForMeanThrowsShapeException(): void
    {
        $arr = NDArray::array([[1, 2], [3, 4]], DType::Float64);

        $this->expectException(ShapeException::class);
        $arr->mean(-3);  // Axis -3 is out of bounds for 2D array
    }

    public function testInvalidAxisForMinThrowsShapeException(): void
    {
        $arr = NDArray::array([1, 2, 3], DType::Float64);

        $this->expectException(ShapeException::class);
        $arr->min(1);  // Axis 1 doesn't exist for 1D array
    }

    public function testInvalidAxisForMaxThrowsShapeException(): void
    {
        $arr = NDArray::array([1, 2, 3], DType::Float64);

        $this->expectException(ShapeException::class);
        $arr->max(1);  // Axis 1 doesn't exist for 1D array
    }

    public function testInvalidAxisForArgminThrowsShapeException(): void
    {
        $arr = NDArray::array([[1, 2], [3, 4]], DType::Float64);

        $this->expectException(ShapeException::class);
        $arr->argmin(5);
    }

    public function testInvalidAxisForArgmaxThrowsShapeException(): void
    {
        $arr = NDArray::array([[1, 2], [3, 4]], DType::Float64);

        $this->expectException(ShapeException::class);
        $arr->argmax(5);
    }

    public function testInvalidAxisForCumsumThrowsShapeException(): void
    {
        $arr = NDArray::array([[1, 2], [3, 4]], DType::Float64);

        $this->expectException(ShapeException::class);
        $arr->cumsum(5);
    }

    public function testInvalidAxisForCumprodThrowsShapeException(): void
    {
        $arr = NDArray::array([[1, 2], [3, 4]], DType::Float64);

        $this->expectException(ShapeException::class);
        $arr->cumprod(5);
    }

    public function testInvalidAxisForSortThrowsShapeException(): void
    {
        $arr = NDArray::array([[1, 2], [3, 4]], DType::Float64);

        $this->expectException(ShapeException::class);
        $arr->sort(5);
    }

    public function testInvalidAxisForArgsortThrowsShapeException(): void
    {
        $arr = NDArray::array([[1, 2], [3, 4]], DType::Float64);

        $this->expectException(ShapeException::class);
        $arr->argsort(5);
    }

    public function testEmptyConcatenateThrowsShapeException(): void
    {
        $this->expectException(ShapeException::class);
        $this->expectExceptionMessage('concatenate requires at least one array');
        NDArray::concatenate([], 0);
    }

    public function testEmptyStackThrowsShapeException(): void
    {
        $this->expectException(ShapeException::class);
        NDArray::stack([], 0);
    }

    // =========================================================================
    // IndexException Tests
    // =========================================================================

    public function testTakeWithInvalidIndicesThrowsIndexException(): void
    {
        $arr = NDArray::array([1, 2, 3, 4, 5], DType::Float64);
        $indices = NDArray::array([0, 1, 10], DType::Int64);  // 10 is out of bounds

        $this->expectException(IndexException::class);
        $arr->take($indices);
    }

    // =========================================================================
    // DTypeException Tests
    // =========================================================================

    public function testBincountOnFloatArrayThrowsDTypeException(): void
    {
        $arr = NDArray::array([1.5, 2.5, 3.5], DType::Float64);

        $this->expectException(DTypeException::class);
        $arr->bincount();
    }

    // =========================================================================
    // MathException Tests
    // =========================================================================

    public function testStdOnSingleElementThrowsException(): void
    {
        $arr = NDArray::array([42], DType::Float64);

        // std with ddof=1 on single element is undefined (division by zero)
        $this->expectException(NDArrayException::class);
        $this->expectExceptionMessage('ddof');
        $arr->std(null, 1);
    }

    public function testVarOnSingleElementThrowsException(): void
    {
        $arr = NDArray::array([42], DType::Float64);

        // var with ddof=1 on single element is undefined (division by zero)
        $this->expectException(NDArrayException::class);
        $this->expectExceptionMessage('ddof');
        $arr->var(null, 1);
    }

    // =========================================================================
    // Error Message Quality Tests
    // =========================================================================

    public function testShapeErrorMessageIsDescriptive(): void
    {
        $arr = NDArray::array([[1, 2], [3, 4]], DType::Float64);

        try {
            $arr->sum(5);
            $this->fail('Expected ShapeException');
        } catch (ShapeException $e) {
            $message = $e->getMessage();
            $this->assertNotEmpty($message);
            // Should mention the invalid axis
            $this->assertStringContainsString('5', $message);
            // Should not be generic panic message
            $this->assertStringNotContainsString('Rust panic occurred', $message);
        }
    }

    public function testConcatenateErrorMessageIsDescriptive(): void
    {
        $a = NDArray::array([[1, 2]], DType::Float64);
        $b = NDArray::array([[3, 4, 5]], DType::Float64);

        try {
            NDArray::concatenate([$a, $b], 0);
            $this->fail('Expected ShapeException');
        } catch (ShapeException $e) {
            $message = $e->getMessage();
            $this->assertNotEmpty($message);
            $this->assertStringNotContainsString('Rust panic occurred', $message);
        }
    }

    // =========================================================================
    // Multiple Error Code Coverage
    // =========================================================================

    /**
     * @dataProvider reductionOperationsProvider
     */
    public function testAllReductionsThrowShapeExceptionOnInvalidAxis(string $method): void
    {
        $arr = NDArray::array([[1, 2], [3, 4]], DType::Float64);

        $this->expectException(ShapeException::class);
        $arr->$method(5);
    }

    public static function reductionOperationsProvider(): array
    {
        return [
            ['sum'],
            ['mean'],
            ['min'],
            ['max'],
            ['product'],
        ];
    }
}
