<?php

declare(strict_types=1);

namespace NDArray\Tests\Unit;

use NDArray\DType;
use NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Tests for shape manipulation operations
 */
final class ShapeOpsTest extends TestCase
{
    // =========================================================================
    // Reshape Tests
    // =========================================================================

    public function testReshape1DTo2D(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5, 6], DType::Float64);
        $result = $a->reshape([2, 3]);
        
        $this->assertSame([2, 3], $result->shape());
        $this->assertEqualsWithDelta([[1, 2, 3], [4, 5, 6]], $result->toArray(), 0.0001);
    }

    public function testReshape2DTo1D(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->reshape([6]);
        
        $this->assertSame([6], $result->shape());
        $this->assertEqualsWithDelta([1, 2, 3, 4, 5, 6], $result->toArray(), 0.0001);
    }

    public function testReshape3D(): void
    {
        $a = NDArray::array([1, 2, 3, 4, 5, 6, 7, 8], DType::Float64);
        $result = $a->reshape([2, 2, 2]);
        
        $this->assertSame([2, 2, 2], $result->shape());
    }

    public function testReshapePreservesDtype(): void
    {
        $a = NDArray::array([1, 2, 3, 4], DType::Int32);
        $result = $a->reshape([2, 2]);
        
        $this->assertSame(DType::Int32, $result->dtype());
    }

    // =========================================================================
    // Transpose Tests
    // =========================================================================

    public function testTranspose2D(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->transpose();
        
        $this->assertSame([3, 2], $result->shape());
        $this->assertEqualsWithDelta([[1, 4], [2, 5], [3, 6]], $result->toArray(), 0.0001);
    }

    public function testTransposePreservesData(): void
    {
        $a = NDArray::array([[1, 2], [3, 4], [5, 6]], DType::Float64);
        $result = $a->transpose();
        
        $this->assertSame([2, 3], $result->shape());
        $this->assertEqualsWithDelta([[1, 3, 5], [2, 4, 6]], $result->toArray(), 0.0001);
    }

    // =========================================================================
    // Swap Axes Tests
    // =========================================================================

    public function testSwapAxes2D(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->swapAxes(0, 1);
        
        $this->assertSame([3, 2], $result->shape());
        $this->assertEqualsWithDelta([[1, 4], [2, 5], [3, 6]], $result->toArray(), 0.0001);
    }

    // =========================================================================
    // Flatten Tests
    // =========================================================================

    public function testFlatten2D(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->flatten();
        
        $this->assertSame([6], $result->shape());
        $this->assertEqualsWithDelta([1, 2, 3, 4, 5, 6], $result->toArray(), 0.0001);
    }

    public function testFlatten3D(): void
    {
        $a = NDArray::array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], DType::Float64);
        $result = $a->flatten();
        
        $this->assertSame([8], $result->shape());
        $this->assertEqualsWithDelta([1, 2, 3, 4, 5, 6, 7, 8], $result->toArray(), 0.0001);
    }

    // =========================================================================
    // Squeeze Tests
    // =========================================================================

    public function testSqueezeAllOnes(): void
    {
        $a = NDArray::array([[[[1]]]], DType::Float64);
        $result = $a->squeeze();
        
        // Should keep at least 1 dimension
        $this->assertSame([1], $result->shape());
    }

    public function testSqueezeSpecificAxis(): void
    {
        $a = NDArray::zeros([3, 1, 4], DType::Float64);
        $result = $a->squeeze([1]);
        
        $this->assertSame([3, 4], $result->shape());
    }

    public function testSqueezeMultipleAxes(): void
    {
        $a = NDArray::zeros([1, 3, 1, 4, 1], DType::Float64);
        $result = $a->squeeze([0, 2, 4]);
        
        $this->assertSame([3, 4], $result->shape());
    }

    // =========================================================================
    // Expand Dims Tests
    // =========================================================================

    public function testExpandDimsAtBeginning(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $result = $a->expandDims(0);
        
        $this->assertSame([1, 3], $result->shape());
    }

    public function testExpandDimsAtEnd(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $result = $a->expandDims(1);
        
        $this->assertSame([3, 1], $result->shape());
    }

    public function testExpandDimsMiddle(): void
    {
        $a = NDArray::zeros([2, 3], DType::Float64);
        $result = $a->expandDims(1);
        
        $this->assertSame([2, 1, 3], $result->shape());
    }

    public function testExpandDimsNegativeAxis(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Float64);
        $result = $a->expandDims(-1);
        
        $this->assertSame([3, 1], $result->shape());
    }

    // =========================================================================
    // Ravel Tests
    // =========================================================================

    public function testRavel2D(): void
    {
        $a = NDArray::array([[1, 2, 3], [4, 5, 6]], DType::Float64);
        $result = $a->ravel();
        
        $this->assertSame([6], $result->shape());
        $this->assertEqualsWithDelta([1, 2, 3, 4, 5, 6], $result->toArray(), 0.0001);
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    public function testChainedOperations(): void
    {
        // Create 1D array, reshape to 2D, transpose, then flatten
        $a = NDArray::array([1, 2, 3, 4, 5, 6], DType::Float64);
        $result = $a->reshape([2, 3])->transpose()->flatten();
        
        $this->assertSame([6], $result->shape());
        // After reshape: [[1,2,3],[4,5,6]], transpose: [[1,4],[2,5],[3,6]], flatten: [1,4,2,5,3,6]
        $this->assertEqualsWithDelta([1, 4, 2, 5, 3, 6], $result->toArray(), 0.0001);
    }

    public function testSqueezeThenExpand(): void
    {
        $a = NDArray::zeros([1, 3, 1, 4, 1], DType::Float64);
        $squeezed = $a->squeeze();
        $expanded = $squeezed->expandDims(0);
        
        $this->assertSame([1, 3, 4], $expanded->shape());
    }
}
