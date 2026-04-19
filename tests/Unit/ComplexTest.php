<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Unit;

use PhpMlKit\NDArray\Complex;
use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * @internal
 *
 * @coversNothing
 */
final class ComplexTest extends TestCase
{
    public function testComplexArrayCreation(): void
    {
        $arr = NDArray::array([
            [new Complex(1, 2), new Complex(3, 4)],
            [new Complex(5, 6), new Complex(7, 8)],
        ], DType::Complex128);

        $this->assertSame([2, 2], $arr->shape());
        $this->assertSame(DType::Complex128, $arr->dtype());

        $result = $arr->toArray();
        $this->assertInstanceOf(Complex::class, $result[0][0]);
        $this->assertEqualsWithDelta(1.0, $result[0][0]->real, 0.0001);
        $this->assertEqualsWithDelta(2.0, $result[0][0]->imag, 0.0001);
        $this->assertEqualsWithDelta(7.0, $result[1][1]->real, 0.0001);
        $this->assertEqualsWithDelta(8.0, $result[1][1]->imag, 0.0001);
    }

    public function testComplex64ArrayCreation(): void
    {
        $arr = NDArray::array([
            [new Complex(1.5, 2.5), new Complex(3.5, 4.5)],
        ], DType::Complex64);

        $this->assertSame(DType::Complex64, $arr->dtype());

        $result = $arr->toArray();
        $this->assertEqualsWithDelta(1.5, $result[0][0]->real, 0.0001);
        $this->assertEqualsWithDelta(2.5, $result[0][0]->imag, 0.0001);
    }

    public function testComplexArrayFromNestedScalars(): void
    {
        $arr = NDArray::array([
            [1, 2],
            [3, 4],
        ], DType::Complex128);

        $this->assertSame(DType::Complex128, $arr->dtype());

        $result = $arr->toArray();
        $this->assertInstanceOf(Complex::class, $result[0][0]);
        $this->assertEqualsWithDelta(1.0, $result[0][0]->real, 0.0001);
        $this->assertEqualsWithDelta(0.0, $result[0][0]->imag, 0.0001);
    }

    public function testComplexDtypeInference(): void
    {
        $arr = NDArray::array([
            [new Complex(1, 2), 3],
            [4.0, 5],
        ]);

        $this->assertSame(DType::Complex128, $arr->dtype());
    }

    public function testComplexFull(): void
    {
        $arr = NDArray::full(new Complex(3, 4), [2, 2], DType::Complex128);

        $result = $arr->toArray();
        $this->assertEqualsWithDelta(3.0, $result[0][0]->real, 0.0001);
        $this->assertEqualsWithDelta(4.0, $result[0][0]->imag, 0.0001);
        $this->assertEqualsWithDelta(3.0, $result[1][1]->real, 0.0001);
        $this->assertEqualsWithDelta(4.0, $result[1][1]->imag, 0.0001);
    }

    public function testComplexElementAccess(): void
    {
        $arr = NDArray::array([
            [new Complex(1, 2), new Complex(3, 4)],
            [new Complex(5, 6), new Complex(7, 8)],
        ], DType::Complex128);

        $el = $arr->get(0, 1);
        $this->assertInstanceOf(Complex::class, $el);
        $this->assertEqualsWithDelta(3.0, $el->real, 0.0001);
        $this->assertEqualsWithDelta(4.0, $el->imag, 0.0001);
    }

    public function testComplexElementSet(): void
    {
        $arr = NDArray::array([
            [new Complex(1, 2), new Complex(3, 4)],
            [new Complex(5, 6), new Complex(7, 8)],
        ], DType::Complex128);

        $arr->set([1, 0], new Complex(10, 20));

        $el = $arr->get(1, 0);
        $this->assertInstanceOf(Complex::class, $el);
        $this->assertEqualsWithDelta(10.0, $el->real, 0.0001);
        $this->assertEqualsWithDelta(20.0, $el->imag, 0.0001);
    }

    public function testComplexSum(): void
    {
        $arr = NDArray::array([
            [new Complex(1, 2), new Complex(3, 4)],
            [new Complex(5, 6), new Complex(7, 8)],
        ], DType::Complex128);

        $sum = $arr->sum();
        $this->assertInstanceOf(Complex::class, $sum);
        $this->assertEqualsWithDelta(16.0, $sum->real, 0.0001);
        $this->assertEqualsWithDelta(20.0, $sum->imag, 0.0001);
    }

    public function testComplexMean(): void
    {
        $arr = NDArray::array([
            [new Complex(2, 4), new Complex(4, 8)],
        ], DType::Complex128);

        $mean = $arr->mean();
        $this->assertInstanceOf(Complex::class, $mean);
        $this->assertEqualsWithDelta(3.0, $mean->real, 0.0001);
        $this->assertEqualsWithDelta(6.0, $mean->imag, 0.0001);
    }

    public function testComplexProduct(): void
    {
        $arr = NDArray::array([
            [new Complex(1, 0), new Complex(0, 1)],
        ], DType::Complex128);

        $product = $arr->product();
        $this->assertInstanceOf(Complex::class, $product);
        $this->assertEqualsWithDelta(0.0, $product->real, 0.0001);
        $this->assertEqualsWithDelta(1.0, $product->imag, 0.0001);
    }

    public function testComplexSumAxis(): void
    {
        $arr = NDArray::array([
            [new Complex(1, 2), new Complex(3, 4)],
            [new Complex(5, 6), new Complex(7, 8)],
        ], DType::Complex128);

        $sum0 = $arr->sum(axis: 0);
        $this->assertSame([2], $sum0->shape());

        $result = $sum0->toArray();
        $this->assertInstanceOf(Complex::class, $result[0]);
        $this->assertEqualsWithDelta(6.0, $result[0]->real, 0.0001);
        $this->assertEqualsWithDelta(8.0, $result[0]->imag, 0.0001);
        $this->assertEqualsWithDelta(10.0, $result[1]->real, 0.0001);
        $this->assertEqualsWithDelta(12.0, $result[1]->imag, 0.0001);
    }

    public function testComplexFlatIterator(): void
    {
        $arr = NDArray::array([
            [new Complex(1, 2), new Complex(3, 4)],
            [new Complex(5, 6), new Complex(7, 8)],
        ], DType::Complex128);

        $flat = $arr->flat()->toArray();
        $this->assertCount(4, $flat);
        $this->assertInstanceOf(Complex::class, $flat[0]);
        $this->assertEqualsWithDelta(1.0, $flat[0]->real, 0.0001);
        $this->assertEqualsWithDelta(2.0, $flat[0]->imag, 0.0001);
    }

    public function testComplexArrayAccess(): void
    {
        $arr = NDArray::array([
            [new Complex(1, 2), new Complex(3, 4)],
            [new Complex(5, 6), new Complex(7, 8)],
        ], DType::Complex128);

        $el = $arr[0][1];
        $this->assertInstanceOf(Complex::class, $el);
        $this->assertEqualsWithDelta(3.0, $el->real, 0.0001);
        $this->assertEqualsWithDelta(4.0, $el->imag, 0.0001);
    }

    public function testComplexToScalar(): void
    {
        $arr = NDArray::full(new Complex(3, 4), [], DType::Complex128);
        $scalar = $arr->toScalar();
        $this->assertInstanceOf(Complex::class, $scalar);
        $this->assertEqualsWithDelta(3.0, $scalar->real, 0.0001);
        $this->assertEqualsWithDelta(4.0, $scalar->imag, 0.0001);
    }

    public function testComplexArithmeticAdd(): void
    {
        $a = NDArray::array([
            [new Complex(1, 2), new Complex(3, 4)],
        ], DType::Complex128);

        $b = NDArray::array([
            [new Complex(5, 6), new Complex(7, 8)],
        ], DType::Complex128);

        $c = $a->add($b);
        $result = $c->toArray();
        $this->assertEqualsWithDelta(6.0, $result[0][0]->real, 0.0001);
        $this->assertEqualsWithDelta(8.0, $result[0][0]->imag, 0.0001);
        $this->assertEqualsWithDelta(10.0, $result[0][1]->real, 0.0001);
        $this->assertEqualsWithDelta(12.0, $result[0][1]->imag, 0.0001);
    }

    public function testComplexAbs(): void
    {
        $arr = NDArray::array([
            [new Complex(3, 4), new Complex(0, 1)],
        ], DType::Complex128);

        $abs = $arr->abs();
        $result = $abs->toArray();
        $this->assertEqualsWithDelta(5.0, $result[0][0], 0.0001);
        $this->assertEqualsWithDelta(1.0, $result[0][1], 0.0001);
    }

    public function testComplexDeterminant(): void
    {
        $arr = NDArray::array([
            [new Complex(1, 0), new Complex(0, 1)],
            [new Complex(0, -1), new Complex(1, 0)],
        ], DType::Complex128);

        $det = $arr->det();
        $this->assertInstanceOf(Complex::class, $det);
        $this->assertEqualsWithDelta(0.0, $det->real, 0.0001);
        $this->assertEqualsWithDelta(0.0, $det->imag, 0.0001);
    }

    public function testComplexZerosOnes(): void
    {
        $zeros = NDArray::zeros([2, 2], DType::Complex128);
        $ones = NDArray::ones([2, 2], DType::Complex128);

        $z = $zeros->toArray();
        $o = $ones->toArray();

        $this->assertEqualsWithDelta(0.0, $z[0][0]->real, 0.0001);
        $this->assertEqualsWithDelta(0.0, $z[0][0]->imag, 0.0001);
        $this->assertEqualsWithDelta(1.0, $o[0][0]->real, 0.0001);
        $this->assertEqualsWithDelta(0.0, $o[0][0]->imag, 0.0001);
    }

    public function testComplexScalarAdd(): void
    {
        $arr = NDArray::array([
            [new Complex(1, 2), new Complex(3, 4)],
        ], DType::Complex128);

        $result = $arr->add(new Complex(10, 20));
        $values = $result->toArray();
        $this->assertEqualsWithDelta(11.0, $values[0][0]->real, 0.0001);
        $this->assertEqualsWithDelta(22.0, $values[0][0]->imag, 0.0001);
        $this->assertEqualsWithDelta(13.0, $values[0][1]->real, 0.0001);
        $this->assertEqualsWithDelta(24.0, $values[0][1]->imag, 0.0001);
    }

    public function testComplexScalarMultiply(): void
    {
        $arr = NDArray::array([
            [new Complex(1, 2), new Complex(3, 4)],
        ], DType::Complex128);

        $result = $arr->multiply(new Complex(2, 0));
        $values = $result->toArray();
        $this->assertEqualsWithDelta(2.0, $values[0][0]->real, 0.0001);
        $this->assertEqualsWithDelta(4.0, $values[0][0]->imag, 0.0001);
        $this->assertEqualsWithDelta(6.0, $values[0][1]->real, 0.0001);
        $this->assertEqualsWithDelta(8.0, $values[0][1]->imag, 0.0001);
    }

    public function testComplexScalarSubtract(): void
    {
        $arr = NDArray::array([
            [new Complex(5, 10), new Complex(3, 4)],
        ], DType::Complex128);

        $result = $arr->subtract(new Complex(2, 3));
        $values = $result->toArray();
        $this->assertEqualsWithDelta(3.0, $values[0][0]->real, 0.0001);
        $this->assertEqualsWithDelta(7.0, $values[0][0]->imag, 0.0001);
    }

    public function testComplexScalarDivide(): void
    {
        $arr = NDArray::array([
            [new Complex(6, 8)],
        ], DType::Complex128);

        $result = $arr->divide(new Complex(2, 0));
        $values = $result->toArray();
        $this->assertEqualsWithDelta(3.0, $values[0][0]->real, 0.0001);
        $this->assertEqualsWithDelta(4.0, $values[0][0]->imag, 0.0001);
    }

    public function testComplexCopy(): void
    {
        $arr = NDArray::array([
            [new Complex(1, 2), new Complex(3, 4)],
        ], DType::Complex128);

        $copy = $arr->copy();
        $this->assertSame(DType::Complex128, $copy->dtype());

        $result = $copy->toArray();
        $this->assertEqualsWithDelta(1.0, $result[0][0]->real, 0.0001);
        $this->assertEqualsWithDelta(2.0, $result[0][0]->imag, 0.0001);
    }

    // ==================== Broadcasting ====================

    public function testComplexBroadcastingWithRealScalar(): void
    {
        $a = NDArray::array([[new Complex(1, 2), new Complex(3, 4)], [new Complex(5, 6), new Complex(7, 8)]], DType::Complex128);
        $result = $a->add(10);

        $this->assertSame([2, 2], $result->shape());
        $values = $result->toArray();
        $this->assertEqualsWithDelta(11.0, $values[0][0]->real, 0.0001);
        $this->assertEqualsWithDelta(2.0, $values[0][0]->imag, 0.0001);
    }

    public function testComplexBroadcastingWithComplexScalar(): void
    {
        $a = NDArray::array([[new Complex(1, 2), new Complex(3, 4)]], DType::Complex128);
        $result = $a->add(new Complex(10, 20));

        $this->assertSame([1, 2], $result->shape());
        $values = $result->toArray();
        $this->assertEqualsWithDelta(11.0, $values[0][0]->real, 0.0001);
        $this->assertEqualsWithDelta(22.0, $values[0][0]->imag, 0.0001);
    }

    public function testComplexBroadcastingRowToMatrix(): void
    {
        $a = NDArray::array([[new Complex(1, 1), new Complex(2, 2)], [new Complex(3, 3), new Complex(4, 4)]], DType::Complex128);
        $b = NDArray::array([new Complex(10, 10), new Complex(20, 20)], DType::Complex128);
        $result = $a->add($b);

        $this->assertSame([2, 2], $result->shape());
        $values = $result->toArray();
        $this->assertEqualsWithDelta(11.0, $values[0][0]->real, 0.0001);
        $this->assertEqualsWithDelta(11.0, $values[0][0]->imag, 0.0001);
        $this->assertEqualsWithDelta(24.0, $values[1][1]->real, 0.0001);
        $this->assertEqualsWithDelta(24.0, $values[1][1]->imag, 0.0001);
    }

    // ==================== Slicing ====================

    public function testComplexSlicingRow(): void
    {
        $a = NDArray::array([
            [new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)],
            [new Complex(7, 8), new Complex(9, 10), new Complex(11, 12)],
        ], DType::Complex128);

        $row = $a[0];
        $this->assertSame([3], $row->shape());
        $values = $row->toArray();
        $this->assertEqualsWithDelta(1.0, $values[0]->real, 0.0001);
        $this->assertEqualsWithDelta(2.0, $values[0]->imag, 0.0001);
    }

    public function testComplexSlicingColumn(): void
    {
        $a = NDArray::array([
            [new Complex(1, 2), new Complex(3, 4)],
            [new Complex(5, 6), new Complex(7, 8)],
        ], DType::Complex128);

        $col = $a[':, 1'];
        $this->assertSame([2], $col->shape());
        $values = $col->toArray();
        $this->assertEqualsWithDelta(3.0, $values[0]->real, 0.0001);
        $this->assertEqualsWithDelta(4.0, $values[0]->imag, 0.0001);
    }

    public function testComplexSlicingElement(): void
    {
        $a = NDArray::array([
            [new Complex(1, 2), new Complex(3, 4)],
            [new Complex(5, 6), new Complex(7, 8)],
        ], DType::Complex128);

        $el = $a['1, 1'];
        $this->assertInstanceOf(Complex::class, $el);
        $this->assertEqualsWithDelta(7.0, $el->real, 0.0001);
        $this->assertEqualsWithDelta(8.0, $el->imag, 0.0001);
    }

    // ==================== Comparisons ====================

    public function testComplexEquality(): void
    {
        $a = NDArray::array([new Complex(1, 2), new Complex(3, 4), new Complex(1, 2)], DType::Complex128);
        $b = NDArray::array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)], DType::Complex128);

        $result = $a->eq($b);
        $this->assertSame(DType::Bool, $result->dtype());
        $this->assertSame([true, true, false], $result->toArray());
    }

    public function testComplexNotEqual(): void
    {
        $a = NDArray::array([new Complex(1, 2), new Complex(3, 4)], DType::Complex128);
        $b = NDArray::array([new Complex(1, 2), new Complex(5, 6)], DType::Complex128);

        $result = $a->ne($b);
        $this->assertSame([false, true], $result->toArray());
    }

    public function testComplexScalarEquality(): void
    {
        $a = NDArray::array([new Complex(1, 2), new Complex(3, 4), new Complex(1, 2)], DType::Complex128);
        $result = $a->eq(new Complex(1, 2));

        $this->assertSame([true, false, true], $result->toArray());
    }

    public function testComplexOrderingComparisonShouldFail(): void
    {
        $this->expectException(\Exception::class);
        $this->expectExceptionMessage('complex');

        $a = NDArray::array([new Complex(1, 2), new Complex(3, 4)], DType::Complex128);
        $a->gt(new Complex(1, 0));
    }

    // ==================== astype ====================

    public function testComplexAstypeToFloat(): void
    {
        $a = NDArray::array([new Complex(1.5, 2.5), new Complex(3.5, 4.5)], DType::Complex128);
        $result = $a->astype(DType::Float64);
        $this->assertSame(DType::Float64, $result->dtype());
        $this->assertEqualsWithDelta([1.5, 3.5], $result->toArray(), 0.0001);
    }

    public function testFloatAstypeToComplex(): void
    {
        $a = NDArray::array([1.5, 2.5, 3.5], DType::Float64);
        $result = $a->astype(DType::Complex128);

        $this->assertSame(DType::Complex128, $result->dtype());
        $values = $result->toArray();
        $this->assertEqualsWithDelta(1.5, $values[0]->real, 0.0001);
        $this->assertEqualsWithDelta(0.0, $values[0]->imag, 0.0001);
    }

    public function testIntAstypeToComplex(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Int64);
        $result = $a->astype(DType::Complex128);
        $this->assertSame(DType::Complex128, $result->dtype());
        $values = $result->toArray();
        $this->assertEqualsWithDelta(1.0, $values[0]->real, 0.0001);
        $this->assertEqualsWithDelta(0.0, $values[0]->imag, 0.0001);
    }

    public function testComplex128ToComplex64(): void
    {
        $a = NDArray::array([new Complex(1.5, 2.5)], DType::Complex128);
        $result = $a->astype(DType::Complex64);

        $this->assertSame(DType::Complex64, $result->dtype());
        $values = $result->toArray();
        $this->assertEqualsWithDelta(1.5, $values[0]->real, 0.0001);
        $this->assertEqualsWithDelta(2.5, $values[0]->imag, 0.0001);
    }

    // ==================== iscomplex / isreal ====================

    public function testIscomplexOnComplex128(): void
    {
        $a = NDArray::array([new Complex(1, 2), new Complex(3, 0), new Complex(0, 4)], DType::Complex128);
        $result = $a->iscomplex();

        $this->assertSame(DType::Bool, $result->dtype());
        $this->assertSame([true, false, true], $result->toArray());
    }

    public function testIscomplexOnComplex64(): void
    {
        $a = NDArray::array([new Complex(1, 0), new Complex(0, 1)], DType::Complex64);
        $result = $a->iscomplex();

        $this->assertSame([false, true], $result->toArray());
    }

    public function testIscomplexOnRealArray(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Int64);
        $result = $a->iscomplex();

        $this->assertSame(DType::Bool, $result->dtype());
        $this->assertSame([false, false, false], $result->toArray());
    }

    public function testIscomplexOnFloatArray(): void
    {
        $a = NDArray::array([1.5, 2.5], DType::Float64);
        $result = $a->iscomplex();

        $this->assertSame([false, false], $result->toArray());
    }

    public function testIsrealOnComplex128(): void
    {
        $a = NDArray::array([new Complex(1, 2), new Complex(3, 0), new Complex(0, 4)], DType::Complex128);
        $result = $a->isreal();

        $this->assertSame(DType::Bool, $result->dtype());
        $this->assertSame([false, true, false], $result->toArray());
    }

    public function testIsrealOnComplex64(): void
    {
        $a = NDArray::array([new Complex(1, 0), new Complex(0, 1)], DType::Complex64);
        $result = $a->isreal();

        $this->assertSame([true, false], $result->toArray());
    }

    public function testIsrealOnRealArray(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Int64);
        $result = $a->isreal();

        $this->assertSame(DType::Bool, $result->dtype());
        $this->assertSame([true, true, true], $result->toArray());
    }

    public function testIsrealOnFloatArray(): void
    {
        $a = NDArray::array([1.5, 2.5], DType::Float64);
        $result = $a->isreal();

        $this->assertSame([true, true], $result->toArray());
    }

    public function testIscomplexIsrealAreInverses(): void
    {
        $a = NDArray::array([new Complex(1, 2), new Complex(3, 0), new Complex(0, 4)], DType::Complex128);
        $isComplex = $a->iscomplex();
        $isReal = $a->isreal();

        $complexValues = $isComplex->toArray();
        $realValues = $isReal->toArray();

        for ($i = 0; $i < \count($complexValues); ++$i) {
            $this->assertNotSame($complexValues[$i], $realValues[$i]);
        }
    }

    public function testIscomplexIsrealOn2DArray(): void
    {
        $a = NDArray::array([
            [new Complex(1, 2), new Complex(3, 0)],
            [new Complex(0, 0), new Complex(5, 6)],
        ], DType::Complex128);

        $isComplex = $a->iscomplex();
        $isReal = $a->isreal();

        $this->assertSame([2, 2], $isComplex->shape());
        $this->assertSame([[true, false], [false, true]], $isComplex->toArray());
        $this->assertSame([[false, true], [true, false]], $isReal->toArray());
    }

    // ==================== Stacking ====================

    public function testComplexStack(): void
    {
        $a = NDArray::array([new Complex(1, 2), new Complex(3, 4)], DType::Complex128);
        $b = NDArray::array([new Complex(5, 6), new Complex(7, 8)], DType::Complex128);

        $result = NDArray::stack([$a, $b]);
        $this->assertSame([2, 2], $result->shape());
        $this->assertSame(DType::Complex128, $result->dtype());

        $values = $result->toArray();
        $this->assertEqualsWithDelta(1.0, $values[0][0]->real, 0.0001);
        $this->assertEqualsWithDelta(7.0, $values[1][1]->real, 0.0001);
    }

    public function testComplexConcatenate(): void
    {
        $a = NDArray::array([[new Complex(1, 2), new Complex(3, 4)]], DType::Complex128);
        $b = NDArray::array([[new Complex(5, 6), new Complex(7, 8)]], DType::Complex128);

        $result = NDArray::concatenate([$a, $b]);
        $this->assertSame([2, 2], $result->shape());
        $this->assertSame(DType::Complex128, $result->dtype());
    }

    // ==================== Reshape ====================

    public function testComplexReshape(): void
    {
        $a = NDArray::array([
            [new Complex(1, 2), new Complex(3, 4)],
            [new Complex(5, 6), new Complex(7, 8)],
        ], DType::Complex128);

        $result = $a->reshape([4]);
        $this->assertSame([4], $result->shape());
        $this->assertSame(DType::Complex128, $result->dtype());

        $values = $result->toArray();
        $this->assertEqualsWithDelta(5.0, $values[2]->real, 0.0001);
        $this->assertEqualsWithDelta(6.0, $values[2]->imag, 0.0001);
    }

    // ==================== real() ====================

    public function testRealOnComplex128(): void
    {
        $a = NDArray::array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)], DType::Complex128);
        $result = $a->real();

        $this->assertSame(DType::Float64, $result->dtype());
        $this->assertEqualsWithDelta([1.0, 3.0, 5.0], $result->toArray(), 0.0001);
    }

    public function testRealOnComplex64(): void
    {
        $a = NDArray::array([new Complex(1.5, 2.5), new Complex(3.5, 4.5)], DType::Complex64);
        $result = $a->real();

        $this->assertSame(DType::Float32, $result->dtype());
        $this->assertEqualsWithDelta([1.5, 3.5], $result->toArray(), 0.0001);
    }

    public function testRealOnFloat64(): void
    {
        $a = NDArray::array([1.5, 2.5, 3.5], DType::Float64);
        $result = $a->real();

        $this->assertSame(DType::Float64, $result->dtype());
        $this->assertEqualsWithDelta([1.5, 2.5, 3.5], $result->toArray(), 0.0001);
    }

    public function testRealOnInt64(): void
    {
        $a = NDArray::array([10, 20, 30], DType::Int64);
        $result = $a->real();

        $this->assertSame(DType::Int64, $result->dtype());
        $this->assertSame([10, 20, 30], $result->toArray());
    }

    public function testRealReturnsCopy(): void
    {
        $a = NDArray::array([1, 2, 3], DType::Int64);
        $result = $a->real();

        // Should be a copy, not the same object
        $this->assertNotSame($a, $result);
    }

    // ==================== imag() ====================

    public function testImagOnComplex128(): void
    {
        $a = NDArray::array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)], DType::Complex128);
        $result = $a->imag();

        $this->assertSame(DType::Float64, $result->dtype());
        $this->assertEqualsWithDelta([2.0, 4.0, 6.0], $result->toArray(), 0.0001);
    }

    public function testImagOnComplex64(): void
    {
        $a = NDArray::array([new Complex(1.5, 2.5), new Complex(3.5, 4.5)], DType::Complex64);
        $result = $a->imag();

        $this->assertSame(DType::Float32, $result->dtype());
        $this->assertEqualsWithDelta([2.5, 4.5], $result->toArray(), 0.0001);
    }

    public function testImagOnFloat64(): void
    {
        $a = NDArray::array([1.5, 2.5, 3.5], DType::Float64);
        $result = $a->imag();

        $this->assertSame(DType::Float64, $result->dtype());
        $this->assertEqualsWithDelta([0.0, 0.0, 0.0], $result->toArray(), 0.0001);
    }

    public function testImagOnInt64(): void
    {
        $a = NDArray::array([10, 20, 30], DType::Int64);
        $result = $a->imag();

        $this->assertSame(DType::Int64, $result->dtype());
        $this->assertSame([0, 0, 0], $result->toArray());
    }

    public function testImagOnUInt32(): void
    {
        $a = NDArray::array([1, 2, 3], DType::UInt32);
        $result = $a->imag();

        $this->assertSame(DType::UInt32, $result->dtype());
        $this->assertSame([0, 0, 0], $result->toArray());
    }

    // ==================== conjugate() ====================

    public function testConjugateOnComplex128(): void
    {
        $a = NDArray::array([new Complex(1, 2), new Complex(3, -4), new Complex(5, 0)], DType::Complex128);
        $result = $a->conjugate();

        $this->assertSame(DType::Complex128, $result->dtype());
        $values = $result->toArray();
        $this->assertEqualsWithDelta(1.0, $values[0]->real, 0.0001);
        $this->assertEqualsWithDelta(-2.0, $values[0]->imag, 0.0001);
        $this->assertEqualsWithDelta(3.0, $values[1]->real, 0.0001);
        $this->assertEqualsWithDelta(4.0, $values[1]->imag, 0.0001);
        $this->assertEqualsWithDelta(5.0, $values[2]->real, 0.0001);
        $this->assertEqualsWithDelta(0.0, $values[2]->imag, 0.0001);
    }

    public function testConjugateOnComplex64(): void
    {
        $a = NDArray::array([new Complex(1.5, 2.5)], DType::Complex64);
        $result = $a->conjugate();

        $this->assertSame(DType::Complex64, $result->dtype());
        $values = $result->toArray();
        $this->assertEqualsWithDelta(1.5, $values[0]->real, 0.0001);
        $this->assertEqualsWithDelta(-2.5, $values[0]->imag, 0.0001);
    }

    public function testConjugateOnFloat64(): void
    {
        $a = NDArray::array([1.5, 2.5, 3.5], DType::Float64);
        $result = $a->conjugate();

        $this->assertSame(DType::Float64, $result->dtype());
        $this->assertEqualsWithDelta([1.5, 2.5, 3.5], $result->toArray(), 0.0001);
    }

    public function testConjugateOnInt64(): void
    {
        $a = NDArray::array([10, 20, 30], DType::Int64);
        $result = $a->conjugate();

        $this->assertSame(DType::Int64, $result->dtype());
        $this->assertSame([10, 20, 30], $result->toArray());
    }

    public function testConjAlias(): void
    {
        $a = NDArray::array([new Complex(1, 2)], DType::Complex128);
        $result = $a->conj();

        $this->assertSame(DType::Complex128, $result->dtype());
        $values = $result->toArray();
        $this->assertEqualsWithDelta(1.0, $values[0]->real, 0.0001);
        $this->assertEqualsWithDelta(-2.0, $values[0]->imag, 0.0001);
    }

    // ==================== angle() ====================

    public function testAngleOnComplex128Radians(): void
    {
        $a = NDArray::array([new Complex(1, 0), new Complex(0, 1), new Complex(1, 1)], DType::Complex128);
        $result = $a->angle();

        $this->assertSame(DType::Float64, $result->dtype());
        $this->assertEqualsWithDelta([0.0, \M_PI_2, \M_PI_4], $result->toArray(), 0.0001);
    }

    public function testAngleOnComplex128Degrees(): void
    {
        $a = NDArray::array([new Complex(1, 0), new Complex(0, 1), new Complex(1, 1)], DType::Complex128);
        $result = $a->angle(true);

        $this->assertSame(DType::Float64, $result->dtype());
        $this->assertEqualsWithDelta([0.0, 90.0, 45.0], $result->toArray(), 0.0001);
    }

    public function testAngleOnComplex64(): void
    {
        $a = NDArray::array([new Complex(0, 1)], DType::Complex64);
        $result = $a->angle();

        $this->assertSame(DType::Float64, $result->dtype());
        $this->assertEqualsWithDelta([\M_PI_2], $result->toArray(), 0.0001);
    }

    public function testAngleOnPositiveReal(): void
    {
        $a = NDArray::array([1.0, 2.0, 3.0], DType::Float64);
        $result = $a->angle();

        $this->assertSame(DType::Float64, $result->dtype());
        $this->assertEqualsWithDelta([0.0, 0.0, 0.0], $result->toArray(), 0.0001);
    }

    public function testAngleOnNegativeReal(): void
    {
        $a = NDArray::array([-1.0, -2.0, -3.0], DType::Float64);
        $result = $a->angle();

        $this->assertSame(DType::Float64, $result->dtype());
        $this->assertEqualsWithDelta([\M_PI, \M_PI, \M_PI], $result->toArray(), 0.0001);
    }

    public function testAngleOnInt64(): void
    {
        $a = NDArray::array([1, -1, 0], DType::Int64);
        $result = $a->angle();

        $this->assertSame(DType::Float64, $result->dtype());
        $this->assertEqualsWithDelta([0.0, \M_PI, 0.0], $result->toArray(), 0.0001);
    }

    public function testAngleOnUInt32(): void
    {
        $a = NDArray::array([1, 2, 3], DType::UInt32);
        $result = $a->angle();

        $this->assertSame(DType::Float64, $result->dtype());
        $this->assertEqualsWithDelta([0.0, 0.0, 0.0], $result->toArray(), 0.0001);
    }

    public function testAngleOnBool(): void
    {
        $a = NDArray::array([true, false], DType::Bool);
        $result = $a->angle();

        $this->assertSame(DType::Float64, $result->dtype());
        $this->assertEqualsWithDelta([0.0, 0.0], $result->toArray(), 0.0001);
    }

    public function testAngleReturnsFloat64ForAllTypes(): void
    {
        $types = [
            DType::Int8, DType::Int16, DType::Int32, DType::Int64,
            DType::UInt8, DType::UInt16, DType::UInt32, DType::UInt64,
            DType::Float32, DType::Float64, DType::Bool,
        ];

        foreach ($types as $dtype) {
            $a = NDArray::array([1], $dtype);
            $result = $a->angle();
            $this->assertSame(
                DType::Float64,
                $result->dtype(),
                "angle() should return Float64 for input dtype {$dtype->name}"
            );
        }
    }

    // ==================== Edge Cases ====================

    public function testRealOn2DArray(): void
    {
        $a = NDArray::array([[new Complex(1, 2), new Complex(3, 4)], [new Complex(5, 6), new Complex(7, 8)]], DType::Complex128);
        $result = $a->real();

        $this->assertSame([2, 2], $result->shape());
        $this->assertEqualsWithDelta([[1.0, 3.0], [5.0, 7.0]], $result->toArray(), 0.0001);
    }

    public function testImagOn2DArray(): void
    {
        $a = NDArray::array([[new Complex(1, 2), new Complex(3, 4)], [new Complex(5, 6), new Complex(7, 8)]], DType::Complex128);
        $result = $a->imag();

        $this->assertSame([2, 2], $result->shape());
        $this->assertEqualsWithDelta([[2.0, 4.0], [6.0, 8.0]], $result->toArray(), 0.0001);
    }

    public function testAngleOnNegativeComplex(): void
    {
        $a = NDArray::array([new Complex(-1, 0), new Complex(0, -1), new Complex(-1, -1)], DType::Complex128);
        $result = $a->angle();

        $this->assertEqualsWithDelta([\M_PI, -\M_PI_2, -3 * \M_PI_4], $result->toArray(), 0.0001);
    }

    // ==================== Edge Cases ====================

    public function testNegativeRealAndImag(): void
    {
        $a = NDArray::array([
            [new Complex(-1, -2), new Complex(-3, 4)],
            [new Complex(5, -6), new Complex(-7, -8)],
        ], DType::Complex128);

        $sum = $a->sum();
        $this->assertEqualsWithDelta(-6.0, $sum->real, 0.0001);
        $this->assertEqualsWithDelta(-12.0, $sum->imag, 0.0001);

        $abs = $a->abs();
        $absValues = $abs->toArray();
        $this->assertEqualsWithDelta(hypot(-1, -2), $absValues[0][0], 0.0001);
        $this->assertEqualsWithDelta(hypot(-3, 4), $absValues[0][1], 0.0001);
    }

    public function testPureRealComplexArray(): void
    {
        $a = NDArray::array([
            [new Complex(1, 0), new Complex(2, 0)],
            [new Complex(3, 0), new Complex(4, 0)],
        ], DType::Complex128);

        $this->assertSame(DType::Complex128, $a->dtype());

        $isComplex = $a->iscomplex();
        $this->assertSame([[false, false], [false, false]], $isComplex->toArray());

        $isReal = $a->isreal();
        $this->assertSame([[true, true], [true, true]], $isReal->toArray());

        $imag = $a->imag();
        $this->assertEqualsWithDelta([[0.0, 0.0], [0.0, 0.0]], $imag->toArray(), 0.0001);
    }

    public function testPureImaginaryArray(): void
    {
        $a = NDArray::array([
            [new Complex(0, 1), new Complex(0, 2)],
            [new Complex(0, 3), new Complex(0, 4)],
        ], DType::Complex128);

        $real = $a->real();
        $this->assertEqualsWithDelta([[0.0, 0.0], [0.0, 0.0]], $real->toArray(), 0.0001);

        $isReal = $a->isreal();
        $this->assertSame([[false, false], [false, false]], $isReal->toArray());
    }

    public function testComplexCumsum(): void
    {
        $a = NDArray::array([
            new Complex(1, 1),
            new Complex(2, 2),
            new Complex(3, 3),
        ], DType::Complex128);

        $cumsum = $a->cumsum();
        $this->assertSame(DType::Complex128, $cumsum->dtype());

        $values = $cumsum->toArray();
        $this->assertEqualsWithDelta(1.0, $values[0]->real, 0.0001);
        $this->assertEqualsWithDelta(1.0, $values[0]->imag, 0.0001);
        $this->assertEqualsWithDelta(3.0, $values[1]->real, 0.0001);
        $this->assertEqualsWithDelta(3.0, $values[1]->imag, 0.0001);
        $this->assertEqualsWithDelta(6.0, $values[2]->real, 0.0001);
        $this->assertEqualsWithDelta(6.0, $values[2]->imag, 0.0001);
    }

    public function testComplexMatmulIdentity(): void
    {
        $identity = NDArray::array([
            [new Complex(1, 0), new Complex(0, 0)],
            [new Complex(0, 0), new Complex(1, 0)],
        ], DType::Complex128);

        $a = NDArray::array([
            [new Complex(1, 2), new Complex(3, 4)],
            [new Complex(5, 6), new Complex(7, 8)],
        ], DType::Complex128);

        $result = $identity->matmul($a);
        $this->assertSame(DType::Complex128, $result->dtype());

        $values = $result->toArray();
        $this->assertEqualsWithDelta(1.0, $values[0][0]->real, 0.0001);
        $this->assertEqualsWithDelta(2.0, $values[0][0]->imag, 0.0001);
        $this->assertEqualsWithDelta(5.0, $values[1][0]->real, 0.0001);
        $this->assertEqualsWithDelta(6.0, $values[1][0]->imag, 0.0001);
    }
}
