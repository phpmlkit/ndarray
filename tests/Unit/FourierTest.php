<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Unit;

use PhpMlKit\NDArray\Complex;
use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\Exceptions\DTypeException;
use PhpMlKit\NDArray\NDArray;
use PhpMlKit\NDArray\Normalization;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;

/**
 * @internal
 *
 * @coversNothing
 */
final class FourierTest extends TestCase
{
    #[DataProvider('reconstructingNormProvider')]
    public function testFftIfftRoundTripMatchesNorm(Normalization $norm): void
    {
        $x = NDArray::array([0.0, 1.0, 2.0, 3.0], DType::Float64);

        $z = $x->fft(norm: $norm)->ifft(norm: $norm);

        $this->assertSame($x->shape(), $z->shape(), "fft/ifft {$norm->value}");
        $this->assertEqualsWithDelta($x->toArray(), $z->real()->toArray(), 1e-9, "fft/ifft {$norm->value}");
    }

    #[DataProvider('reconstructingNormAndFftLengthProvider')]
    public function testFftIfftRoundTripVariousLengths(Normalization $norm, int $n): void
    {
        $values = [];
        for ($i = 0; $i < $n; ++$i) {
            $values[] = 0.25 * $i - 0.125 * ($i % 3);
        }
        $x = NDArray::array($values, DType::Float64);

        $z = $x->fft(norm: $norm)->ifft(norm: $norm);

        $this->assertSame($x->shape(), $z->shape(), "fft/ifft {$norm->value} N={$n}");
        $this->assertEqualsWithDelta($x->toArray(), $z->real()->toArray(), $n > 12 ? 1e-8 : 1e-9, "fft/ifft {$norm->value} N={$n}");
    }

    #[DataProvider('reconstructingNormProvider')]
    public function testFftn2DRoundTripNorms(Normalization $norm): void
    {
        $a = NDArray::array([
            [1.0, 0.0, 2.0],
            [0.0, 3.0, -0.5],
        ], DType::Float64);

        $z = $a->fftn(norm: $norm)->ifftn(norm: $norm);
        $this->assertSame($a->shape(), $z->shape());
        $this->assertEqualsWithDelta($a->toArray(), $z->real()->toArray(), 1e-8, "fftn/ifftn {$norm->value}");
    }

    #[DataProvider('reconstructingNormProvider')]
    public function testFft2Ifft2ThreeDimensionalNorms(Normalization $norm): void
    {
        $a = NDArray::array([
            [[1.0, 2.0], [0.0, -1.0]],
            [[0.5, 0.0], [3.0, 1.0]],
        ], DType::Float64);

        $z = $a->fft2($norm)->ifft2($norm);
        $this->assertSame($a->shape(), $z->shape());
        $this->assertEqualsWithDelta($a->toArray(), $z->real()->toArray(), 1e-8, "fft2/ifft2 {$norm->value}");
    }

    public function testFftIfftNoneScalesByLength(): void
    {
        foreach ([2, 4, 8, 16] as $n) {
            $x = NDArray::arange(0, $n, 1, DType::Float64);
            $z = $x->fft(norm: Normalization::None)->ifft(norm: Normalization::None);

            $this->assertEqualsWithDelta($x->multiply($n)->toArray(), $z->real()->toArray(), 1e-7, "none N={$n}");
        }
    }

    public function testFftnNone2DScalesByAxisProduct(): void
    {
        $a = NDArray::array([
            [1.0, 2.0, 0.5],
            [-1.0, 0.0, 1.5],
        ], DType::Float64);

        $prod = 2 * 3;
        $z = $a->fftn(norm: Normalization::None)->ifftn(norm: Normalization::None);
        $this->assertEqualsWithDelta($a->multiply($prod)->toArray(), $z->real()->toArray(), 1e-7);
        $this->assertEqualsWithDelta($z->imag()->toArray(), [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], 1e-7);
    }

    #[DataProvider('reconstructingNormProvider')]
    public function testRfftIrfftRoundTripNorms(Normalization $norm): void
    {
        $x = NDArray::array([1.0, 2.0, 3.0, 4.0, 5.0], DType::Float64);

        $z = $x->rfft(norm: $norm)->irfft(n: 5, norm: $norm);
        $this->assertSame($x->shape(), $z->shape());
        $this->assertEqualsWithDelta($x->toArray(), $z->toArray(), 1e-8, "rfft/irfft {$norm->value}");
    }

    #[DataProvider('reconstructingNormAndFftLengthProvider')]
    public function testRfftIrfftRoundTripOddAndEvenLengths(Normalization $norm, int $n): void
    {
        $values = [];
        for ($i = 0; $i < $n; ++$i) {
            $values[] = 0.5 * $i - 0.25 * ($i % 2);
        }
        $x = NDArray::array($values, DType::Float64);

        $z = $x->rfft(norm: $norm)->irfft(n: $n, norm: $norm);
        $this->assertSame($x->shape(), $z->shape());
        $this->assertEqualsWithDelta($x->toArray(), $z->toArray(), 1e-8, "rfft/irfft {$norm->value} N={$n}");
    }

    /**
     * @return \Generator<string, array{0: Normalization, 1: int}>
     */
    public static function reconstructingNormAndFftLengthProvider(): \Generator
    {
        foreach ([Normalization::Backward, Normalization::Default, Normalization::Ortho, Normalization::Forward] as $norm) {
            foreach ([2, 3, 4, 8, 16] as $n) {
                yield "{$norm->name}_N{$n}" => [$norm, $n];
            }
        }
    }

    public function testRfftIrfftNoneScalesByLength(): void
    {
        foreach ([4, 8] as $n) {
            $x = NDArray::arange(0, $n, 1, DType::Float64);

            $z = $x->rfft(norm: Normalization::None)->irfft(n: $n, norm: Normalization::None);
            $this->assertEqualsWithDelta($x->multiply($n)->toArray(), $z->toArray(), 1e-6, "rfft none N={$n}");
        }
    }

    public function testDefaultNormAliasMatchesExplicitBackward(): void
    {
        $x = NDArray::array([0.25, -1.0, 0.75, 2.0], DType::Float64);
        $yB = $x->fft(norm: Normalization::Backward);
        $yD = $x->fft(norm: Normalization::Default);

        $this->assertEqualsWithDelta($yB->toArray(), $yD->toArray(), 1e-12);

        $zB = $yB->ifft(norm: Normalization::Backward);
        $zD = $yD->ifft(norm: Normalization::Default);

        $this->assertEqualsWithDelta($zB->toArray(), $zD->toArray(), 1e-12);
    }

    #[DataProvider('dctRoundTripTypeNormLengthProvider')]
    public function testDctIdctRoundTripReconstructingNorms(int $dctType, int $n, Normalization $norm): void
    {
        $values = [];
        for ($i = 0; $i < $n; ++$i) {
            $values[] = 0.2 * $i - 0.07 * ($i % 4);
        }
        $x = NDArray::array($values, DType::Float64);

        $y = $x->dct($dctType, norm: $norm);
        $z = $y->idct($dctType, norm: $norm);
        $this->assertEqualsWithDelta($x->toArray(), $z->toArray(), 1e-6, "dct/idct type {$dctType} {$norm->value} N={$n}");
    }

    /**
     * @return \Generator<string, array{0: int, 1: int, 2: Normalization}>
     */
    public static function dctRoundTripTypeNormLengthProvider(): \Generator
    {
        foreach ([1, 2, 3, 4] as $type) {
            foreach ([Normalization::Backward, Normalization::Default, Normalization::Ortho, Normalization::Forward] as $norm) {
                foreach ([2, 3, 4, 5, 8, 16] as $n) {
                    yield "t{$type}_{$norm->name}_N{$n}" => [$type, $n, $norm];
                }
            }
        }
    }

    #[DataProvider('dctNoneTypeLengthProvider')]
    public function testDctIdctNoneIsScalarMultiple(int $dctType, int $n): void
    {
        $values = [];
        for ($i = 0; $i < $n; ++$i) {
            $values[] = 0.3 * $i + 0.1;
        }
        $x = NDArray::array($values, DType::Float64);

        $z = $x->dct($dctType, norm: Normalization::None)->idct($dctType, norm: Normalization::None);
        $c = self::expectedDctNoneRoundTripScale($dctType, $n);
        $this->assertEqualsWithDelta($x->multiply($c)->toArray(), $z->toArray(), 1e-5, "dct none t{$dctType} N={$n}");
    }

    /**
     * @return \Generator<string, array{0: int, 1: int}>
     */
    public static function dctNoneTypeLengthProvider(): \Generator
    {
        foreach ([1, 2, 3, 4] as $type) {
            foreach ([2, 3, 4, 5, 8, 16] as $n) {
                yield "t{$type}_N{$n}" => [$type, $n];
            }
        }
    }

    #[DataProvider('reconstructingNormProvider')]
    public function testDct2Idct2RoundTrip2DGridNorms(Normalization $norm): void
    {
        $a = NDArray::array([
            [1.0, 0.0, 2.0],
            [0.0, 3.0, -0.25],
            [-0.5, 1.0, 0.0],
        ], DType::Float64);

        $y = $a->dct2(2, norm: $norm);
        $z = $y->idct2(2, norm: $norm);
        $this->assertSame($a->shape(), $z->shape());
        $this->assertEqualsWithDelta($a->toArray(), $z->toArray(), 1e-5, "dct2/idct2 {$norm->value}");
    }

    #[DataProvider('reconstructingNormProvider')]
    public function testDctnIdctnAllAxesMatchesSequentialNorms(Normalization $norm): void
    {
        $a = NDArray::array([[1.0, -0.5], [2.0, 0.25], [0.0, 1.0]], DType::Float64);

        $full = $a->dctn(type: 3, norm: $norm);
        $step1 = $a->dct(3, axis: 0, norm: $norm);
        $step2 = $step1->dct(3, axis: 1, norm: $norm);

        $this->assertSame($a->shape(), $full->shape());
        $this->assertEqualsWithDelta($full->toArray(), $step2->toArray(), 1e-8, "dctn dct3 {$norm->value}");

        $back = $full->idctn(type: 3, norm: $norm);
        $this->assertEqualsWithDelta($a->toArray(), $back->toArray(), 1e-5, "dctn/idctn round-trip {$norm->value}");
    }

    /**
     * Norms where `ifft(fft(x)) ≈ x` and `idct(dct(x)) ≈ x` (SciPy-style).
     *
     * @return \Generator<string, array{0: Normalization}>
     */
    public static function reconstructingNormProvider(): \Generator
    {
        foreach ([Normalization::Backward, Normalization::Default, Normalization::Ortho, Normalization::Forward] as $norm) {
            yield $norm->name => [$norm];
        }
    }

    public function testFftIfftRoundTrip1D(): void
    {
        $x = NDArray::array([0.0, 1.0, 2.0, 3.0], DType::Float64);

        $X = $x->fft(norm: Normalization::Backward);

        $expectedX = NDArray::array([new Complex(6.0), new Complex(-2.0, 2.0), new Complex(-2.0), new Complex(-2.0, -2.0)]);
        $this->assertEqualsWithDelta($expectedX->toArray(), $X->toArray(), 1e-10);

        $y = $X->ifft(norm: Normalization::Backward);

        $expectedY = NDArray::array([0.0, 1.0, 2.0, 3.0], DType::Complex128);
        $this->assertEqualsWithDelta($expectedY->toArray(), $y->toArray(), 1e-10);
    }

    public function testRfftIrfftRoundTrip1D(): void
    {
        $x = NDArray::array([1.0, 2.0, 3.0, 4.0], DType::Float64);

        $X = $x->rfft();

        $expectedX = NDArray::array([new Complex(10.0), new Complex(-2.0, 2.0), new Complex(-2.0)]);
        $this->assertEqualsWithDelta($expectedX->toArray(), $X->toArray(), 1e-10);

        $y = $X->irfft();

        $expectedY = NDArray::array([1.0, 2.0, 3.0, 4.0], DType::Float64);
        $this->assertEqualsWithDelta($expectedY->toArray(), $y->toArray(), 1e-10);
    }

    public function testFftn2D(): void
    {
        $a = NDArray::array([
            [1.0, 0.0],
            [0.0, 0.0],
        ], DType::Float64);

        $F = $a->fftn();
        $expectedF = NDArray::array([[1, 1], [1, 1]], DType::Complex128);
        $this->assertEqualsWithDelta($expectedF->toArray(), $F->toArray(), 1e-10);

        $a2 = $F->ifftn();
        $expectedA2 = NDArray::array([[1.0, 0.0], [0.0, 0.0]], DType::Complex128);
        $this->assertEqualsWithDelta($expectedA2->toArray(), $a2->toArray(), 1e-10);
    }

    public function testFft2LastTwoAxes(): void
    {
        $a = NDArray::zeros([2, 3, 4], DType::Float64);
        $F = $a->fft2();
        $expectedF = NDArray::zeros([2, 3, 4], DType::Complex128);
        $this->assertEqualsWithDelta($expectedF->toArray(), $F->toArray(), 1e-10);
    }

    public function testFft2Rejects1D(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        NDArray::array([1.0, 2.0], DType::Float64)->fft2();
    }

    public function testFftFloat32PromotesToComplex64(): void
    {
        $x = NDArray::array([1.0, 0.0, -1.0, 0.0], DType::Float32);
        $X = $x->fft();
        $this->assertSame(DType::Complex64, $X->dtype());
    }

    // -------------------------------------------------------------------------
    // DCT / inverse DCT (real, types I–IV; inverse pairing matches SciPy `idct`)
    // -------------------------------------------------------------------------

    /**
     * DCT-II forward (SciPy backward norm) and full round-trip `idct(dct(x)) ≈ x`.
     */
    public function testDct2ForwardAndIdctRoundTrip1D(): void
    {
        $x = NDArray::array([1.0, 2.0, 3.0, 4.0], DType::Float64);

        $y = $x->dct(2, norm: Normalization::Backward);

        $expectedY = NDArray::array([20.0, -6.3086440597979, 0.0, -0.44834152916797], DType::Float64);
        $this->assertEqualsWithDelta($expectedY->toArray(), $y->toArray(), 1e-9);

        $z = $y->idct(2, norm: Normalization::Backward);

        $expectedZ = NDArray::array([1.0, 2.0, 3.0, 4.0], DType::Float64);
        $this->assertEqualsWithDelta($expectedZ->toArray(), $z->toArray(), 1e-9);
    }

    /**
     * For each DCT type, `idct(dct(x), backward)` recovers `x`.
     */
    public function testDctIdctCompositionIsScalarMultiple1D(): void
    {
        $x = NDArray::array([0.25, -0.5, 1.0, 0.125], DType::Float64);

        foreach ([1, 2, 3, 4] as $type) {
            $y = $x->dct($type, norm: Normalization::Backward);
            $z = $y->idct($type, norm: Normalization::Backward);
            $this->assertSame([4], $z->shape(), "shape after idct type {$type}");
            $this->assertEqualsWithDelta($x->toArray(), $z->toArray(), 1e-6, "DCT type {$type}");
        }
    }

    /**
     * 2-D DCT on last two axes: `idct2(dct2(A)) ≈ A` with backward norm.
     */
    public function testDct2Idct2CompositionIsScalarMultiple2D(): void
    {
        $a = NDArray::array([
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
        ], DType::Float64);

        $Y = $a->dct2(2, norm: Normalization::Backward);
        $this->assertSame([2, 3], $Y->shape());
        $this->assertSame(DType::Float64, $Y->dtype());

        $a2 = $Y->idct2(2, norm: Normalization::Backward);
        $this->assertSame([2, 3], $a2->shape());

        $this->assertEqualsWithDelta($a->toArray(), $a2->toArray(), 1e-6);
    }

    /** `dctn` over all axes equals sequential application along axis 0 then 1 for a 2×2 matrix. */
    public function testDctnAllAxesMatchesSequential2D(): void
    {
        $a = NDArray::array([[1.0, 2.0], [3.0, 4.0]], DType::Float64);

        $full = $a->dctn(null, 2, norm: Normalization::Backward);

        $step1 = $a->dct(2, axis: 0, norm: Normalization::Backward);
        $step2 = $step1->dct(2, axis: 1, norm: Normalization::Backward);

        $this->assertSame([2, 2], $full->shape());
        $this->assertEqualsWithDelta($full->toArray(), $step2->toArray(), 1e-10);
    }

    public function testDctFloat32PreservesDtypeAndCompositionScaling(): void
    {
        $x = NDArray::array([1.0, 0.0, -1.0, 0.0], DType::Float32);
        $y = $x->dct(2, norm: Normalization::Backward);
        $this->assertSame(DType::Float32, $y->dtype());
        $z = $y->idct(2, norm: Normalization::Backward);
        $this->assertSame(DType::Float32, $z->dtype());
        $this->assertEqualsWithDelta($x->toArray(), $z->toArray(), 1e-4);
    }

    public function testDctRejectsComplexInput(): void
    {
        $this->expectException(DTypeException::class);
        NDArray::array([new Complex(1.0, 0.0)], DType::Complex128)->dct();
    }

    public function testDctInvalidTypeThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('DCT type must be');
        NDArray::array([1.0, 2.0], DType::Float64)->dct(5);
    }

    /** After raw `rustdct` + inverse, round-trip equals `c * x` (see Rust `norm='none'` path). */
    private static function expectedDctNoneRoundTripScale(int $dctType, int $n): float
    {
        if (1 === $dctType) {
            return $n > 1 ? ($n - 1) / 2.0 : 1.0;
        }

        return $n / 2.0;
    }
}
