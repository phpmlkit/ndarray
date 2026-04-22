<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Unit;

use PhpMlKit\NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * @internal
 *
 * @coversNothing
 */
final class WindowFunctionsTest extends TestCase
{
    public function testWindowM1IsOne(): void
    {
        $this->assertEqualsWithDelta(1.0, NDArray::bartlett(1)[0], 1e-12);
        $this->assertEqualsWithDelta(1.0, NDArray::blackman(1)[0], 1e-12);
        $this->assertEqualsWithDelta(1.0, NDArray::bohman(1)[0], 1e-12);
        $this->assertEqualsWithDelta(1.0, NDArray::boxcar(1)[0], 1e-12);
        $this->assertEqualsWithDelta(1.0, NDArray::hamming(1)[0], 1e-12);
        $this->assertEqualsWithDelta(1.0, NDArray::hanning(1)[0], 1e-12);
        $this->assertEqualsWithDelta(1.0, NDArray::kaiser(1, 12.0)[0], 1e-12);
        $this->assertEqualsWithDelta(1.0, NDArray::lanczos(1)[0], 1e-12);
        $this->assertEqualsWithDelta(1.0, NDArray::triang(1)[0], 1e-12);
    }

    public function testKnownValuesM4(): void
    {
        $m = 4;

        $hann = NDArray::hanning($m);
        $this->assertEqualsWithDelta($hann->toArray(), [0.0, 0.75, 0.75, 0.0], 1e-12);

        $hamming = NDArray::hamming($m);
        $this->assertEqualsWithDelta($hamming->toArray(), [0.08, 0.77, 0.77, 0.08], 1e-12);

        $blackman = NDArray::blackman($m);
        $this->assertEqualsWithDelta($blackman->toArray(), [0.0, 0.63, 0.63, 0.0], 1e-12);

        $bartlett = NDArray::bartlett($m);
        $this->assertEqualsWithDelta($bartlett->toArray(), [0.0, 2.0 / 3.0, 2.0 / 3.0, 0.0], 1e-12);

        $kaiser = NDArray::kaiser($m, 0.0);
        $this->assertEqualsWithDelta($kaiser->toArray(), [1.0, 1.0, 1.0, 1.0], 1e-12);

        $boxcar = NDArray::boxcar($m);
        $this->assertEqualsWithDelta($boxcar->toArray(), [1.0, 1.0, 1.0, 1.0], 1e-12);

        $triang = NDArray::triang($m);
        $this->assertEqualsWithDelta($triang->toArray(), [0.25, 0.75, 0.75, 0.25], 1e-12);

        $lanczos = NDArray::lanczos($m);
        $this->assertEqualsWithDelta($lanczos->toArray(), [0.0, 0.8269933431326881, 0.8269933431326881, 0.0], 1e-12);

        $bohman = NDArray::bohman($m);
        $this->assertEqualsWithDelta($bohman->toArray(), [0.0, 0.6089977810442294, 0.6089977810442294, 0.0], 1e-12);
    }

    public function testPeriodicEqualsSymmetricDropLast(): void
    {
        $m = 8;

        $wPer = NDArray::hanning($m, periodic: true);
        $wSym = NDArray::hanning($m + 1, periodic: false);
        $this->assertEqualsWithDelta($wSym[':-1']->toArray(), $wPer->toArray(), 1e-12);

        $wPer = NDArray::hamming($m, periodic: true);
        $wSym = NDArray::hamming($m + 1, periodic: false);
        $this->assertEqualsWithDelta($wSym[':-1']->toArray(), $wPer->toArray(), 1e-12);

        $wPer = NDArray::blackman($m, periodic: true);
        $wSym = NDArray::blackman($m + 1, periodic: false);
        $this->assertEqualsWithDelta($wSym[':-1']->toArray(), $wPer->toArray(), 1e-12);

        $wPer = NDArray::bartlett($m, periodic: true);
        $wSym = NDArray::bartlett($m + 1, periodic: false);
        $this->assertEqualsWithDelta($wSym[':-1']->toArray(), $wPer->toArray(), 1e-12);

        $wPer = NDArray::bohman($m, periodic: true);
        $wSym = NDArray::bohman($m + 1, periodic: false);
        $this->assertEqualsWithDelta($wSym[':-1']->toArray(), $wPer->toArray(), 1e-12);

        $wPer = NDArray::boxcar($m, periodic: true);
        $wSym = NDArray::boxcar($m + 1, periodic: false);
        $this->assertEqualsWithDelta($wSym[':-1']->toArray(), $wPer->toArray(), 1e-12);

        $beta = 8.6;
        $wPer = NDArray::kaiser($m, $beta, periodic: true);
        $wSym = NDArray::kaiser($m + 1, $beta, periodic: false);
        $this->assertEqualsWithDelta($wSym[':-1']->toArray(), $wPer->toArray(), 1e-10);

        $wPer = NDArray::lanczos($m, periodic: true);
        $wSym = NDArray::lanczos($m + 1, periodic: false);
        $this->assertEqualsWithDelta($wSym[':-1']->toArray(), $wPer->toArray(), 1e-12);

        $wPer = NDArray::triang($m, periodic: true);
        $wSym = NDArray::triang($m + 1, periodic: false);
        $this->assertEqualsWithDelta($wSym[':-1']->toArray(), $wPer->toArray(), 1e-12);
    }
}
