<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Integration;

use PhpMlKit\NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Integration tests for data normalization and standardization workflows.
 *
 * @internal
 *
 * @coversNothing
 */
final class DataNormalizationTest extends TestCase
{
    public function testZScoreStandardization(): void
    {
        $data = NDArray::array([
            [10, 20, 30],
            [20, 30, 40],
            [30, 40, 50],
            [40, 50, 60],
            [50, 60, 70],
        ]);

        $mean = $data->mean(axis: 0);
        $std = $data->std(axis: 0);
        $standardized = $data->subtract($mean)->divide($std);

        $newMean = $standardized->mean(axis: 0);
        $newStd = $standardized->std(axis: 0);

        foreach ($newMean->toArray() as $m) {
            $this->assertEqualsWithDelta(0.0, $m, 0.001);
        }
        foreach ($newStd->toArray() as $s) {
            $this->assertEqualsWithDelta(1.0, $s, 0.001);
        }
    }

    public function testMinMaxScaling(): void
    {
        $data = NDArray::array([
            [1, 100, 1000],
            [2, 200, 2000],
            [3, 300, 3000],
            [4, 400, 4000],
            [5, 500, 5000],
        ]);

        $min = $data->min(axis: 0);
        $max = $data->max(axis: 0);
        $range = $max->subtract($min);
        $scaled = $data->subtract($min)->divide($range);

        $this->assertEqualsWithDelta(0.0, $scaled->min(), 0.001);
        $this->assertEqualsWithDelta(1.0, $scaled->max(), 0.001);
    }

    public function testFeatureNormalization(): void
    {
        $features = NDArray::array([
            [1000, 0.5, 10],
            [2000, 1.0, 20],
            [3000, 1.5, 30],
        ]);

        $norms = $features->pow2()->sum(axis: 1, keepdims: true)->sqrt();
        $normalized = $features->divide($norms);

        $newNorms = $normalized->pow2()->sum(axis: 1)->sqrt();
        foreach ($newNorms as $norm) {
            $this->assertEqualsWithDelta(1.0, $norm, 0.001);
        }
    }
}
