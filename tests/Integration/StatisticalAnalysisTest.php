<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Integration;

use PhpMlKit\NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Integration tests for statistical analysis workflows.
 *
 * @internal
 *
 * @coversNothing
 */
final class StatisticalAnalysisTest extends TestCase
{
    public function testCorrelationMatrix(): void
    {
        $data = NDArray::array([
            [1, 2, 3],
            [2, 4, 6],
            [3, 6, 9],
            [4, 8, 12],
        ]);

        $mean = $data->mean(axis: 0);
        $centered = $data->subtract($mean);
        $n = $data->shape()[0];
        $covariance = $centered->transpose()->matmul($centered)->divide($n - 1);

        $variances = $covariance->diagonal();
        $this->assertGreaterThan(0, $variances->getAt(0));
        $this->assertGreaterThan(0, $variances->getAt(1));
        $this->assertGreaterThan(0, $variances->getAt(2));
    }

    public function testOutlierDetectionUsingZScore(): void
    {
        $data = NDArray::array([1, 2, 3, 4, 5, 100]);

        $mean = $data->mean();
        $std = $data->std();
        $zScores = $data->subtract($mean)->divide($std)->abs();
        $isOutlier = $zScores->gt(1.5);

        $this->assertFalse((bool) $isOutlier->getAt(0));
        $this->assertFalse((bool) $isOutlier->getAt(4));
        $this->assertTrue((bool) $isOutlier->getAt(5));
    }
}
