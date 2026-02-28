<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Integration;

use PhpMlKit\NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * Integration tests for batch processing patterns.
 *
 * @internal
 *
 * @coversNothing
 */
final class BatchProcessingTest extends TestCase
{
    public function testBatchGradientAccumulation(): void
    {
        $gradients = [
            NDArray::array([[0.1, 0.2], [0.3, 0.4]]),
            NDArray::array([[0.2, 0.3], [0.4, 0.5]]),
            NDArray::array([[0.3, 0.4], [0.5, 0.6]]),
        ];

        $accumulated = NDArray::zeros([2, 2]);
        foreach ($gradients as $grad) {
            $accumulated = $accumulated->add($grad);
        }

        $averaged = $accumulated->divide(\count($gradients));

        $expected = [[0.2, 0.3], [0.4, 0.5]];
        $this->assertEqualsWithDelta($expected, $averaged->toArray(), 0.001);
    }

    public function testBatchPredictionAggregation(): void
    {
        $predictions = NDArray::array([
            [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7], [0.3, 0.3, 0.4]],
            [[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7], [0.4, 0.3, 0.3]],
            [[0.8, 0.1, 0.1], [0.1, 0.9, 0.0], [0.3, 0.1, 0.6], [0.2, 0.4, 0.4]],
        ]);

        $avgPredictions = $predictions->mean(axis: 0);

        $this->assertEquals([4, 3], $avgPredictions->shape());

        $sums = $avgPredictions->sum(axis: 1);
        foreach ($sums->toArray() as $sum) {
            $this->assertEqualsWithDelta(1.0, $sum, 0.001);
        }
    }

    public function testMiniBatchSplitting(): void
    {
        $dataset = NDArray::arange(0, 50)->reshape([10, 5]);

        $batchSize = 3;
        $numSamples = $dataset->shape()[0];
        $numBatches = (int) ceil($numSamples / $batchSize);

        $batches = [];
        for ($i = 0; $i < $numBatches; ++$i) {
            $start = $i * $batchSize;
            $end = min(($i + 1) * $batchSize, $numSamples);
            $batches[] = $dataset->slice(["{$start}:{$end}"]);
        }

        $this->assertEquals(3, $batches[0]->shape()[0]);
        $this->assertEquals(3, $batches[1]->shape()[0]);
        $this->assertEquals(3, $batches[2]->shape()[0]);
        $this->assertEquals(1, $batches[3]->shape()[0]);
    }
}
