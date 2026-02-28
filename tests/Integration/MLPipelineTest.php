<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Integration;

use PhpMlKit\NDArray\NDArray;
use PHPUnit\Framework\TestCase;

/**
 * End-to-end integration test for ML pipeline.
 *
 * @internal
 *
 * @coversNothing
 */
final class MLPipelineTest extends TestCase
{
    /**
     * Simple linear regression pipelines with batch gradient descent.
     */
    public function testFullMLPipeline(): void
    {
        $X = NDArray::array([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ]);
        $y = NDArray::array([3.0, 5.0, 7.0, 9.0]);

        $XMean = $X->mean(axis: 0);
        $XStd = $X->std(axis: 0);
        $XNormalized = $X->subtract($XMean)->divide($XStd);

        $weights = NDArray::array([0.0, 0.0]);
        $bias = 0.0;
        $learningRate = 0.01;

        for ($epoch = 0; $epoch < 500; ++$epoch) {
            $predictions = $XNormalized->matmul($weights->reshape([2, 1]))->flatten()->add($bias);
            $error = $predictions->subtract($y);
            $gradWeights = $XNormalized->transpose()->matmul($error->reshape([4, 1]))->flatten()->divide(4);
            $gradBias = $error->mean();

            $weights = $weights->subtract($gradWeights->multiply($learningRate));
            $bias -= $gradBias * $learningRate;
        }

        $XNew = NDArray::array([[2.5, 3.5]]);
        $XNewNormalized = $XNew->subtract($XMean)->divide($XStd);
        $prediction = $XNewNormalized->matmul($weights->reshape([2, 1]))->flatten()->add($bias);

        $this->assertEqualsWithDelta(6.0, $prediction->getAt(0), 0.05);
    }
}
