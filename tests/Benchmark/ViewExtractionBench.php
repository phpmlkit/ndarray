<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Tests\Benchmark;

use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\NDArray;

/**
 * Benchmark tests for view extraction performance.
 */
class ViewExtractionBench
{
    /**
     * Run all benchmarks.
     */
    public static function run(): void
    {
        echo "=== View Extraction Performance Benchmarks ===\n\n";

        self::benchContiguousViewExtraction();
        self::benchStridedViewExtraction();
        self::benchSingleElementView();
        self::benchLargeArrayViews();
    }

    /**
     * Benchmark contiguous view extraction (should be fast).
     */
    private static function benchContiguousViewExtraction(): void
    {
        echo "Benchmark: Contiguous View Extraction\n";
        echo str_repeat('-', 50)."\n";

        $sizes = [100, 1000];

        foreach ($sizes as $size) {
            $arr = NDArray::zeros([$size, $size], DType::Float64);

            $start = microtime(true);
            for ($i = 0; $i < 100; ++$i) {
                $view = $arr->slice([':', ':']);
                $result = $view->add(1.0);
            }
            $elapsed = (microtime(true) - $start) * 1000;

            echo \sprintf("  Size %dx%d: %.3f ms (100 iterations)\n", $size, $size, $elapsed);
        }

        echo "\n";
    }

    /**
     * Benchmark strided view extraction (may be slower).
     */
    private static function benchStridedViewExtraction(): void
    {
        echo "Benchmark: Strided View Extraction\n";
        echo str_repeat('-', 50)."\n";

        $sizes = [100, 1000];

        foreach ($sizes as $size) {
            $arr = NDArray::zeros([$size, $size], DType::Float64);

            $start = microtime(true);
            for ($i = 0; $i < 100; ++$i) {
                $view = $arr->slice(['::2', '::2']);
                $result = $view->add(1.0);
            }
            $elapsed = (microtime(true) - $start) * 1000;

            echo \sprintf("  Size %dx%d (step 2): %.3f ms (100 iterations)\n", $size, $size, $elapsed);
        }

        echo "\n";
    }

    /**
     * Benchmark single element view (edge case).
     */
    private static function benchSingleElementView(): void
    {
        echo "Benchmark: Single Element View\n";
        echo str_repeat('-', 50)."\n";

        $arr = NDArray::zeros([100, 100], DType::Float64);

        $start = microtime(true);
        for ($i = 0; $i < 1000; ++$i) {
            $view = $arr->slice(['50:51', '50:51']);
            $result = $view->add(1.0);
        }
        $elapsed = (microtime(true) - $start) * 1000;

        echo \sprintf("  Single element (1000 iterations): %.3f ms\n", $elapsed);
        echo "\n";
    }

    /**
     * Benchmark large array views.
     */
    private static function benchLargeArrayViews(): void
    {
        echo "Benchmark: Large Array Views\n";
        echo str_repeat('-', 50)."\n";

        $size = 1000;
        $arr = NDArray::zeros([$size, $size, 10], DType::Float64);

        $start = microtime(true);
        for ($i = 0; $i < 10; ++$i) {
            $view = $arr->slice(['0:500', '0:500', ':']);
            $result = $view->add(1.0);
        }
        $elapsed = (microtime(true) - $start) * 1000;

        echo \sprintf("  3D array slice (10 iterations): %.3f ms\n", $elapsed);
        echo "\n";
    }
}
