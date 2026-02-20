<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray;

use PhpMlKit\NDArray\Exceptions\IndexException;

/**
 * Value object representing a slice operation (start:stop:step).
 *
 * Handles parsing of string slice syntax (e.g., "0:5", "::2", ":") and
 * resolution of indices against a specific dimension size, following
 * NumPy-compatible rules.
 */
final class Slice
{
    public function __construct(
        public readonly ?int $start,
        public readonly ?int $stop,
        public readonly int $step = 1
    ) {
        if ($step === 0) {
            throw new IndexException('Slice step cannot be zero');
        }
        if ($step < 0) {
            throw new IndexException('Negative slice steps are not yet supported');
        }
    }

    /**
     * Parse a slice string into a Slice object.
     *
     * Formats:
     *   ":"       -> start=null, stop=null, step=1
     *   "i:j"     -> start=i, stop=j, step=1
     *   "i:j:k"   -> start=i, stop=j, step=k
     *   "::k"     -> start=null, stop=null, step=k
     */
    public static function parse(string $spec): self
    {
        $parts = explode(':', $spec);
        $count = count($parts);

        if ($count > 3) {
            throw new IndexException("Invalid slice syntax '$spec': too many colons");
        }

        // Helper to convert empty strings to null, numeric strings to int
        $toInt = fn(string $s, string $ctx): ?int => 
            $s === '' ? null : (
                is_numeric($s) ? (int)$s : throw new IndexException("Invalid slice component '$s' in '$spec'")
            );

        $start = $toInt(trim($parts[0]), 'start');
        $stop = isset($parts[1]) ? $toInt(trim($parts[1]), 'stop') : null;
        $step = isset($parts[2]) ? $toInt(trim($parts[2]), 'step') : 1;

        // If step is not provided in string, default is 1
        return new self($start, $stop, $step ?? 1);
    }

    /**
     * Resolve slice indices against a dimension size.
     *
     * Returns the concrete start, stop, step, and the number of elements (shape).
     *
     * @param int $dimSize Size of the dimension
     * @return array{start: int, stop: int, step: int, shape: int}
     */
    public function resolve(int $dimSize): array
    {
        $step = $this->step;

        // Default start
        $start = $this->start ?? 0;

        // Default stop
        $stop = $this->stop ?? $dimSize;

        // Handle negative indices
        if ($start < 0) {
            $start += $dimSize;
        }
        if ($stop < 0) {
            $stop += $dimSize;
        }

        // Clamp indices to bounds [0, dimSize]
        // NumPy behavior: slices clamp, they don't throw out-of-bounds
        $start = max(0, min($dimSize, $start));
        $stop = max(0, min($dimSize, $stop));

        // If start >= stop (with positive step), result is empty
        if ($start >= $stop) {
            return ['start' => $start, 'stop' => $start, 'step' => $step, 'shape' => 0];
        }

        // Calculate number of steps
        // shape = ceil((stop - start) / step)
        $diff = $stop - $start;
        $shape = (int)ceil($diff / $step);

        return [
            'start' => $start,
            'stop' => $stop,
            'step' => $step,
            'shape' => $shape,
        ];
    }
}
