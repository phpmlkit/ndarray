<?php

declare(strict_types=1);

namespace NDArray\Traits;

use NDArray\Exceptions\IndexException;

/**
 * ArrayAccess interface implementation for bracket-style element access.
 *
 * Supports:
 *   $arr[0]     — integer index (first dimension)
 *   $arr['1,2'] — comma-separated multi-index
 *   $arr['0:2'] — reserved for future slice syntax (throws)
 */
trait HasArrayAccess
{
    /**
     * Check if an offset exists.
     *
     * @param int|string $offset Integer index or comma-separated string
     */
    public function offsetExists(mixed $offset): bool
    {
        try {
            $indices = $this->parseOffset($offset);
        } catch (IndexException) {
            return false;
        }

        foreach ($indices as $dim => $index) {
            if ($dim >= $this->ndim || $index < 0 || $index >= $this->shape[$dim]) {
                return false;
            }
        }

        return true;
    }

    /**
     * Get element or sub-array at offset.
     *
     * @param int|string $offset
     * @return self|int|float|bool
     */
    public function offsetGet(mixed $offset): self|int|float|bool
    {
        $indices = $this->parseOffset($offset);

        return $this->get(...$indices);
    }

    /**
     * Set element at offset (full indexing only).
     *
     * @param int|string $offset
     * @param int|float|bool $value
     */
    public function offsetSet(mixed $offset, mixed $value): void
    {
        $indices = $this->parseOffset($offset);

        if (count($indices) !== $this->ndim) {
            throw new IndexException(
                "ArrayAccess set requires exactly {$this->ndim} indices for a scalar assignment"
            );
        }

        $this->set($indices, $value);
    }

    /**
     * Unsetting array elements is not supported.
     *
     * @param int|string $offset
     */
    public function offsetUnset(mixed $offset): void
    {
        throw new IndexException('Cannot unset NDArray elements');
    }

    // =========================================================================
    // Private Helpers
    // =========================================================================

    /**
     * Parse an ArrayAccess offset into integer indices.
     *
     * @param int|string $offset
     * @return array<int>
     * @throws IndexException
     */
    private function parseOffset(mixed $offset): array
    {
        if (is_int($offset)) {
            return [$offset];
        }

        if (is_string($offset)) {
            // Check for slice syntax (reserved for future)
            if (str_contains($offset, ':')) {
                throw new IndexException(
                    "Slice syntax ('$offset') is not yet supported. Use get() for indexing."
                );
            }

            // Comma-separated multi-index: "1,2" -> [1, 2]
            $parts = explode(',', $offset);
            $indices = [];
            foreach ($parts as $part) {
                $trimmed = trim($part);
                if (!is_numeric($trimmed) || str_contains($trimmed, '.')) {
                    throw new IndexException("Invalid index component: '$trimmed'");
                }
                $indices[] = (int) $trimmed;
            }

            return $indices;
        }

        throw new IndexException(
            sprintf("Invalid offset type: expected int or string, got %s", get_debug_type($offset))
        );
    }
}
