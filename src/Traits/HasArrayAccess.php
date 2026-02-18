<?php

declare(strict_types=1);

namespace NDArray\Traits;

use NDArray\Exceptions\IndexException;
use NDArray\NDArray;

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
            if ($this->isSlice($offset)) {
                $this->slice($this->parseSelectors($offset));
                return true;
            }

            $indices = $this->parseOffset($offset);
        } catch (IndexException) {
            return false;
        }

        foreach ($indices as $dim => $index) {
            if ($dim >= $this->ndim) {
                return false;
            }
            // Handle negative indices: -1 means last element
            $dimSize = $this->shape[$dim];
            if ($index < 0) {
                $index = $dimSize + $index;
            }
            if ($index < 0 || $index >= $dimSize) {
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
        if ($this->isSlice($offset)) {
            return $this->slice($this->parseSelectors($offset));
        }

        $indices = $this->parseOffset($offset);

        return $this->get(...$indices);
    }

    /**
     * Set element at offset.
     *
     * Supports:
     * - Scalar assignment: $arr[0,0] = 5
     * - Slice assignment: $arr['0:2'] = $other
     * - Partial assignment: $arr[0] = $row
     *
     * @param int|string $offset
     * @param mixed $value
     */
    public function offsetSet(mixed $offset, mixed $value): void
    {
        if (is_array($value)) {
            $value = NDArray::array($value, $this->dtype);
        }

        if ($this->isSlice($offset)) {
            $view = $this->slice($this->parseSelectors($offset));
            $view->assign($value);
            return;
        }

        $indices = $this->parseOffset($offset);

        if (count($indices) === $this->ndim) {
            if (!is_scalar($value)) {
                throw new IndexException("Cannot assign array to scalar index");
            }
            $this->set($indices, $value);
        } else {
            $view = $this->get(...$indices);
            if ($view instanceof self) {
                $view->assign($value);
            } else {
                throw new IndexException("Unexpected scalar return for partial index");
            }
        }
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
     * Check if offset is a slice syntax string.
     */
    private function isSlice(mixed $offset): bool
    {
        return is_string($offset) && (str_contains($offset, ':') || str_contains($offset, '...'));
    }

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
            $parts = explode(',', $offset);
            $indices = [];
            foreach ($parts as $part) {
                $trimmed = trim($part);
                if (!is_numeric($trimmed)) {
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

    /**
     * Parse a slice string into selectors (int or string).
     *
     * @param string $offset e.g. "0:5, 1" or "..., 1:3"
     * @return array<int|string>
     */
    private function parseSelectors(string $offset): array
    {
        $parts = explode(',', $offset);
        $selectors = [];

        foreach ($parts as $part) {
            $part = trim($part);
            if ($part === '...' || $part === '…') {
                $selectors[] = '...';
            } elseif (str_contains($part, ':')) {
                $selectors[] = $part;
            } elseif (is_numeric($part)) {
                $selectors[] = (int)$part;
            } else {
                throw new IndexException("Invalid slice selector: '$part'");
            }
        }

        return $selectors;
    }
}
