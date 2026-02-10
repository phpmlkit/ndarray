<?php

declare(strict_types=1);

namespace NDArray\Traits;

use NDArray\Exceptions\ShapeException;

/**
 * Shape manipulation operations.
 *
 * Will provide reshape(), transpose(), flatten(), squeeze(),
 * expand_dims(), ravel(), swapaxes(), etc.
 */
trait HasShapeOps
{
    /**
     * Reshape the array to a new shape.
     *
     * Returns a new view if possible (i.e., if the array is C-contiguous).
     * If the array is not contiguous, this currently throws an exception.
     * Use copy()->reshape() for non-contiguous arrays.
     *
     * @param array<int> $newShape
     * @return self
     */
    public function reshape(array $newShape): self
    {
        $newSize = (int) array_product($newShape);
        if ($newSize !== $this->size) {
            throw new ShapeException(
                "Cannot reshape array of size {$this->size} into shape " . json_encode($newShape)
            );
        }

        if (!$this->isContiguous()) {
            throw new ShapeException(
                "Reshaping non-contiguous arrays (e.g. slices) is not yet supported. Use ->copy()->reshape()."
            );
        }

        // Compute new strides for C-contiguous layout
        $newStrides = self::computeStrides($newShape);

        // Determine base: if this is already a view, use its base; otherwise this is the base
        $root = $this->base ?? $this;

        return new self(
            handle: $this->handle,
            shape: $newShape,
            dtype: $this->dtype,
            strides: $newStrides,
            offset: $this->offset,
            base: $root,
        );
    }
}
