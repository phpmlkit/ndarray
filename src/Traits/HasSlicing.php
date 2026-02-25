<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Traits;

use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\Exceptions\IndexException;
use PhpMlKit\NDArray\Exceptions\ShapeException;
use PhpMlKit\NDArray\FFI\Lib;
use PhpMlKit\NDArray\NDArray;
use PhpMlKit\NDArray\Slice;

/**
 * Slicing operations: slice() and assign().
 *
 * Slices return views sharing the same underlying Rust handle.
 */
trait HasSlicing
{
    /**
     * Create a view of the array using slice syntax.
     *
     * Selectors can be integers (to reduce dimensions) or slice strings
     * (e.g., "0:5", ":", "::2") to subset dimensions.
     *
     * @param array<int|string> $selection List of selectors for each dimension
     *
     * @return self A new view sharing the same data
     */
    public function slice(array $selection): self
    {
        $count = \count($selection);

        $ellipsisPos = null;
        foreach ($selection as $i => $sel) {
            if ('...' === $sel || '…' === $sel) {
                if (null !== $ellipsisPos) {
                    throw new IndexException('Only one ellipsis (...) allowed per slice');
                }
                $ellipsisPos = $i;
            }
        }

        if (null !== $ellipsisPos) {
            $nonEllipsisCount = $count - 1;
            $ellipsisDims = $this->ndim() - $nonEllipsisCount;

            if ($ellipsisDims < 0) {
                throw new IndexException(
                    "Too many indices for slice: ellipsis expansion would exceed {$this->ndim()} dimensions"
                );
            }

            $before = \array_slice($selection, 0, $ellipsisPos);
            $ellipsisFill = array_fill(0, $ellipsisDims, ':');
            $after = \array_slice($selection, $ellipsisPos + 1);
            $selection = array_merge($before, $ellipsisFill, $after);
            $count = \count($selection);
        }

        if ($count > $this->ndim()) {
            throw new IndexException(
                "Too many indices for slice: got {$count}, expected <= {$this->ndim()}"
            );
        }

        if ($count < $this->ndim()) {
            $selection = array_merge(
                $selection,
                array_fill(0, $this->ndim() - $count, ':')
            );
        }

        $newShape = [];
        $newStrides = [];
        $newOffset = $this->getOffset();

        foreach ($selection as $dim => $selector) {
            $dimSize = $this->shape()[$dim];
            $stride = $this->strides()[$dim];

            if (\is_int($selector)) {
                if ($selector < 0) {
                    $selector += $dimSize;
                }
                if ($selector < 0 || $selector >= $dimSize) {
                    throw new IndexException(
                        "Index {$selector} out of bounds for axis {$dim} with size {$dimSize}"
                    );
                }
                $newOffset += $selector * $stride;
            } elseif (\is_string($selector)) {
                $slice = Slice::parse($selector);
                $res = $slice->resolve($dimSize);

                $newOffset += $res['start'] * $stride;
                $newShape[] = $res['shape'];
                $newStrides[] = $stride * $res['step'];
            } else {
                throw new IndexException(
                    'Invalid slice selector type: ' . get_debug_type($selector)
                );
            }
        }

        $root = $this->base ?? $this;

        return new self(
            handle: $this->handle,
            shape: $newShape,
            dtype: $this->dtype,
            strides: $newStrides,
            offset: $newOffset,
            base: $root,
        );
    }

    /**
     * Assign values to the current array/view.
     *
     * Supports scalar assignment (fill) or array assignment (from PHP array or NDArray).
     *
     * @param mixed $value Scalar value or array/NDArray
     */
    public function assign(mixed $value): void
    {
        if (\is_scalar($value)) {
            $this->fill($value);

            return;
        }

        if ($value instanceof NDArray) {
            $this->assignFromNDArray($value);

            return;
        }

        throw new \InvalidArgumentException(
            'Assignment value must be scalar or NDArray, got ' . get_debug_type($value)
        );
    }

    /**
     * Fill the array with a scalar value.
     *
     * @param mixed $value Scalar value
     */
    private function fill(mixed $value): void
    {
        $ffi = Lib::get();
        $cValue = $this->dtype->createCValue($value);

        $meta = $this->viewMetadata()->toCData();
        $status = $ffi->ndarray_fill($this->handle, Lib::addr($meta), Lib::addr($cValue));

        Lib::checkStatus($status);
    }

    /**
     * Assign values from an NDArray to the current array/view.
     *
     * @param NDArray $value Source NDArray
     */
    private function assignFromNDArray(NDArray $value): void
    {
        if ($value->size() !== $this->size) {
            throw new ShapeException(
                "Cannot assign array of size {$value->size()} to view of size {$this->size}"
            );
        }

        $ffi = Lib::get();
        $dstMeta = $this->viewMetadata()->toCData();
        $src = $value;
        if ($src->shape() !== $this->shape()) {
            if ($src->isContiguous()) {
                $src = $src->reshape($this->shape());
            } else {
                $src = $src->copy()->reshape($this->shape());
            }
        }
        $srcMeta = $src->viewMetadata()->toCData();

        $status = $ffi->ndarray_assign(
            $this->handle,
            Lib::addr($dstMeta),
            $src->handle(),
            Lib::addr($srcMeta)
        );

        Lib::checkStatus($status);
    }
}
