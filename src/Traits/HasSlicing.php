<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Traits;

use PhpMlKit\NDArray\ArrayMetadata;
use PhpMlKit\NDArray\Complex;
use PhpMlKit\NDArray\Exceptions\IndexException;
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
        $newOffset = $this->offset();

        foreach ($selection as $dim => $selector) {
            $dimSize = $this->shape()[$dim];
            $stride = $this->strides()[$dim];

            if (\is_string($selector) && !str_contains($selector, ':') && is_numeric($selector)) {
                $selector = (int) $selector;
            }

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
                    'Invalid slice selector type: '.get_debug_type($selector)
                );
            }
        }

        $root = $this->base ?? $this;

        return new self(
            handle: $this->handle,
            meta: new ArrayMetadata($newShape, $newStrides, $newOffset),
            dtype: $this->dtype,
            base: $root,
        );
    }

    /**
     * Assign values to the current array/view.
     *
     * Supports scalar assignment (fill) or NDArray assignment (rhs is broadcast to this view’s shape when compatible).
     *
     * @param bool|Complex|float|int|NDArray $value Scalar value or NDArray
     */
    public function assign(bool|Complex|float|int|NDArray $value): void
    {
        $lib = Lib::get();
        $src = $value;

        if ($value instanceof NDArray) {
            $srcMeta = $src->meta()->toCData();
            $dstMeta = $this->meta()->toCData();

            $status = $lib->ndarray_assign($this->handle, Lib::addr($dstMeta), $src->handle(), Lib::addr($srcMeta));
        } else {
            $cValue = $this->dtype->createCValue($src);

            $meta = $this->meta()->toCData();
            $status = $lib->ndarray_fill($this->handle, Lib::addr($meta), Lib::addr($cValue));
        }

        $lib->checkStatus($status);
    }
}
