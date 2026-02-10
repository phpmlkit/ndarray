<?php

declare(strict_types=1);

namespace NDArray\Traits;

use NDArray\DType;
use NDArray\Exceptions\IndexException;
use NDArray\Exceptions\ShapeException;
use NDArray\FFI\Lib;
use NDArray\NDArray;
use NDArray\Slice;

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
     * @return self A new view sharing the same data
     */
    public function slice(array $selection): self
    {
        $count = count($selection);
        if ($count > $this->ndim) {
            throw new IndexException(
                "Too many indices for slice: got $count, expected <= {$this->ndim}"
            );
        }

        // Pad selection with full slices for missing dimensions
        if ($count < $this->ndim) {
            $selection = array_merge(
                $selection,
                array_fill(0, $this->ndim - $count, ':')
            );
        }

        $newShape = [];
        $newStrides = [];
        $newOffset = $this->offset;

        foreach ($selection as $dim => $selector) {
            $dimSize = $this->shape[$dim];
            $stride = $this->strides[$dim];

            if (is_int($selector)) {
                // Integer index: reduces dimension
                if ($selector < 0) {
                    $selector += $dimSize;
                }
                if ($selector < 0 || $selector >= $dimSize) {
                    throw new IndexException(
                        "Index $selector out of bounds for axis $dim with size $dimSize"
                    );
                }
                $newOffset += $selector * $stride;
            } elseif (is_string($selector)) {
                // Slice string: keeps dimension (potentially resized)
                $slice = Slice::parse($selector);
                $res = $slice->resolve($dimSize);

                $newOffset += $res['start'] * $stride;
                $newShape[] = $res['shape'];
                $newStrides[] = $stride * $res['step'];
            } else {
                throw new IndexException(
                    "Invalid slice selector type: " . get_debug_type($selector)
                );
            }
        }

        // Determine base: if this is already a view, use its base; otherwise this is the base
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
        if (is_scalar($value)) {
            $this->fill($value);
            return;
        }

        if ($value instanceof NDArray) {
            $this->assignFromNDArray($value);
            return;
        }

        if (is_array($value)) {
            $this->assignFromArray($value);
            return;
        }

        throw new \InvalidArgumentException(
            "Unsupported value type for assignment: " . get_debug_type($value)
        );
    }

    /**
     * Fill the array with a scalar value using efficient Rust backend.
     */
    private function fill(mixed $value): void
    {
        $ffi = Lib::get();
        
        $cShape = Lib::createCArray('size_t', $this->shape);
        $cStrides = Lib::createCArray('size_t', $this->strides);
        
        if ($this->dtype === DType::Bool) {
            $value = $value ? 1 : 0;
        }

        $funcName = "ndarray_fill_{$this->dtype->name()}";
        
        $status = $ffi->$funcName(
            $this->handle,
            $value,
            $this->offset,
            $cShape,
            $cStrides,
            $this->ndim
        );
        
        Lib::checkStatus($status);
    }

    /**
     * Assign from another NDArray using efficient Rust backend.
     */
    private function assignFromNDArray(NDArray $value): void
    {
        if ($value->size() !== $this->size) {
            throw new ShapeException(
                "Cannot assign array of size {$value->size()} to view of size {$this->size}"
            );
        }

        // FFI call
        $ffi = Lib::get();

        $dstShape = Lib::createCArray('size_t', $this->shape);
        $dstStrides = Lib::createCArray('size_t', $this->strides);

        // If shapes differ but sizes match, we treat source as if reshaped to destination shape
        // For strict correctness, we pass source shape/strides as is, but our Rust `impl_assign_slice`
        // assumes it can iterate/zip. If we want flattened assignment, we need to handle that.
        // But `ndarray` assign requires broadcast compatibility.
        // If shapes match, great. If not, we might need to reshape the source view temporarily?
        // Actually, if we just want to copy data in order, we can reshape source to destination shape locally.
        
        $src = $value;
        if ($src->shape() !== $this->shape) {
            // Try to reshape source to match destination shape (if size matches, which we checked)
            // If source is not contiguous, reshape might fail/copy.
            // But we need compatible shapes for `assign`.
            if ($src->isContiguous()) {
                $src = $src->reshape($this->shape);
            } else {
                // If not contiguous, we copy to make it reshape-able
                $src = $src->copy()->reshape($this->shape);
            }
        }

        $srcShape = Lib::createCArray('size_t', $src->shape());
        $srcStrides = Lib::createCArray('size_t', $src->strides());

        $funcName = "ndarray_assign_{$this->dtype->name()}";

        $status = $ffi->$funcName(
            $this->handle,
            $this->offset,
            $dstShape,
            $dstStrides,
            $src->getHandle(),
            // We need src offset. NDArray doesn't expose public offset?
            // Wait, `offset` is private in NDArray. 
            // I need to access it. But I'm in a trait used by NDArray.
            // `$src` is an instance of NDArray. Private properties of other instances of the same class
            // ARE accessible in PHP.
            $src->offset,
            $srcShape,
            $srcStrides,
            $this->ndim
        );

        Lib::checkStatus($status);
    }

    /**
     * Assign from PHP array by converting to temporary NDArray.
     */
    private function assignFromArray(array $value): void
    {
        // Create temporary array
        $tmp = NDArray::array($value, $this->dtype);
        
        // Use NDArray assignment
        $this->assignFromNDArray($tmp);
    }
}
