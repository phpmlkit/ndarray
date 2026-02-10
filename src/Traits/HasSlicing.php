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
        // 1. Scalar assignment (fill)
        if (is_scalar($value)) {
            $this->fill($value);
            return;
        }

        // 2. NDArray assignment
        if ($value instanceof NDArray) {
            if ($value->size() !== $this->size) {
                throw new ShapeException(
                    "Cannot assign array of size {$value->size()} to view of size {$this->size}"
                );
            }
            // For now, convert to flat array and assign.
            // TODO: Implement optimized C-side copy/broadcast when available.
            $flatData = $value->toArray(); // Takes into account source strides/views
            $this->assignFlat($flatData);
            return;
        }

        // 3. PHP Array assignment
        if (is_array($value)) {
            // Flatten the input array to match the view's iteration order
            // Note: This assumes the input array structure matches the view's shape
            // or is at least compatible in total element count.
            $flatData = $this->flattenPhpArray($value);
            
            if (count($flatData) !== $this->size) {
                throw new ShapeException(
                    "Input array has " . count($flatData) . " elements, expected {$this->size}"
                );
            }
            $this->assignFlat($flatData);
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
        
        // Handle boolean conversion
        if ($this->dtype === DType::Bool) {
            $value = $value ? 1 : 0;
        }

        $funcName = "ndarray_fill_{$this->dtype->name()}";
        
        // FFI call to fill the view
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
     * Assign flat data to the array in row-major order.
     */
    private function assignFlat(array $flatData): void
    {
        // TODO: Move this to Rust via ndarray_assign
        $i = 0;
        $this->mapInPlace(function() use (&$flatData, &$i) {
            return $flatData[$i++] ?? 0;
        });
    }

    /**
     * Internal helper to iterate over all elements and set values.
     * 
     * @param callable $callback fn(current_indices) -> value_to_set
     */
    private function mapInPlace(callable $callback): void
    {
        // We need to iterate over all indices of the current view.
        // We'll reuse the recursive logic similar to offset calculation.
        
        $iterate = function (array $currentIndices, int $dim) use (&$iterate, $callback) {
            if ($dim === $this->ndim) {
                // Leaf node: set value
                $val = $callback();
                $this->set($currentIndices, $val);
                return;
            }

            for ($i = 0; $i < $this->shape[$dim]; $i++) {
                $currentIndices[$dim] = $i;
                $iterate($currentIndices, $dim + 1);
            }
        };

        $iterate(array_fill(0, $this->ndim, 0), 0);
    }

    /**
     * Flatten a nested PHP array into a flat list.
     */
    private function flattenPhpArray(array $array): array
    {
        $result = [];
        array_walk_recursive($array, function ($a) use (&$result) {
            $result[] = $a;
        });
        return $result;
    }
}
