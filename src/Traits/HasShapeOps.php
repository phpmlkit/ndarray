<?php

declare(strict_types=1);

namespace NDArray\Traits;

use NDArray\PadMode;
use NDArray\Exceptions\ShapeException;
use NDArray\FFI\Lib;
use NDArray\NDArray;

/**
 * Shape manipulation operations.
 *
 * Provides reshape(), transpose(), flatten(), squeeze(),
 * expand_dims(), ravel(), swap_axes(), etc.
 */
trait HasShapeOps
{
    /**
     * Pad an array.
     *
     * Supported modes:
     * - Constant (default)
     * - Symmetric
     * - Reflect
     *
     * @param int|array<int>|array<array<int>> $padWidth
     * @param PadMode $mode
     * @param int|float|bool|array<int|float|bool> $constantValues
     * @return NDArray
     */
    public function pad(
        int|array $padWidth,
        PadMode $mode = PadMode::Constant,
        int|float|bool|array $constantValues = 0
    ): NDArray {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

        $normalizedPad = $this->normalizePadWidth($padWidth);
        $padFlat = [];
        foreach ($normalizedPad as [$before, $after]) {
            $padFlat[] = $before;
            $padFlat[] = $after;
        }

        $constants = $this->normalizePadConstants($constantValues, $mode);
        $constantsC = Lib::createCArray('double', $constants);

        $status = $ffi->ndarray_pad(
            $this->handle,
            $this->offset,
            Lib::createShapeArray($this->shape),
            Lib::createCArray('size_t', $this->strides),
            $this->ndim,
            Lib::createShapeArray($padFlat),
            $mode->value,
            $constantsC,
            count($constants),
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        $newShape = [];
        foreach ($this->shape as $axis => $dim) {
            [$before, $after] = $normalizedPad[$axis];
            $newShape[] = $dim + $before + $after;
        }

        return new NDArray($outHandle, $newShape, $this->dtype);
    }

    /**
     * Reshape the array to a new shape.
     *
     * Returns a new array with the specified shape.
     * Supports both C-order (row-major, order='C') and F-order (column-major, order='F').
     *
     * @param array<int> $newShape New shape
     * @param string $order Memory layout: 'C' for row-major, 'F' for column-major
     * @return NDArray
     */
    public function reshape(array $newShape, string $order = 'C'): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

        $cShape = Lib::createShapeArray($newShape);
        $orderCode = $order === 'F' ? 1 : 0; // 0=C (RowMajor), 1=F (ColumnMajor)

        $status = $ffi->ndarray_reshape(
            $this->handle,
            $this->offset,
            Lib::createShapeArray($this->shape),
            Lib::createCArray('size_t', $this->strides),
            $this->ndim,
            $cShape,
            count($newShape),
            $orderCode,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new NDArray($outHandle, $newShape, $this->dtype);
    }

    /**
     * Transpose the array.
     *
     * For 2D arrays, swaps rows and columns.
     * For nD arrays, reverses the order of all axes.
     *
     * @return NDArray
     */
    public function transpose(): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

        $status = $ffi->ndarray_transpose(
            $this->handle,
            $this->offset,
            Lib::createShapeArray($this->shape),
            Lib::createCArray('size_t', $this->strides),
            $this->ndim,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        // Compute new shape (reversed)
        $newShape = array_reverse($this->shape);

        return new NDArray($outHandle, $newShape, $this->dtype);
    }

    /**
     * Swap two axes of the array.
     *
     * @param int $axis1 First axis to swap
     * @param int $axis2 Second axis to swap
     * @return NDArray
     */
    public function swapAxes(int $axis1, int $axis2): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

        $status = $ffi->ndarray_swap_axes(
            $this->handle,
            $this->offset,
            Lib::createShapeArray($this->shape),
            Lib::createCArray('size_t', $this->strides),
            $this->ndim,
            $axis1,
            $axis2,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        // Compute new shape
        $newShape = $this->shape;
        $temp = $newShape[$axis1];
        $newShape[$axis1] = $newShape[$axis2];
        $newShape[$axis2] = $temp;

        return new NDArray($outHandle, $newShape, $this->dtype);
    }

    /**
     * Permute axes of the array.
     *
     * Reorders the axes according to the given permutation.
     * For example, permuteAxes([1, 0]) on a 2D array is equivalent to transpose().
     *
     * @param array<int> $axes New order of axes
     * @return NDArray
     */
    public function permuteAxes(array $axes): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

        // Validate axes
        if (count($axes) !== $this->ndim) {
            throw new ShapeException("permute_axes requires {$this->ndim} axes, got " . count($axes));
        }

        // Convert to 0-indexed if negative indices provided
        $normalizedAxes = [];
        foreach ($axes as $axis) {
            if ($axis < 0) {
                $axis = $this->ndim + $axis;
            }
            if ($axis < 0 || $axis >= $this->ndim) {
                throw new ShapeException("Axis $axis is out of bounds for array with {$this->ndim} dimensions");
            }
            $normalizedAxes[] = $axis;
        }

        // Check for duplicates
        if (count(array_unique($normalizedAxes)) !== count($normalizedAxes)) {
            throw new ShapeException("Duplicate axes in permutation");
        }

        $status = $ffi->ndarray_permute_axes(
            $this->handle,
            $this->offset,
            Lib::createShapeArray($this->shape),
            Lib::createCArray('size_t', $this->strides),
            $this->ndim,
            Lib::createShapeArray($normalizedAxes),
            count($normalizedAxes),
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        // Compute new shape
        $newShape = [];
        foreach ($normalizedAxes as $axis) {
            $newShape[] = $this->shape[$axis];
        }

        return new NDArray($outHandle, $newShape, $this->dtype);
    }

    /**
     * Merge axes by combining take into into.
     *
     * If possible, merge in the axis take to into. Returns the merged array.
     *
     * @param int $take Axis to merge from
     * @param int $into Axis to merge into
     * @return NDArray
     */
    public function mergeAxes(int $take, int $into): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

        $status = $ffi->ndarray_merge_axes(
            $this->handle,
            $this->offset,
            Lib::createShapeArray($this->shape),
            Lib::createCArray('size_t', $this->strides),
            $this->ndim,
            $take,
            $into,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        // Compute new shape (product of merged axes)
        $newShape = $this->shape;
        $newShape[$into] = $newShape[$take] * $newShape[$into];
        array_splice($newShape, $take, 1);

        return new NDArray($outHandle, $newShape, $this->dtype);
    }

    /**
     * Reverse the stride of an axis.
     *
     * @param int $axis Axis to invert
     * @return NDArray
     */
    public function invertAxis(int $axis): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

        // Handle negative axis
        if ($axis < 0) {
            $axis = $this->ndim + $axis;
        }

        // Validate axis
        if ($axis < 0 || $axis >= $this->ndim) {
            throw new ShapeException("Axis $axis is out of bounds for array with {$this->ndim} dimensions");
        }

        $status = $ffi->ndarray_invert_axis(
            $this->handle,
            $this->offset,
            Lib::createShapeArray($this->shape),
            Lib::createCArray('size_t', $this->strides),
            $this->ndim,
            $axis,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new NDArray($outHandle, $this->shape, $this->dtype);
    }

    /**
     * Insert a new axis at the specified position.
     *
     * The new axis always has length 1.
     *
     * @param int $axis Position where new axis is inserted
     * @return NDArray
     */
    public function insertAxis(int $axis): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

        // Handle negative axis
        if ($axis < 0) {
            $axis = $this->ndim + $axis + 1;
        }

        // Validate axis
        if ($axis < 0 || $axis > $this->ndim) {
            throw new ShapeException("Axis $axis is out of bounds for array with {$this->ndim} dimensions");
        }

        $status = $ffi->ndarray_insert_axis(
            $this->handle,
            $this->offset,
            Lib::createShapeArray($this->shape),
            Lib::createCArray('size_t', $this->strides),
            $this->ndim,
            $axis,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        // Compute new shape
        $newShape = $this->shape;
        array_splice($newShape, $axis, 0, [1]);

        return new NDArray($outHandle, $newShape, $this->dtype);
    }

    /**
     * Flatten the array to 1D.
     *
     * Always returns a copy in C-order (row-major).
     *
     * @return NDArray
     */
    public function flatten(): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

        $status = $ffi->ndarray_flatten(
            $this->handle,
            $this->offset,
            Lib::createShapeArray($this->shape),
            Lib::createCArray('size_t', $this->strides),
            $this->ndim,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new NDArray($outHandle, [$this->size], $this->dtype);
    }

    /**
     * Ravel the array to 1D.
     *
     * Similar to flatten() but may return a view if the array is contiguous.
     *
     * @param string $order Memory layout: 'C' for row-major, 'F' for column-major
     * @return NDArray
     */
    public function ravel(string $order = 'C'): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

        $orderCode = $order === 'F' ? 1 : 0;

        $status = $ffi->ndarray_ravel(
            $this->handle,
            $this->offset,
            Lib::createShapeArray($this->shape),
            Lib::createCArray('size_t', $this->strides),
            $this->ndim,
            $orderCode,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new NDArray($outHandle, [$this->size], $this->dtype);
    }

    /**
     * Remove axes of length 1 from the array.
     *
     * If no axes are specified, removes all length-1 axes.
     *
     * @param array<int>|null $axes Specific axes to squeeze (null for all)
     * @return NDArray
     */
    public function squeeze(?array $axes = null): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

        if ($axes === null) {
            // Squeeze all length-1 axes
            $cAxes = null;
            $numAxes = 0;
            
            // Compute new shape
            $newShape = array_values(array_filter($this->shape, fn($dim) => $dim !== 1));
            
            // Ensure at least 1 dimension
            if (empty($newShape)) {
                $newShape = [1];
            }
        } else {
            // Squeeze specific axes
            $cAxes = Lib::createShapeArray($axes);
            $numAxes = count($axes);
            
            // Compute new shape
            $newShape = [];
            foreach ($this->shape as $i => $dim) {
                if (!in_array($i, $axes, true)) {
                    $newShape[] = $dim;
                }
            }
        }

        $status = $ffi->ndarray_squeeze(
            $this->handle,
            $this->offset,
            Lib::createShapeArray($this->shape),
            Lib::createCArray('size_t', $this->strides),
            $this->ndim,
            $cAxes,
            $numAxes,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new NDArray($outHandle, $newShape, $this->dtype);
    }

    /**
     * Expand dimensions by inserting a new axis.
     *
     * Equivalent to NumPy's expand_dims.
     *
     * @param int $axis Position where new axis is inserted
     * @return NDArray
     */
    public function expandDims(int $axis): NDArray
    {
        return $this->insertAxis($axis);
    }

    /**
     * Static helper to compute strides from shape (C-contiguous).
     *
     * @param array<int> $shape
     * @return array<int>
     */
    private static function computeStrides(array $shape): array
    {
        $strides = [];
        $stride = 1;
        for ($i = count($shape) - 1; $i >= 0; $i--) {
            $strides[$i] = $stride;
            $stride *= $shape[$i];
        }
        return $strides;
    }

    /**
     * Normalize pad width to [[before, after], ...] for each axis.
     *
     * @param int|array $padWidth
     * @return array<array{0:int,1:int}>
     */
    private function normalizePadWidth(int|array $padWidth): array
    {
        if (is_int($padWidth)) {
            if ($padWidth < 0) {
                throw new ShapeException('padWidth must be non-negative');
            }
            return array_fill(0, $this->ndim, [$padWidth, $padWidth]);
        }

        if ($this->ndim === 0) {
            throw new ShapeException('pad() requires at least 1 dimension');
        }

        if (count($padWidth) === 2 && isset($padWidth[0], $padWidth[1]) && !is_array($padWidth[0])) {
            $before = (int) $padWidth[0];
            $after = (int) $padWidth[1];
            if ($before < 0 || $after < 0) {
                throw new ShapeException('padWidth values must be non-negative');
            }
            return array_fill(0, $this->ndim, [$before, $after]);
        }

        if (count($padWidth) === $this->ndim) {
            $normalized = [];
            foreach ($padWidth as $axis => $entry) {
                if (is_int($entry)) {
                    if ($entry < 0) {
                        throw new ShapeException("padWidth for axis $axis must be non-negative");
                    }
                    $normalized[] = [$entry, $entry];
                    continue;
                }

                if (!is_array($entry) || count($entry) !== 2) {
                    throw new ShapeException("padWidth for axis $axis must be int or [before, after]");
                }

                $before = (int) $entry[0];
                $after = (int) $entry[1];
                if ($before < 0 || $after < 0) {
                    throw new ShapeException("padWidth for axis $axis must be non-negative");
                }
                $normalized[] = [$before, $after];
            }
            return $normalized;
        }

        throw new ShapeException(
            "padWidth must be an int, [before, after], or per-axis list of length {$this->ndim}"
        );
    }

    /**
     * Normalize constants passed to pad.
     *
     * Rust accepts:
     * - [v] for scalar constant
     * - [before, after] global pair
     * - [b0, a0, b1, a1, ...] per-axis pairs
     *
     * @param int|float|bool|array<int|float|bool> $constantValues
     * @param PadMode $mode
     * @return array<float>
     */
    private function normalizePadConstants(int|float|bool|array $constantValues, PadMode $mode): array
    {
        if ($mode !== PadMode::Constant) {
            return [0.0];
        }

        if (!is_array($constantValues)) {
            return [(float) $constantValues];
        }

        if (count($constantValues) === 0) {
            return [0.0];
        }

        if (count($constantValues) === 2 && !is_array($constantValues[0])) {
            return [(float) $constantValues[0], (float) $constantValues[1]];
        }

        $flat = [];
        foreach ($constantValues as $entry) {
            if (is_array($entry)) {
                if (count($entry) !== 2) {
                    throw new ShapeException('constantValues per-axis entries must be [before, after]');
                }
                $flat[] = (float) $entry[0];
                $flat[] = (float) $entry[1];
            } else {
                $flat[] = (float) $entry;
            }
        }

        if (count($flat) !== 1 && count($flat) !== 2 && count($flat) !== $this->ndim * 2) {
            throw new ShapeException(
                "constantValues must be scalar, [before, after], or per-axis pairs of length " . ($this->ndim * 2)
            );
        }

        return $flat;
    }
}
