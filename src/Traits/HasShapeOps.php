<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Traits;

use PhpMlKit\NDArray\Exceptions\ShapeException;
use PhpMlKit\NDArray\FFI\Lib;
use PhpMlKit\NDArray\NDArray;
use PhpMlKit\NDArray\PadMode;

/**
 * Shape manipulation operations.
 *
 * Provides reshape(), transpose(), flatten(), squeeze(),
 * insert_axis(), ravel(), swap_axes(), etc.
 */
trait HasShapeOps
{
    /**
     * Pad an array.
     *
     * @param array<array<int>>|array<int>|int     $padWidth       number of elements to pad on each side of each axis
     * @param PadMode                              $mode           padding mode
     * @param array<bool|float|int>|bool|float|int $constantValues constant value to pad with (used for PadMode::Constant)
     *
     * @return NDArray padded array
     */
    public function pad(array|int $padWidth, PadMode $mode = PadMode::Constant, array|bool|float|int $constantValues = 0): NDArray
    {
        $normalizedPad = $this->normalizePadWidth($padWidth);
        $padFlat = [];
        foreach ($normalizedPad as [$before, $after]) {
            $padFlat[] = $before;
            $padFlat[] = $after;
        }

        $constants = $this->normalizePadConstants($constantValues, $mode);

        $constantsC = Lib::createCArray('double', $constants);
        $padFlatC = Lib::createShapeArray($padFlat);

        return $this->unaryOp('ndarray_pad', $padFlatC, $mode, $constantsC, \count($constants));
    }

    /**
     * Reshape the array to a new shape.
     *
     * Returns a new array with the specified shape.
     * Supports both C-order (row-major, order='C') and F-order (column-major, order='F').
     *
     * @param array<int> $newShape New shape
     * @param string     $order    Memory layout: 'C' for row-major, 'F' for column-major
     */
    public function reshape(array $newShape, string $order = 'C'): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

        $cShape = Lib::createShapeArray($newShape);
        $orderCode = 'F' === $order ? 1 : 0; // 0=C (RowMajor), 1=F (ColumnMajor)

        $status = $ffi->ndarray_reshape(
            $this->handle,
            $this->offset,
            Lib::createShapeArray($this->shape),
            Lib::createCArray('size_t', $this->strides),
            $this->ndim,
            $cShape,
            \count($newShape),
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
     */
    public function transpose(): NDArray
    {
        return $this->unaryOp('ndarray_transpose');
    }

    /**
     * Swap two axes of the array.
     *
     * @param int $axis1 First axis to swap
     * @param int $axis2 Second axis to swap
     */
    public function swapAxes(int $axis1, int $axis2): NDArray
    {
        return $this->unaryOp('ndarray_swap_axes', $axis1, $axis2);
    }

    /**
     * Permute axes of the array.
     *
     * Reorders the axes according to the given permutation.
     * For example, permuteAxes([1, 0]) on a 2D array is equivalent to transpose().
     *
     * @param array<int> $axes New order of axes
     */
    public function permuteAxes(array $axes): NDArray
    {
        if (\count($axes) !== $this->ndim) {
            throw new ShapeException("permute_axes requires {$this->ndim} axes, got ".\count($axes));
        }

        $normalizedAxes = [];
        foreach ($axes as $axis) {
            if ($axis < 0) {
                $axis = $this->ndim + $axis;
            }
            if ($axis < 0 || $axis >= $this->ndim) {
                throw new ShapeException("Axis {$axis} is out of bounds for array with {$this->ndim} dimensions");
            }
            $normalizedAxes[] = $axis;
        }

        if (\count(array_unique($normalizedAxes)) !== \count($normalizedAxes)) {
            throw new ShapeException('Duplicate axes in permutation');
        }

        $normalizedAxesC = Lib::createShapeArray($normalizedAxes);

        return $this->unaryOp('ndarray_permute_axes', $normalizedAxesC, \count($normalizedAxes));
    }

    /**
     * Merge axes by combining take into into.
     *
     * If possible, merge in the axis take to into. Returns the merged array.
     *
     * @param int $take Axis to merge from
     * @param int $into Axis to merge into
     */
    public function mergeAxes(int $take, int $into): NDArray
    {
        return $this->unaryOp('ndarray_merge_axes', $take, $into);
    }

    /**
     * Reverse the stride of an axis.
     *
     * @param int $axis Axis to invert
     */
    public function invertAxis(int $axis): NDArray
    {
        if ($axis < 0) {
            $axis = $this->ndim + $axis;
        }

        if ($axis < 0 || $axis >= $this->ndim) {
            throw new ShapeException("Axis {$axis} is out of bounds for array with {$this->ndim} dimensions");
        }

        return $this->unaryOp('ndarray_invert_axis', $axis);
    }

    /**
     * Insert a new axis at the specified position.
     *
     * The new axis always has length 1.
     *
     * @param int $axis Position where new axis is inserted
     */
    public function insertAxis(int $axis): NDArray
    {
        if ($axis < 0) {
            $axis = $this->ndim + $axis + 1;
        }

        if ($axis < 0 || $axis > $this->ndim) {
            throw new ShapeException("Axis {$axis} is out of bounds for array with {$this->ndim} dimensions");
        }

        return $this->unaryOp('ndarray_insert_axis', $axis);
    }

    /**
     * Flatten the array to 1D.
     *
     * Always returns a copy in C-order (row-major).
     */
    public function flatten(): NDArray
    {
        return $this->unaryOp('ndarray_flatten');
    }

    /**
     * Ravel the array to 1D.
     *
     * Similar to flatten() but may return a view if the array is contiguous.
     *
     * @param string $order Memory layout: 'C' for row-major, 'F' for column-major
     */
    public function ravel(string $order = 'C'): NDArray
    {
        $orderCode = 'F' === $order ? 1 : 0;

        return $this->unaryOp('ndarray_ravel', $orderCode);
    }

    /**
     * Remove axes of length 1 from the array.
     *
     * If no axes are specified, removes all length-1 axes.
     *
     * @param null|array<int> $axes Specific axes to squeeze (null for all)
     */
    public function squeeze(?array $axes = null): NDArray
    {
        if (null === $axes) {
            $cAxes = null;
            $numAxes = 0;
        } else {
            $cAxes = Lib::createShapeArray($axes);
            $numAxes = \count($axes);
        }

        return $this->unaryOp('ndarray_squeeze', $cAxes, $numAxes);
    }

    /**
     * Expand dimensions by inserting a new axis.
     *
     * Alias for insertAxis().
     *
     * @param int $axis Position where new axis is inserted
     */
    public function expandDims(int $axis): NDArray
    {
        return $this->insertAxis($axis);
    }

    /**
     * Construct an array by repeating A the number of times given by reps.
     *
     * @param array<int>|int|NDArray $reps the number of repetitions of A along each axis
     *
     * @return NDArray the tiled output array
     */
    public function tile(array|int|NDArray $reps): NDArray
    {
        // Normalize reps to an array
        if (\is_int($reps)) {
            $repsArray = [$reps];
        } elseif ($reps instanceof NDArray) {
            $repsArray = $reps->toArray();
            if (!\is_array($repsArray)) {
                $repsArray = [$repsArray];
            }
        } else {
            $repsArray = $reps;
        }

        return $this->unaryOp('ndarray_tile', Lib::createShapeArray($repsArray), \count($repsArray));
    }

    /**
     * Repeat elements of an array.
     *
     * @param array<int>|int|NDArray $repeats the number of repetitions for each element
     * @param null|int               $axis    The axis along which to repeat values. By default, use the flattened input array.
     *
     * @return NDArray output array which has the same shape as a, except along the given axis
     */
    public function repeat(array|int|NDArray $repeats, ?int $axis = null): NDArray
    {
        // Normalize repeats to an array
        if (\is_int($repeats)) {
            $repeatsArray = [$repeats];
        } elseif ($repeats instanceof NDArray) {
            $repeatsArray = $repeats->toArray();
            if (!\is_array($repeatsArray)) {
                $repeatsArray = [$repeatsArray];
            }
        } else {
            $repeatsArray = $repeats;
        }

        $axisValue = $axis ?? -1;

        return $this->unaryOp('ndarray_repeat', Lib::createShapeArray($repeatsArray), \count($repeatsArray), $axisValue);
    }

    /**
     * Normalize pad width to [[before, after], ...] for each axis.
     *
     * @return array<array{0:int,1:int}>
     */
    private function normalizePadWidth(array|int $padWidth): array
    {
        if (\is_int($padWidth)) {
            if ($padWidth < 0) {
                throw new ShapeException('padWidth must be non-negative');
            }

            return array_fill(0, $this->ndim, [$padWidth, $padWidth]);
        }

        if (0 === $this->ndim) {
            throw new ShapeException('pad() requires at least 1 dimension');
        }

        if (2 === \count($padWidth) && isset($padWidth[0], $padWidth[1]) && !\is_array($padWidth[0])) {
            $before = (int) $padWidth[0];
            $after = (int) $padWidth[1];
            if ($before < 0 || $after < 0) {
                throw new ShapeException('padWidth values must be non-negative');
            }

            return array_fill(0, $this->ndim, [$before, $after]);
        }

        if (\count($padWidth) === $this->ndim) {
            $normalized = [];
            foreach ($padWidth as $axis => $entry) {
                if (\is_int($entry)) {
                    if ($entry < 0) {
                        throw new ShapeException("padWidth for axis {$axis} must be non-negative");
                    }
                    $normalized[] = [$entry, $entry];

                    continue;
                }

                if (!\is_array($entry) || 2 !== \count($entry)) {
                    throw new ShapeException("padWidth for axis {$axis} must be int or [before, after]");
                }

                $before = (int) $entry[0];
                $after = (int) $entry[1];
                if ($before < 0 || $after < 0) {
                    throw new ShapeException("padWidth for axis {$axis} must be non-negative");
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
     * @param array<bool|float|int>|bool|float|int $constantValues
     *
     * @return array<float>
     */
    private function normalizePadConstants(array|bool|float|int $constantValues, PadMode $mode): array
    {
        if (PadMode::Constant !== $mode) {
            return [0.0];
        }

        if (!\is_array($constantValues)) {
            return [(float) $constantValues];
        }

        if (0 === \count($constantValues)) {
            return [0.0];
        }

        if (2 === \count($constantValues) && !\is_array($constantValues[0])) {
            return [(float) $constantValues[0], (float) $constantValues[1]];
        }

        $flat = [];
        foreach ($constantValues as $entry) {
            if (\is_array($entry)) {
                if (2 !== \count($entry)) {
                    throw new ShapeException('constantValues per-axis entries must be [before, after]');
                }
                $flat[] = (float) $entry[0];
                $flat[] = (float) $entry[1];
            } else {
                $flat[] = (float) $entry;
            }
        }

        if (1 !== \count($flat) && 2 !== \count($flat) && \count($flat) !== $this->ndim * 2) {
            throw new ShapeException(
                'constantValues must be scalar, [before, after], or per-axis pairs of length '.($this->ndim * 2)
            );
        }

        return $flat;
    }
}
