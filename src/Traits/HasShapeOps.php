<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Traits;

use PhpMlKit\NDArray\Exceptions\ShapeException;
use PhpMlKit\NDArray\FFI\Lib;
use PhpMlKit\NDArray\NDArray;
use PhpMlKit\NDArray\PadMode;
use PhpMlKit\NDArray\ViewMetadata;

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
     * @param array<array{int,int}|int>|array{int,int}|int $padWidth       Number of elements to pad on each side of each axis.
     *                                                                     - int: pad same amount on all sides of every axis
     *                                                                     - array{int,int}: pad [before, after] for all axes
     *                                                                     - array<int|array{int,int}>: per-axis pad (int or [before, after] per axis)
     * @param PadMode                                      $mode           padding mode
     * @param array<bool|float|int>|bool|float|int         $constantValues constant value to pad with (used for PadMode::Constant)
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
     * For contiguous arrays, this returns a zero-copy view with updated metadata.
     * For non-contiguous arrays, data is copied to make it contiguous first.
     *
     * @param array<int> $newShape New shape
     * @param string     $order    Memory layout: 'C' for row-major, 'F' for column-major
     */
    public function reshape(array $newShape, string $order = 'C'): NDArray
    {
        $oldSize = $this->size();
        $newSize = (int) array_product($newShape);

        if ($oldSize !== $newSize) {
            throw new ShapeException(
                "Cannot reshape array of size {$oldSize} into shape ".json_encode($newShape)." with size {$newSize}"
            );
        }

        if (0 === $newSize) {
            $root = $this->base ?? $this;

            return new self(
                handle: $this->handle,
                shape: $newShape,
                dtype: $this->dtype,
                strides: [],
                offset: $this->getOffset(),
                base: $root,
            );
        }

        if ($this->isContiguous()) {
            if ('F' === $order) {
                $newStrides = [];
                $stride = 1;
                foreach ($newShape as $dimSize) {
                    $newStrides[] = $stride;
                    $stride *= $dimSize;
                }
            } else {
                $newStrides = ViewMetadata::computeStrides($newShape);
            }

            $root = $this->base ?? $this;

            return new self(
                handle: $this->handle,
                shape: $newShape,
                dtype: $this->dtype,
                strides: $newStrides,
                offset: $this->getOffset(),
                base: $root,
            );
        }

        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

        $cShape = Lib::createShapeArray($newShape);
        $orderCode = 'F' === $order ? 1 : 0; // 0=C (RowMajor), 1=F (ColumnMajor)

        $meta = $this->viewMetadata()->toCData();
        $status = $ffi->ndarray_reshape(
            $this->handle,
            Lib::addr($meta),
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
     *
     * For contiguous arrays, this is a zero-copy operation returning a view.
     * For non-contiguous arrays, data is copied first.
     */
    public function transpose(): NDArray
    {
        if ($this->isContiguous()) {
            $shape = $this->shape();
            $strides = $this->strides();

            $newShape = array_reverse($shape);
            $newStrides = array_reverse($strides);

            $root = $this->base ?? $this;

            return new self(
                handle: $this->handle,
                shape: $newShape,
                dtype: $this->dtype,
                strides: $newStrides,
                offset: $this->getOffset(),
                base: $root,
            );
        }

        return $this->unaryOp('ndarray_transpose');
    }

    /**
     * Swap two axes of the array.
     *
     * This is a zero-copy operation that returns a view with swapped
     * shape and stride metadata. The underlying data is shared.
     *
     * @param int $axis1 First axis to swap
     * @param int $axis2 Second axis to swap
     */
    public function swapaxes(int $axis1, int $axis2): NDArray
    {
        $ndim = $this->ndim();

        // Normalize negative indices
        if ($axis1 < 0) {
            $axis1 += $ndim;
        }
        if ($axis2 < 0) {
            $axis2 += $ndim;
        }

        // Validate indices
        if ($axis1 < 0 || $axis1 >= $ndim) {
            throw new ShapeException("Axis {$axis1} is out of bounds for array with {$ndim} dimensions");
        }
        if ($axis2 < 0 || $axis2 >= $ndim) {
            throw new ShapeException("Axis {$axis2} is out of bounds for array with {$ndim} dimensions");
        }

        // No change needed if axes are the same
        if ($axis1 === $axis2) {
            return $this;
        }

        // Swap shape and strides
        $newShape = $this->shape();
        $newStrides = $this->strides();

        $tempShape = $newShape[$axis1];
        $newShape[$axis1] = $newShape[$axis2];
        $newShape[$axis2] = $tempShape;

        $tempStride = $newStrides[$axis1];
        $newStrides[$axis1] = $newStrides[$axis2];
        $newStrides[$axis2] = $tempStride;

        // Return view
        $root = $this->base ?? $this;

        return new self(
            handle: $this->handle,
            shape: $newShape,
            dtype: $this->dtype,
            strides: $newStrides,
            offset: $this->getOffset(),
            base: $root,
        );
    }

    /**
     * Permute axes of the array.
     *
     * Reorders the axes according to the given permutation.
     * For example, permute(1, 0) on a 2D array is equivalent to transpose().
     *
     * @param int ...$axes New order of axes
     */
    public function permute(int ...$axes): NDArray
    {
        if (\count($axes) !== $this->ndim()) {
            throw new ShapeException("permute requires {$this->ndim()} axes, got ".\count($axes));
        }

        $normalizedAxes = [];
        foreach ($axes as $axis) {
            if ($axis < 0) {
                $axis = $this->ndim() + $axis;
            }
            if ($axis < 0 || $axis >= $this->ndim()) {
                throw new ShapeException("Axis {$axis} is out of bounds for array with {$this->ndim()} dimensions");
            }
            $normalizedAxes[] = $axis;
        }

        if (\count(array_unique($normalizedAxes)) !== \count($normalizedAxes)) {
            throw new ShapeException('Duplicate axes in permutation');
        }

        $normalizedAxesC = Lib::createShapeArray($normalizedAxes);

        return $this->unaryOp('ndarray_permute', $normalizedAxesC, \count($normalizedAxes));
    }

    /**
     * Merge axes by combining take into into.
     *
     * Merges the axis 'take' into axis 'into' by folding the dimensions.
     * The 'take' axis is removed and its size is multiplied into the 'into' axis.
     * This is a zero-copy operation that returns a view with updated metadata.
     *
     * For example, on an array with shape [2, 3, 4]:
     * - mergeaxes(1, 0) merges axis 1 into axis 0, resulting in shape [6, 4]
     * - mergeaxes(0, 1) merges axis 0 into axis 1, resulting in shape [3, 8]
     *
     * @param int $take Axis to merge from (will be removed)
     * @param int $into Axis to merge into (size will be multiplied)
     */
    public function mergeaxes(int $take, int $into): NDArray
    {
        $ndim = $this->ndim();

        if ($take < 0) {
            $take += $ndim;
        }
        if ($into < 0) {
            $into += $ndim;
        }

        if ($take < 0 || $take >= $ndim) {
            throw new ShapeException("Axis {$take} is out of bounds for array with {$ndim} dimensions");
        }
        if ($into < 0 || $into >= $ndim) {
            throw new ShapeException("Axis {$into} is out of bounds for array with {$ndim} dimensions");
        }

        if ($take === $into) {
            throw new ShapeException('Cannot merge axis into itself');
        }

        $shape = $this->shape();
        $strides = $this->strides();

        $newShape = $shape;
        $newStrides = $strides;

        $newShape[$into] = $shape[$into] * $shape[$take];

        array_splice($newShape, $take, 1);
        array_splice($newStrides, $take, 1);

        if ($take < $into) {
            $newStrides[$into - 1] = $strides[$take];
        } else {
            $newStrides[$into] = $strides[$into];
        }

        $root = $this->base ?? $this;

        return new self(
            handle: $this->handle,
            shape: $newShape,
            dtype: $this->dtype,
            strides: $newStrides,
            offset: $this->getOffset(),
            base: $root,
        );
    }

    /**
     * Reverse the order of elements in an array along the given axis or axes.
     *
     * @param null|array<int>|int $axes Axis or axes to flip. If null, flip over all axes.
     */
    public function flip(array|int|null $axes = null): NDArray
    {
        if (null === $axes) {
            $axesArray = [];
            $numAxes = 0;
        } elseif (\is_int($axes)) {
            $axis = $axes;
            if ($axis < 0) {
                $axis = $this->ndim() + $axis;
            }
            if ($axis < 0 || $axis >= $this->ndim()) {
                throw new ShapeException("Axis {$axes} is out of bounds for array with {$this->ndim()} dimensions");
            }
            $axesArray = [$axis];
            $numAxes = 1;
        } else {
            $axesArray = [];
            foreach ($axes as $axis) {
                if ($axis < 0) {
                    $axis = $this->ndim() + $axis;
                }
                if ($axis < 0 || $axis >= $this->ndim()) {
                    throw new ShapeException("Axis {$axis} is out of bounds for array with {$this->ndim()} dimensions");
                }
                $axesArray[] = $axis;
            }
            $numAxes = \count($axesArray);
        }

        return $this->unaryOp('ndarray_flip', Lib::createCArray('int64_t', $axesArray), $numAxes);
    }

    /**
     * Insert a new axis at the specified position.
     *
     * The new axis always has length 1. This is a zero-copy operation that returns
     * a view with updated metadata.
     *
     * @param int $axis Position where new axis is inserted
     */
    public function insertaxis(int $axis): NDArray
    {
        if ($axis < 0) {
            $axis = $this->ndim() + $axis + 1;
        }

        if ($axis < 0 || $axis > $this->ndim()) {
            throw new ShapeException("Axis {$axis} is out of bounds for array with {$this->ndim()} dimensions");
        }

        $shape = $this->shape();
        $strides = $this->strides();

        array_splice($shape, $axis, 0, [1]);

        if ($axis < \count($strides)) {
            $strideToInsert = $strides[$axis];
        } else {
            $strideToInsert = 1;
            if (\count($strides) > 0) {
                $strideToInsert = $strides[\count($strides) - 1];
            }
        }
        array_splice($strides, $axis, 0, [$strideToInsert]);

        $root = $this->base ?? $this;

        return new self(
            handle: $this->handle,
            shape: $shape,
            dtype: $this->dtype,
            strides: $strides,
            offset: $this->getOffset(),
            base: $root,
        );
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
     * Similar to flatten() but returns a view if the array is contiguous.
     * For non-contiguous arrays, data is copied to make it contiguous.
     *
     * @param string $order Memory layout: 'C' for row-major, 'F' for column-major
     */
    public function ravel(string $order = 'C'): NDArray
    {
        if ($this->isContiguous()) {
            $n = $this->size();

            $root = $this->base ?? $this;

            return new self(
                handle: $this->handle,
                shape: [$n],
                dtype: $this->dtype,
                strides: [1],
                offset: $this->getOffset(),
                base: $root,
            );
        }

        $orderCode = 'F' === $order ? 1 : 0;

        return $this->unaryOp('ndarray_ravel', $orderCode);
    }

    /**
     * Remove axes of length 1 from the array.
     *
     * If no axes are specified, removes all length-1 axes (NumPy behavior).
     * This is a zero-copy operation that returns a view with updated metadata.
     *
     * @param null|array<int> $axes Specific axes to squeeze (null for all)
     */
    public function squeeze(?array $axes = null): NDArray
    {
        $shape = $this->shape();
        $strides = $this->strides();

        if (null === $axes) {
            $axesToRemove = [];
            foreach ($shape as $i => $dim) {
                if (1 === $dim) {
                    $axesToRemove[] = $i;
                }
            }
        } else {
            $axesToRemove = [];
            foreach ($axes as $axis) {
                $normalizedAxis = $axis;
                if ($normalizedAxis < 0) {
                    $normalizedAxis = $this->ndim() + $normalizedAxis;
                }
                if ($normalizedAxis < 0 || $normalizedAxis >= $this->ndim()) {
                    throw new ShapeException("Axis {$axis} is out of bounds for array with {$this->ndim()} dimensions");
                }
                if (1 !== $shape[$normalizedAxis]) {
                    throw new ShapeException("Cannot squeeze axis {$normalizedAxis} with size {$shape[$normalizedAxis]}");
                }
                $axesToRemove[] = $normalizedAxis;
            }
        }

        rsort($axesToRemove);
        $newShape = $shape;
        $newStrides = $strides;

        foreach ($axesToRemove as $axis) {
            array_splice($newShape, $axis, 1);
            array_splice($newStrides, $axis, 1);
        }

        if (empty($newShape)) {
            $lastSqueezedAxis = $axesToRemove[\count($axesToRemove) - 1];
            $newShape = [1];
            $newStrides = [$strides[$lastSqueezedAxis]];
        }

        $root = $this->base ?? $this;

        return new self(
            handle: $this->handle,
            shape: $newShape,
            dtype: $this->dtype,
            strides: $newStrides,
            offset: $this->getOffset(),
            base: $root,
        );
    }

    /**
     * Expand dimensions by inserting a new axis.
     *
     * Alias for insertaxis().
     *
     * @param int $axis Position where new axis is inserted
     */
    public function expandDims(int $axis): NDArray
    {
        return $this->insertaxis($axis);
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
     * @param array<array{int,int}|int>|array{int,int}|int $padWidth number of elements to pad on each side of each axis
     *
     * @return array<array{0:int,1:int}>
     */
    private function normalizePadWidth(array|int $padWidth): array
    {
        if (\is_int($padWidth)) {
            if ($padWidth < 0) {
                throw new ShapeException('padWidth must be non-negative');
            }

            return array_fill(0, $this->ndim(), [$padWidth, $padWidth]);
        }

        if (0 === $this->ndim()) {
            throw new ShapeException('pad() requires at least 1 dimension');
        }

        if (2 === \count($padWidth) && isset($padWidth[0], $padWidth[1]) && !\is_array($padWidth[0])) {
            $before = (int) $padWidth[0];
            $after = (int) $padWidth[1];
            if ($before < 0 || $after < 0) {
                throw new ShapeException('padWidth values must be non-negative');
            }

            return array_fill(0, $this->ndim(), [$before, $after]);
        }

        if (\count($padWidth) === $this->ndim()) {
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
            "padWidth must be an int, [before, after], or per-axis list of length {$this->ndim()}"
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

        if (1 !== \count($flat) && 2 !== \count($flat) && \count($flat) !== $this->ndim() * 2) {
            throw new ShapeException(
                'constantValues must be scalar, [before, after], or per-axis pairs of length '.($this->ndim() * 2)
            );
        }

        return $flat;
    }
}
