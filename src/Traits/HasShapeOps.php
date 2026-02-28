<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Traits;

use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\Exceptions\NDArrayException;
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
     * @param int $axis1 First axis to swap
     * @param int $axis2 Second axis to swap
     */
    public function swapaxes(int $axis1, int $axis2): NDArray
    {
        return $this->unaryOp('ndarray_swap', $axis1, $axis2);
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
     * If possible, merge in the axis take to into. Returns the merged array.
     *
     * @param int $take Axis to merge from
     * @param int $into Axis to merge into
     */
    public function merge(int $take, int $into): NDArray
    {
        return $this->unaryOp('ndarray_merge', $take, $into);
    }

    /**
     * Reverse the order of elements in an array along the given axis or axes.
     *
     * @param null|array<int>|int $axes Axis or axes to flip. If null, flip over all axes.
     */
    public function flip(array|int|null $axes = null): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');
        [$outDtypeBuf, $outNdimBuf, $outShapeBuf] = Lib::createOutputMetadataBuffers();

        $meta = $this->viewMetadata()->toCData();

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

        $status = $ffi->ndarray_flip(
            $this->handle,
            Lib::addr($meta),
            Lib::createCArray('int64_t', $axesArray),
            $numAxes,
            Lib::addr($outHandle),
            Lib::addr($outDtypeBuf),
            Lib::addr($outNdimBuf),
            $outShapeBuf,
            Lib::MAX_NDIM
        );

        Lib::checkStatus($status);

        $dtype = DType::tryFrom((int) $outDtypeBuf->cdata);
        if (null === $dtype) {
            throw new NDArrayException('Invalid dtype returned from Rust');
        }

        $ndim = (int) $outNdimBuf->cdata;
        $shape = Lib::extractShapeFromPointer($outShapeBuf, $ndim);

        return new NDArray($outHandle, $shape, $dtype);
    }

    /**
     * Insert a new axis at the specified position.
     *
     * The new axis always has length 1. This is a zero-copy operation that returns
     * a view with updated metadata.
     *
     * @param int $axis Position where new axis is inserted
     */
    public function insert(int $axis): NDArray
    {
        if ($axis < 0) {
            $axis = $this->ndim() + $axis + 1;
        }

        if ($axis < 0 || $axis > $this->ndim()) {
            throw new ShapeException("Axis {$axis} is out of bounds for array with {$this->ndim()} dimensions");
        }

        // Get current shape and strides
        $shape = $this->shape();
        $strides = $this->strides();

        // Insert size 1 into shape at position $axis
        array_splice($shape, $axis, 0, [1]);

        // Insert appropriate stride
        // The stride for the new axis should be the stride of the next axis,
        // or if at the end, use the last axis's stride
        if ($axis < \count($strides)) {
            // Insert the stride of the axis that will follow
            $strideToInsert = $strides[$axis];
        } else {
            // At the end - stride should be the size of an element in bytes
            // But we need to calculate based on the last dimension
            $strideToInsert = 1;
            if (\count($strides) > 0) {
                $strideToInsert = $strides[\count($strides) - 1];
            }
        }
        array_splice($strides, $axis, 0, [$strideToInsert]);

        // Get the root array for proper view chain
        $root = $this->base ?? $this;

        // Create view with new shape and strides
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
        // Check if we can create a zero-copy view
        if ($this->isContiguous()) {
            $n = $this->size();

            // Get the root array for proper view chain
            $root = $this->base ?? $this;

            // Create 1D view with stride 1 (element count, not bytes)
            // For a contiguous array flattened to 1D, stride is always 1
            return new self(
                handle: $this->handle,
                shape: [$n],
                dtype: $this->dtype,
                strides: [1],
                offset: $this->getOffset(),
                base: $root,
            );
        }

        // Non-contiguous: must copy data first via FFI
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

        // Determine which axes to squeeze
        if (null === $axes) {
            // NumPy behavior: squeeze all length-1 axes
            $axesToRemove = [];
            foreach ($shape as $i => $dim) {
                if (1 === $dim) {
                    $axesToRemove[] = $i;
                }
            }
        } else {
            // Validate and normalize the provided axes
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

        // Remove axes in reverse order to maintain correct indices
        rsort($axesToRemove);
        $newShape = $shape;
        $newStrides = $strides;

        foreach ($axesToRemove as $axis) {
            array_splice($newShape, $axis, 1);
            array_splice($newStrides, $axis, 1);
        }

        // NumPy behavior: if all dimensions were squeezed, keep at least 1 dimension
        if (empty($newShape)) {
            // Keep the last dimension that was squeezed (shape [1], stride from original)
            $lastSqueezedAxis = $axesToRemove[\count($axesToRemove) - 1];
            $newShape = [1];
            $newStrides = [$strides[$lastSqueezedAxis]];
        }

        // Get the root array for proper view chain
        $root = $this->base ?? $this;

        // Create view with new shape and strides
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
     * Alias for insert().
     *
     * @param int $axis Position where new axis is inserted
     */
    public function expandDims(int $axis): NDArray
    {
        return $this->insert($axis);
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
