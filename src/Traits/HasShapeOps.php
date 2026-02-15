<?php

declare(strict_types=1);

namespace NDArray\Traits;

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
}
