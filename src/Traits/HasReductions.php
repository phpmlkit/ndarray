<?php

declare(strict_types=1);

namespace NDArray\Traits;

use NDArray\DType;
use NDArray\NDArray;
use NDArray\SortKind;
use NDArray\FFI\Lib;

/**
 * Reduction and aggregation operations trait for NDArray.
 *
 * Provides scalar reductions (sum, mean, min, max, etc.) and
 * axis reductions with keepdims support.
 */
trait HasReductions
{
    /**
     * Sum of array elements over a given axis.
     *
     * @param int|null $axis Axis along which to sum. If null, sum over all elements.
     * @param bool $keepdims If true, the reduced axis is retained with size 1.
     * @return NDArray|float Scalar if axis is null, otherwise an NDArray.
     */
    public function sum(?int $axis = null, bool $keepdims = false): NDArray|float
    {
        if ($axis === null) {
            return $this->scalarReductionOp('ndarray_sum');
        }
        return $this->unaryOp('ndarray_sum_axis', $axis, $keepdims);
    }

    /**
     * Mean of array elements over a given axis.
     *
     * @param int|null $axis Axis along which to compute mean. If null, compute mean of all elements.
     * @param bool $keepdims If true, the reduced axis is retained with size 1.
     * @return NDArray|float Scalar if axis is null, otherwise an NDArray.
     */
    public function mean(?int $axis = null, bool $keepdims = false): NDArray|float
    {
        if ($axis === null) {
            return $this->scalarReductionOp('ndarray_mean');
        }
        return $this->unaryOp('ndarray_mean_axis', $axis, $keepdims);
    }

    /**
     * Minimum of array elements over a given axis.
     *
     * @param int|null $axis Axis along which to find minimum. If null, find minimum of all elements.
     * @param bool $keepdims If true, the reduced axis is retained with size 1.
     * @return NDArray|float Scalar if axis is null, otherwise an NDArray.
     */
    public function min(?int $axis = null, bool $keepdims = false): NDArray|float
    {
        if ($axis === null) {
            return $this->scalarReductionOp('ndarray_min');
        }
        return $this->unaryOp('ndarray_min_axis', $axis, $keepdims);
    }

    /**
     * Maximum of array elements over a given axis.
     *
     * @param int|null $axis Axis along which to find maximum. If null, find maximum of all elements.
     * @param bool $keepdims If true, the reduced axis is retained with size 1.
     * @return NDArray|float Scalar if axis is null, otherwise an NDArray.
     */
    public function max(?int $axis = null, bool $keepdims = false): NDArray|float
    {
        if ($axis === null) {
            return $this->scalarReductionOp('ndarray_max');
        }
        return $this->unaryOp('ndarray_max_axis', $axis, $keepdims);
    }

    /**
     * Index of minimum value over a given axis.
     *
     * @param int|null $axis Axis along which to find argmin. If null, find argmin of flattened array.
     * @param bool $keepdims If true, the reduced axis is retained with size 1.
     * @return NDArray|int Scalar index if axis is null, otherwise an NDArray of indices.
     */
    public function argmin(?int $axis = null, bool $keepdims = false): NDArray|int
    {
        if ($axis === null) {
            return (int) $this->scalarReductionOp('ndarray_argmin');
        }
        return $this->unaryOp('ndarray_argmin_axis', $axis, $keepdims);
    }

    /**
     * Index of maximum value over a given axis.
     *
     * @param int|null $axis Axis along which to find argmax. If null, find argmax of flattened array.
     * @param bool $keepdims If true, the reduced axis is retained with size 1.
     * @return NDArray|int Scalar index if axis is null, otherwise an NDArray of indices.
     */
    public function argmax(?int $axis = null, bool $keepdims = false): NDArray|int
    {
        if ($axis === null) {
            return (int) $this->scalarReductionOp('ndarray_argmax');
        }
        return $this->unaryOp('ndarray_argmax_axis', $axis, $keepdims);
    }

    /**
     * Return a sorted copy of the array.
     *
     * @param int|null $axis Axis along which to sort. If null, sort flattened data.
     * @param SortKind $kind Sorting algorithm.
     * @return NDArray
     */
    public function sort(?int $axis = -1, SortKind $kind = SortKind::QuickSort): NDArray
    {
        if ($axis === null) {
            return $this->unaryOp('ndarray_sort_flat', $kind);
        }
        return $this->unaryOp('ndarray_sort_axis', $axis, $kind);
    }

    /**
     * Return indices that would sort the array.
     *
     * @param int|null $axis Axis along which to argsort. If null, argsort flattened data.
     * @param SortKind $kind Sorting algorithm.
     * @return NDArray Int64 indices array.
     */
    public function argsort(?int $axis = -1, SortKind $kind = SortKind::QuickSort): NDArray
    {
        if ($axis === null) {
            return $this->unaryOp('ndarray_argsort_flat', $kind);
        }
        return $this->unaryOp('ndarray_argsort_axis', $axis, $kind);
    }

    /**
     * Return top-k values and indices like PyTorch topk.
     *
     * @param int $k Number of elements to select
     * @param int|null $axis Axis along which to select. If null, flatten first.
     * @param bool $largest If true, select largest values; otherwise smallest values
     * @param bool $sorted If true, keep selected values sorted by rank
     * @param SortKind $kind Sorting algorithm
     * @return array{values: NDArray, indices: NDArray}
     */
    public function topk(
        int $k,
        ?int $axis = -1,
        bool $largest = true,
        bool $sorted = true,
        SortKind $kind = SortKind::QuickSort
    ): array {
        if ($k < 0) {
            throw new \InvalidArgumentException('k must be >= 0');
        }

        if ($axis === null) {
            return $this->topkFlatOp($k, $largest, $sorted, $kind);
        }

        return $this->topkAxisOp($k, $axis, $largest, $sorted, $kind);
    }

    /**
     * Product of array elements over a given axis.
     *
     * @param int|null $axis Axis along which to compute product. If null, compute product of all elements.
     * @param bool $keepdims If true, the reduced axis is retained with size 1.
     * @return NDArray|float Scalar if axis is null, otherwise an NDArray.
     */
    public function product(?int $axis = null, bool $keepdims = false): NDArray|float
    {
        if ($axis === null) {
            return $this->scalarReductionOp('ndarray_product');
        }
        return $this->unaryOp('ndarray_product_axis', $axis, $keepdims);
    }

    /**
     * Cumulative sum of array elements.
     *
     * @param int|null $axis Axis along which to compute cumulative sum. If null, flatten and return 1D.
     * @return NDArray
     */
    public function cumsum(?int $axis = null): NDArray
    {
        if ($axis === null) {
            return $this->unaryOp('ndarray_cumsum');
        }
        return $this->unaryOp('ndarray_cumsum_axis', $axis);
    }

    /**
     * Cumulative product of array elements.
     *
     * @param int|null $axis Axis along which to compute cumulative product. If null, flatten and return 1D.
     * @return NDArray
     */
    public function cumprod(?int $axis = null): NDArray
    {
        if ($axis === null) {
            return $this->unaryOp('ndarray_cumprod');
        }
        return $this->unaryOp('ndarray_cumprod_axis', $axis);
    }

    /**
     * Variance of array elements over a given axis.
     *
     * @param int|null $axis Axis along which to compute variance. If null, compute variance of all elements.
     * @param int $ddof Delta degrees of freedom (0 for population, 1 for sample).
     * @param bool $keepdims If true, the reduced axis is retained with size 1.
     * @return NDArray|float Scalar if axis is null, otherwise an NDArray.
     */
    public function var(?int $axis = null, int $ddof = 0, bool $keepdims = false): NDArray|float
    {
        if ($axis === null) {
            return $this->scalarReductionOp('ndarray_var', $ddof);
        }
        return $this->unaryOp('ndarray_var_axis', $axis, $keepdims, $ddof);
    }

    /**
     * Standard deviation of array elements over a given axis.
     *
     * @param int|null $axis Axis along which to compute std. If null, compute std of all elements.
     * @param int $ddof Delta degrees of freedom (0 for population, 1 for sample).
     * @param bool $keepdims If true, the reduced axis is retained with size 1.
     * @return NDArray|float Scalar if axis is null, otherwise an NDArray.
     */
    public function std(?int $axis = null, int $ddof = 0, bool $keepdims = false): NDArray|float
    {
        if ($axis === null) {
            return $this->scalarReductionOp('ndarray_std', $ddof);
        }
        return $this->unaryOp('ndarray_std_axis', $axis, $keepdims, $ddof);
    }

    /**
     * Count occurrences of non-negative integer values in flattened input.
     *
     * @param int|null $minlength Minimum output length
     * @return NDArray Int64 counts array
     */
    public function bincount(?int $minlength = null): NDArray
    {
        $minlength ??= 0;
        if ($minlength < 0) {
            throw new \InvalidArgumentException('minlength must be >= 0');
        }
        return $this->unaryOp('ndarray_bincount', $minlength);
    }

    /**
     * Perform topk along axis.
     *
     * @return array{values: NDArray, indices: NDArray}
     */
    private function topkAxisOp(int $k, int $axis, bool $largest, bool $sorted, SortKind $kind): array
    {
        $ffi = Lib::get();
        $outValuesHandle = $ffi->new('struct NdArrayHandle*');
        $outIndicesHandle = $ffi->new('struct NdArrayHandle*');

        $shape = Lib::createShapeArray($this->shape);
        $strides = Lib::createShapeArray($this->strides);

        $outShapeBuf = Lib::createCArray('size_t', array_fill(0, Lib::MAX_NDIM, 0));

        $status = $ffi->ndarray_topk_axis(
            $this->handle,
            $this->offset,
            $shape,
            $strides,
            count($this->shape),
            $axis,
            $k,
            $largest,
            $sorted,
            $kind->value,
            Lib::addr($outValuesHandle),
            Lib::addr($outIndicesHandle),
            $outShapeBuf,
            Lib::MAX_NDIM
        );

        Lib::checkStatus($status);

        $ndim = count($this->shape);
        $outShape = Lib::extractShapeFromPointer($outShapeBuf, $ndim);

        return [
            'values' => new NDArray($outValuesHandle, $outShape, $this->dtype),
            'indices' => new NDArray($outIndicesHandle, $outShape, DType::Int64),
        ];
    }

    /**
     * Perform topk over flattened array.
     *
     * @return array{values: NDArray, indices: NDArray}
     */
    private function topkFlatOp(int $k, bool $largest, bool $sorted, SortKind $kind): array
    {
        $ffi = Lib::get();
        $outValuesHandle = $ffi->new('struct NdArrayHandle*');
        $outIndicesHandle = $ffi->new('struct NdArrayHandle*');

        $shape = Lib::createShapeArray($this->shape);
        $strides = Lib::createShapeArray($this->strides);

        $outShapeBuf = Lib::createCArray('size_t', [0]);

        $status = $ffi->ndarray_topk_flat(
            $this->handle,
            $this->offset,
            $shape,
            $strides,
            count($this->shape),
            $k,
            $largest,
            $sorted,
            $kind->value,
            Lib::addr($outValuesHandle),
            Lib::addr($outIndicesHandle),
            $outShapeBuf
        );

        Lib::checkStatus($status);

        $outShape = [(int) $outShapeBuf[0]];

        return [
            'values' => new NDArray($outValuesHandle, $outShape, $this->dtype),
            'indices' => new NDArray($outIndicesHandle, $outShape, DType::Int64),
        ];
    }
}
