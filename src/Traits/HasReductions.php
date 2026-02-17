<?php

declare(strict_types=1);

namespace NDArray\Traits;

use FFI;
use FFI\CData;
use NDArray\DType;
use NDArray\FFI\Box;
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
            return $this->scalarReductionOp('ndarray_sum', $this->dtype);
        }
        return $this->axisReductionOp('ndarray_sum_axis', $axis, $keepdims, $this->dtype);
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
            return $this->scalarReductionOp('ndarray_mean', DType::Float64);
        }
        return $this->axisReductionOp('ndarray_mean_axis', $axis, $keepdims, DType::Float64);
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
            return $this->scalarReductionOp('ndarray_min', $this->dtype);
        }
        return $this->axisReductionOp('ndarray_min_axis', $axis, $keepdims, $this->dtype);
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
            return $this->scalarReductionOp('ndarray_max', $this->dtype);
        }
        return $this->axisReductionOp('ndarray_max_axis', $axis, $keepdims, $this->dtype);
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
            return (int) $this->scalarReductionOp('ndarray_argmin', DType::Int64);
        }
        return $this->axisReductionOp('ndarray_argmin_axis', $axis, $keepdims, DType::Int64);
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
            return (int) $this->scalarReductionOp('ndarray_argmax', DType::Int64);
        }
        return $this->axisReductionOp('ndarray_argmax_axis', $axis, $keepdims, DType::Int64);
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
            return $this->sortFlatOp('ndarray_sort_flat', $kind, $this->dtype);
        }
        return $this->sortAxisOp('ndarray_sort_axis', $axis, $kind, $this->dtype);
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
            return $this->sortFlatOp('ndarray_argsort_flat', $kind, DType::Int64);
        }
        return $this->sortAxisOp('ndarray_argsort_axis', $axis, $kind, DType::Int64);
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
            return $this->scalarReductionOp('ndarray_product', $this->dtype);
        }
        return $this->axisReductionOp('ndarray_product_axis', $axis, $keepdims, $this->dtype);
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
            return $this->cumulativeReductionOp('ndarray_cumsum');
        }
        return $this->cumulativeReductionAxisOp('ndarray_cumsum_axis', $axis);
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
            return $this->cumulativeReductionOp('ndarray_cumprod');
        }
        return $this->cumulativeReductionAxisOp('ndarray_cumprod_axis', $axis);
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
            return $this->scalarReductionDdofOp('ndarray_var', DType::Float64, $ddof);
        }
        return $this->axisReductionDdofOp('ndarray_var_axis', $axis, $keepdims, DType::Float64, $ddof);
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
            return $this->scalarReductionDdofOp('ndarray_std', DType::Float64, $ddof);
        }
        return $this->axisReductionDdofOp('ndarray_std_axis', $axis, $keepdims, DType::Float64, $ddof);
    }

    /**
     * Count occurrences of non-negative integer values in flattened input.
     *
     * @param int|null $minlength Minimum output length
     * @return NDArray Int64 counts array
     */
    public function bincount(?int $minlength = null): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');
        $minlength ??= 0;
        if ($minlength < 0) {
            throw new \InvalidArgumentException('minlength must be >= 0');
        }

        $status = $ffi->ndarray_bincount(
            $this->handle,
            $this->offset,
            Lib::createShapeArray($this->shape),
            Lib::createShapeArray($this->strides),
            count($this->shape),
            $minlength,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        $outLen = Lib::createBox('size_t');
        
        $statusLen = $ffi->ndarray_len($outHandle, Lib::addr($outLen));
        Lib::checkStatus($statusLen);

        return new NDArray($outHandle, [$outLen->cdata], DType::Int64);
    }

    /**
     * Perform a scalar reduction and return the value directly (no NDArray allocation).
     *
     * @param string $funcName FFI function name
     * @param DType $dtype Expected dtype of result
     * @return float|int
     */
    private function scalarReductionOp(string $funcName, DType $dtype): float|int
    {
        $ffi = Lib::get();
        $outValue = Lib::createBox('double');
        $outDtype = Lib::createBox('uint8_t');

        $shape = Lib::createShapeArray($this->shape);
        $strides = Lib::createShapeArray($this->strides);

        $status = $ffi->$funcName(
            $this->handle,
            $this->offset,
            $shape,
            $strides,
            count($this->shape),
            FFI::addr($outValue),
            Lib::addr($outDtype)
        );

        Lib::checkStatus($status);

        return $this->interpretScalarFromBuffer($ffi, $outValue, DType::from($outDtype->cdata));
    }

    /**
     * Perform a scalar reduction with ddof parameter.
     *
     * @param string $funcName FFI function name
     * @param DType $dtype Expected dtype of result
     * @param int $ddof Delta degrees of freedom
     * @return float|int
     */
    private function scalarReductionDdofOp(string $funcName, DType $dtype, int $ddof): float|int
    {
        $ffi = Lib::get();
        $outValue = Lib::createBox('double');
        $outDtype = Lib::createBox('uint8_t');

        $shape = Lib::createShapeArray($this->shape);
        $strides = Lib::createShapeArray($this->strides);

        $status = $ffi->$funcName(
            $this->handle,
            $this->offset,
            $shape,
            $strides,
            count($this->shape),
            (float) $ddof,
            FFI::addr($outValue),
            Lib::addr($outDtype)
        );

        Lib::checkStatus($status);

        return $this->interpretScalarFromBuffer($ffi, $outValue, DType::from($outDtype->cdata));
    }

    /**
     * Interpret 8-byte buffer as scalar based on dtype.
     *
     * @param FFI $ffi FFI instance
     * @param CData&Box $outValue 8-byte buffer (allocated as double)
     * @param DType $dtype Result dtype from Rust
     * @return float|int
     */
    private function interpretScalarFromBuffer(FFI $ffi, CData $outValue, DType $dtype): float|int
    {
        $addr = FFI::addr($outValue);
        return match ($dtype) {
            DType::Float64, DType::Float32 => $outValue->cdata,
            DType::Int64, DType::Int32, DType::Int16, DType::Int8 => $ffi->cast('int64_t*', $addr)[0],
            DType::Uint64, DType::Uint32, DType::Uint16, DType::Uint8 => $ffi->cast('uint64_t*', $addr)[0],
            default => 0,
        };
    }

    /**
     * Perform an axis reduction.
     *
     * @param string $funcName FFI function name
     * @param int $axis Axis to reduce along
     * @param bool $keepdims Whether to keep reduced dimensions
     * @param DType $dtype Output dtype
     * @return NDArray
     */
    private function axisReductionOp(string $funcName, int $axis, bool $keepdims, DType $dtype): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

        $shape = Lib::createShapeArray($this->shape);
        $strides = Lib::createShapeArray($this->strides);

        $status = $ffi->$funcName(
            $this->handle,
            $this->offset,
            $shape,
            $strides,
            count($this->shape),
            $axis,
            $keepdims,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        $outShape = $this->computeAxisOutputShape($this->shape, $axis, $keepdims);
        return new NDArray($outHandle, $outShape, $dtype);
    }

    /**
     * Perform an axis reduction with ddof parameter.
     *
     * @param string $funcName FFI function name
     * @param int $axis Axis to reduce along
     * @param bool $keepdims Whether to keep reduced dimensions
     * @param DType $dtype Output dtype
     * @param int $ddof Delta degrees of freedom
     * @return NDArray
     */
    private function axisReductionDdofOp(string $funcName, int $axis, bool $keepdims, DType $dtype, int $ddof): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

        $shape = Lib::createShapeArray($this->shape);
        $strides = Lib::createShapeArray($this->strides);

        $status = $ffi->$funcName(
            $this->handle,
            $this->offset,
            $shape,
            $strides,
            count($this->shape),
            $axis,
            $keepdims,
            (float) $ddof,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        $outShape = $this->computeAxisOutputShape($this->shape, $axis, $keepdims);
        return new NDArray($outHandle, $outShape, $dtype);
    }

    /**
     * Perform a cumulative reduction over flattened array. Returns 1D array.
     *
     * @param string $funcName FFI function name (ndarray_cumsum or ndarray_cumprod)
     * @return NDArray
     */
    private function cumulativeReductionOp(string $funcName): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

        $shape = Lib::createShapeArray($this->shape);
        $strides = Lib::createShapeArray($this->strides);

        $status = $ffi->$funcName(
            $this->handle,
            $this->offset,
            $shape,
            $strides,
            count($this->shape),
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        $size = (int) array_product($this->shape);
        return new NDArray($outHandle, [$size], $this->dtype);
    }

    /**
     * Perform a cumulative reduction along an axis. Returns same shape as input.
     *
     * @param string $funcName FFI function name (ndarray_cumsum_axis or ndarray_cumprod_axis)
     * @param int $axis Axis to reduce along
     * @return NDArray
     */
    private function cumulativeReductionAxisOp(string $funcName, int $axis): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

        $shape = Lib::createShapeArray($this->shape);
        $strides = Lib::createShapeArray($this->strides);

        $status = $ffi->$funcName(
            $this->handle,
            $this->offset,
            $shape,
            $strides,
            count($this->shape),
            $axis,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new NDArray($outHandle, $this->shape, $this->dtype);
    }

    /**
     * Perform sort/argsort along an axis. Returns same shape as input.
     *
     * @param string $funcName FFI function name (ndarray_sort_axis or ndarray_argsort_axis)
     * @param int $axis Axis to sort along
     * @param SortKind $kind Sorting algorithm
     * @param DType $dtype Output dtype
     * @return NDArray
     */
    private function sortAxisOp(string $funcName, int $axis, SortKind $kind, DType $dtype): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

        $shape = Lib::createShapeArray($this->shape);
        $strides = Lib::createShapeArray($this->strides);

        $status = $ffi->$funcName(
            $this->handle,
            $this->offset,
            $shape,
            $strides,
            count($this->shape),
            $axis,
            $kind->value,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new NDArray($outHandle, $this->shape, $dtype);
    }

    /**
     * Perform flattened sort/argsort. Returns 1D array.
     *
     * @param string $funcName FFI function name (ndarray_sort_flat or ndarray_argsort_flat)
     * @param SortKind $kind Sorting algorithm
     * @param DType $dtype Output dtype
     * @return NDArray
     */
    private function sortFlatOp(string $funcName, SortKind $kind, DType $dtype): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

        $shape = Lib::createShapeArray($this->shape);
        $strides = Lib::createShapeArray($this->strides);

        $status = $ffi->$funcName(
            $this->handle,
            $this->offset,
            $shape,
            $strides,
            count($this->shape),
            $kind->value,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        $size = (int) array_product($this->shape);
        return new NDArray($outHandle, [$size], $dtype);
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
            Lib::addr($outIndicesHandle)
        );

        Lib::checkStatus($status);

        $axisNorm = $axis < 0 ? count($this->shape) + $axis : $axis;
        $outShape = $this->shape;
        $outShape[$axisNorm] = $k;

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
            Lib::addr($outIndicesHandle)
        );

        Lib::checkStatus($status);

        return [
            'values' => new NDArray($outValuesHandle, [$k], $this->dtype),
            'indices' => new NDArray($outIndicesHandle, [$k], DType::Int64),
        ];
    }

    /**
     * Compute the output shape after reducing along an axis.
     *
     * @param array<int> $shape Input shape
     * @param int $axis Axis to reduce along
     * @param bool $keepdims Whether to keep reduced dimensions
     * @return array<int>
     */
    private function computeAxisOutputShape(array $shape, int $axis, bool $keepdims): array
    {
        $ndim = count($shape);
        if ($axis < 0) {
            $axis = $ndim + $axis;
        }

        if ($keepdims) {
            return array_map(fn($dim, $i) => $i === $axis ? 1 : $dim, $shape, array_keys($shape));
        }

        return array_values(array_filter($shape, fn($i) => $i !== $axis, ARRAY_FILTER_USE_KEY));
    }
}
