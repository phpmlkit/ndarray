<?php

declare(strict_types=1);

namespace NDArray\Traits;

use NDArray\DType;
use NDArray\NDArray;
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
            $result = $this->performScalarReduction('ndarray_sum', $this->dtype);
            return $this->extractScalarValue($result);
        }
        return $this->performAxisReduction('ndarray_sum_axis', $axis, $keepdims, $this->dtype);
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
            $result = $this->performScalarReduction('ndarray_mean', DType::Float64);
            return $this->extractScalarValue($result);
        }
        return $this->performAxisReduction('ndarray_mean_axis', $axis, $keepdims, DType::Float64);
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
            $result = $this->performScalarReduction('ndarray_min', $this->dtype);
            return $this->extractScalarValue($result);
        }
        return $this->performAxisReduction('ndarray_min_axis', $axis, $keepdims, $this->dtype);
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
            $result = $this->performScalarReduction('ndarray_max', $this->dtype);
            return $this->extractScalarValue($result);
        }
        return $this->performAxisReduction('ndarray_max_axis', $axis, $keepdims, $this->dtype);
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
            $result = $this->performScalarReduction('ndarray_argmin', DType::Int64);
            return (int) $this->extractScalarValue($result);
        }
        return $this->performAxisReduction('ndarray_argmin_axis', $axis, $keepdims, DType::Int64);
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
            $result = $this->performScalarReduction('ndarray_argmax', DType::Int64);
            return (int) $this->extractScalarValue($result);
        }
        return $this->performAxisReduction('ndarray_argmax_axis', $axis, $keepdims, DType::Int64);
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
            $result = $this->performScalarReduction('ndarray_product', $this->dtype);
            return $this->extractScalarValue($result);
        }
        return $this->performAxisReduction('ndarray_product_axis', $axis, $keepdims, $this->dtype);
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
            return $this->performCumulativeReduction('ndarray_cumsum');
        }
        return $this->performCumulativeReductionAxis('ndarray_cumsum_axis', $axis);
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
            return $this->performCumulativeReduction('ndarray_cumprod');
        }
        return $this->performCumulativeReductionAxis('ndarray_cumprod_axis', $axis);
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
            $result = $this->performScalarReductionDdof('ndarray_var', DType::Float64, $ddof);
            return $this->extractScalarValue($result);
        }
        return $this->performAxisReductionDdof('ndarray_var_axis', $axis, $keepdims, DType::Float64, $ddof);
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
            $result = $this->performScalarReductionDdof('ndarray_std', DType::Float64, $ddof);
            return $this->extractScalarValue($result);
        }
        return $this->performAxisReductionDdof('ndarray_std_axis', $axis, $keepdims, DType::Float64, $ddof);
    }

    /**
     * Extract scalar value from a 0-dimensional array.
     *
     * @param NDArray $array 0-dimensional array
     * @return float|int
     */
    private function extractScalarValue(NDArray $array): float|int
    {
        // For 0-dimensional arrays, toArray() returns the scalar directly
        $value = $array->toArray();
        return is_array($value) ? ($value[0] ?? 0) : $value;
    }

    /**
     * Perform a scalar reduction and return the resulting 0-dimensional array.
     *
     * @param string $funcName FFI function name
     * @param DType $dtype Expected dtype of result
     * @return NDArray
     */
    private function performScalarReduction(string $funcName, DType $dtype): NDArray
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

        // Result is 0-dimensional (scalar)
        return new NDArray($outHandle, [], $dtype);
    }

    /**
     * Perform a scalar reduction with ddof parameter.
     *
     * @param string $funcName FFI function name
     * @param DType $dtype Expected dtype of result
     * @param int $ddof Delta degrees of freedom
     * @return NDArray
     */
    private function performScalarReductionDdof(string $funcName, DType $dtype, int $ddof): NDArray
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
            (float) $ddof,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new NDArray($outHandle, [], $dtype);
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
    private function performAxisReduction(string $funcName, int $axis, bool $keepdims, DType $dtype): NDArray
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
    private function performAxisReductionDdof(string $funcName, int $axis, bool $keepdims, DType $dtype, int $ddof): NDArray
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
    private function performCumulativeReduction(string $funcName): NDArray
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
    private function performCumulativeReductionAxis(string $funcName, int $axis): NDArray
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

    // =========================================================================
    // Static Methods
    // =========================================================================

    public static function sumArray(NDArray $arr, ?int $axis = null, bool $keepdims = false): NDArray|float
    {
        return $arr->sum($axis, $keepdims);
    }

    public static function meanArray(NDArray $arr, ?int $axis = null, bool $keepdims = false): NDArray|float
    {
        return $arr->mean($axis, $keepdims);
    }

    public static function minArray(NDArray $arr, ?int $axis = null, bool $keepdims = false): NDArray|float
    {
        return $arr->min($axis, $keepdims);
    }

    public static function maxArray(NDArray $arr, ?int $axis = null, bool $keepdims = false): NDArray|float
    {
        return $arr->max($axis, $keepdims);
    }

    public static function argminArray(NDArray $arr, ?int $axis = null, bool $keepdims = false): NDArray|int
    {
        return $arr->argmin($axis, $keepdims);
    }

    public static function argmaxArray(NDArray $arr, ?int $axis = null, bool $keepdims = false): NDArray|int
    {
        return $arr->argmax($axis, $keepdims);
    }

    public static function productArray(NDArray $arr, ?int $axis = null, bool $keepdims = false): NDArray|float
    {
        return $arr->product($axis, $keepdims);
    }

    public static function cumsumArray(NDArray $arr, ?int $axis = null): NDArray
    {
        return $arr->cumsum($axis);
    }

    public static function cumprodArray(NDArray $arr, ?int $axis = null): NDArray
    {
        return $arr->cumprod($axis);
    }

    public static function varArray(NDArray $arr, ?int $axis = null, int $ddof = 0, bool $keepdims = false): NDArray|float
    {
        return $arr->var($axis, $ddof, $keepdims);
    }

    public static function stdArray(NDArray $arr, ?int $axis = null, int $ddof = 0, bool $keepdims = false): NDArray|float
    {
        return $arr->std($axis, $ddof, $keepdims);
    }
}
