<?php

declare(strict_types=1);

namespace NDArray\Traits;

use FFI;
use NDArray\DType;
use NDArray\Exceptions\ShapeException;
use NDArray\FFI\Lib;

/**
 * Conversion methods for transforming NDArray data into PHP types.
 *
 * Handles both root arrays and strided views using a unified FFI interface.
 */
trait HasConversion
{
    /**
     * Convert 0-dimensional array to scalar.
     *
     * Returns float, int, or bool depending on the array's dtype.
     * Throws if the array is not 0-dimensional.
     *
     * @return float|int|bool
     */
    public function toScalar(): float|int|bool
    {
        if ($this->ndim !== 0) {
            throw new ShapeException(
                'toScalar requires a 0-dimensional array, got array with ' . $this->ndim . ' dimensions'
            );
        }

        $ffi = Lib::get();

        $cShape = Lib::createShapeArray($this->shape);
        $cStrides = Lib::createShapeArray($this->strides);

        $out = $ffi->new($this->dtype->ffiType());

        $status = $ffi->ndarray_scalar(
            $this->handle,
            $this->offset,
            $cShape,
            $cStrides,
            $this->ndim,
            FFI::addr($out),
        );

        Lib::checkStatus($status);

        $raw = $out->cdata;

        return match ($this->dtype) {
            DType::Float64, DType::Float32 => (float) $raw,
            DType::Bool => (bool) $raw,
            default => (int) $raw,
        };
    }

    /**
     * Convert to PHP array.
     *
     * Serializes the array or view to JSON using optimized ndarray view
     * extraction, then decodes to PHP array structure.
     *
     * @return array|float|int Returns array for N-dimensional arrays, scalar for 0-dimensional
     */
    public function toArray(): array|float|int
    {
        $ffi = Lib::get();

        $cShape = Lib::createShapeArray($this->shape);
        $cStrides = Lib::createShapeArray($this->strides);

        $outPtr = $ffi->new("char*");
        $outLen = Lib::createBox("size_t");

        $status = $ffi->ndarray_to_json(
            $this->handle,
            $this->offset,
            $cShape,
            $cStrides,
            $this->ndim,
            Lib::addr($outPtr),
            Lib::addr($outLen),
        );

        Lib::checkStatus($status);

        $json = FFI::string($outPtr, $outLen->cdata);

        $ffi->ndarray_free_string($outPtr);

        return json_decode($json, true, 512, JSON_THROW_ON_ERROR);
    }

    /**
     * Create a deep copy of the array (or view).
     *
     * The returned array is always C-contiguous and owns its data.
     *
     * @return self
     */
    public function copy(): self
    {
        $ffi = Lib::get();

        $cShape = Lib::createCArray('size_t', $this->shape);
        $cStrides = Lib::createCArray('size_t', $this->strides);
        $outHandle = $ffi->new("struct NdArrayHandle*");

        $status = $ffi->ndarray_copy(
            $this->handle,
            $this->offset,
            $cShape,
            $cStrides,
            $this->ndim,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new self($outHandle, $this->shape, $this->dtype);
    }

    /**
     * Cast array to a different data type.
     *
     * Returns a new array with the specified dtype. If the target dtype
     * is the same as the current dtype, this is equivalent to copy().
     *
     * @param DType $dtype Target data type
     * @return self New array with converted data
     */
    public function astype(DType $dtype): self
    {
        if ($this->dtype === $dtype) {
            return $this->copy();
        }

        $ffi = Lib::get();

        $cShape = Lib::createShapeArray($this->shape);
        $cStrides = Lib::createShapeArray($this->strides);
        $outHandle = $ffi->new("struct NdArrayHandle*");

        $status = $ffi->ndarray_astype(
            $this->handle,
            $this->offset,
            $cShape,
            $cStrides,
            $this->ndim,
            $dtype->value,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new self($outHandle, $this->shape, $dtype);
    }
}
