<?php

declare(strict_types=1);

namespace NDArray\Traits;

use FFI;
use NDArray\FFI\Lib;

/**
 * Conversion methods for transforming NDArray data into PHP types.
 *
 * Handles both root arrays and strided views using a unified FFI interface.
 */
trait HasConversion
{
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
}
