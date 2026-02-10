<?php

declare(strict_types=1);

namespace NDArray\Traits;

use FFI;
use NDArray\FFI\Lib;

/**
 * Conversion methods for transforming NDArray data into PHP types.
 *
 * Handles both root arrays and strided views.
 */
trait HasConversion
{
    /**
     * Convert to PHP array.
     *
     * Uses strided view serialization when this is a view,
     * or full array serialization for root arrays.
     *
     * @return array
     */
    public function toArray(): array
    {
        $ffi = Lib::get();

        if ($this->base !== null) {
            return $this->viewToArray($ffi);
        }

        $outPtr = $ffi->new("char*");
        $outLen = $ffi->new("size_t");

        $status = $ffi->ndarray_to_json(
            $this->handle,
            Lib::addr($outPtr),
            Lib::addr($outLen),
            17
        );

        Lib::checkStatus($status);

        $json = FFI::string($outPtr, $outLen->cdata);

        $ffi->ndarray_free_string($outPtr);

        return json_decode($json, true, 512, JSON_THROW_ON_ERROR);
    }

    // =========================================================================
    // Private Helpers
    // =========================================================================

    /**
     * Serialize a view to a PHP array using strided JSON serialization.
     *
     * @param FFI $ffi
     * @return array
     */
    private function viewToArray(FFI $ffi): array
    {
        $cShape = Lib::createShapeArray($this->shape);
        $cStrides = Lib::createShapeArray($this->strides);

        $outPtr = $ffi->new("char*");
        $outLen = $ffi->new("size_t");

        $status = $ffi->ndarray_view_to_json(
            $this->handle,
            $this->offset,
            $cShape,
            $cStrides,
            $this->ndim,
            17,
            Lib::addr($outPtr),
            Lib::addr($outLen),
        );

        Lib::checkStatus($status);

        $json = FFI::string($outPtr, $outLen->cdata);

        $ffi->ndarray_free_string($outPtr);

        return json_decode($json, true, 512, JSON_THROW_ON_ERROR);
    }
}
