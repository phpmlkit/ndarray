<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Traits;

use FFI;
use FFI\CData;
use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\Exceptions\NDArrayException;
use PhpMlKit\NDArray\FFI\Lib;
use PhpMlKit\NDArray\NDArray;

/**
 * Shared operation helpers for FFI-backed unary, binary, and scalar reduction ops.
 *
 * Centralizes argument marshaling and status handling so all traits can
 * reuse consistent call patterns and exception mapping.
 */
trait HasOps
{
    /**
     * Execute a unary FFI operation that returns an NDArray handle.
     *
     * All unary ops return metadata (dtype, ndim, shape) into caller-provided buffers.
     * Signature: (handle, offset, shape, strides, ndim, ...$extraArgs, out_handle, out_dtype, out_ndim, out_shape, max_ndim).
     *
     * @param string $funcName     FFI function name
     * @param mixed  ...$extraArgs Extra FFI args inserted before out_handle (e.g. scalar, axis)
     */
    protected function unaryOp(string $funcName, mixed ...$extraArgs): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');
        [$outDtypeBuf, $outNdimBuf, $outShapeBuf] = Lib::createOutputMetadataBuffers();
        $meta = $this->viewMetadata()->toCData();

        // Normalize extra args: convert BackedEnum to their values
        $normalizedArgs = array_map(
            static fn($arg) => $arg instanceof \BackedEnum ? $arg->value : $arg,
            $extraArgs
        );

        $args = [
            $this->handle,
            Lib::addr($meta),
            ...$normalizedArgs,
            Lib::addr($outHandle),
            Lib::addr($outDtypeBuf),
            Lib::addr($outNdimBuf),
            $outShapeBuf,
            Lib::MAX_NDIM,
        ];

        $status = $ffi->{$funcName}(...$args);

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
     * Execute a binary operation with NDArray RHS.
     *
     * FFI signature: (a..., b..., out_handle, out_dtype_ptr, out_ndim, out_shape, max_ndim).
     * dtype and shape are always read from Rust output metadata.
     *
     * @param string  $funcName FFI function for NDArray RHS
     * @param NDArray $other    RHS operand
     */
    protected function binaryOp(
        string $funcName,
        NDArray $other,
    ): NDArray {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');
        [$outDtypeBuf, $outNdimBuf, $outShapeBuf] = Lib::createOutputMetadataBuffers();

        $aMeta = $this->viewMetadata()->toCData();
        $bMeta = $other->viewMetadata()->toCData();
        $status = $ffi->{$funcName}(
            $this->handle,
            Lib::addr($aMeta),
            $other->handle(),
            Lib::addr($bMeta),
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
     * Execute a scalar reduction FFI operation that returns a single value.
     *
     * FFI signature: (handle, offset, shape, strides, ndim, ...$extraArgs, out_value, out_dtype).
     * Examples: ndarray_sum (no extra args), ndarray_var (ddof before out_value).
     *
     * @param string $funcName     FFI function name
     * @param mixed  ...$extraArgs Extra FFI args inserted before out_value (e.g. ddof)
     */
    protected function scalarReductionOp(string $funcName, mixed ...$extraArgs): float|int
    {
        $ffi = Lib::get();
        $outValue = $ffi->new('double');
        $outDtype = $ffi->new('uint8_t');

        $meta = $this->viewMetadata()->toCData();
        $args = [
            $this->handle,
            Lib::addr($meta),
            ...$extraArgs,
            Lib::addr($outValue),
            Lib::addr($outDtype),
        ];

        $status = $ffi->{$funcName}(...$args);
        Lib::checkStatus($status);

        $dtype = DType::tryFrom((int) $outDtype->cdata);
        if (null === $dtype) {
            throw new NDArrayException('Invalid dtype returned from Rust scalar reduction');
        }

        return $this->interpretScalarValue($ffi, $outValue, $dtype);
    }

    /**
     * Interpret 8-byte value buffer as PHP scalar based on dtype.
     *
     * @param \FFI  $ffi      FFI instance
     * @param CData $outValue 8-byte buffer (allocated as double)
     * @param DType $dtype    Result dtype from Rust
     */
    private function interpretScalarValue(\FFI $ffi, CData $outValue, DType $dtype): float|int
    {
        $addr = \FFI::addr($outValue);

        return match ($dtype) {
            DType::Float64, DType::Float32 => $outValue->cdata,
            DType::Int64, DType::Int32, DType::Int16, DType::Int8 => $ffi->cast('int64_t*', $addr)[0],
            DType::UInt64, DType::UInt32, DType::UInt16, DType::UInt8 => $ffi->cast('uint64_t*', $addr)[0],
            default => 0,
        };
    }
}
