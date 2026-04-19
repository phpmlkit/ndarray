<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Traits;

use FFI\CData;
use PhpMlKit\NDArray\ArrayMetadata;
use PhpMlKit\NDArray\Complex;
use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\Exceptions\NDArrayException;
use PhpMlKit\NDArray\FFI\Lib;
use PhpMlKit\NDArray\NDArray;

/**
 * Shared helpers for calling the Rust FFI layer from NDArray traits.
 *
 * Wraps common call shapes (unary/binary array ops, scalar reductions) and
 * pairs {@see scalarToBuffer} with {@see bufferToScalar} so PHP values and
 * raw C buffers stay in sync at the boundary.
 */
trait HasOps
{
    /**
     * Call a unary FFI function that allocates a new array handle and fills output metadata.
     *
     * Signature: `(handle, metadata, ...$extraArgs, out_handle, out_dtype, out_ndim, out_shape, max_ndim)`.
     * `BackedEnum` arguments (e.g. {@see DType}) are passed as their raw integer values.
     *
     * @param string $funcName     FFI symbol
     * @param mixed  ...$extraArgs Inserted after metadata (scalars, axes, packed scalar + dtype, …)
     */
    protected function unaryOp(string $funcName, mixed ...$extraArgs): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');
        [$outDtypeBuf, $outNdimBuf, $outShapeBuf] = Lib::createOutputMetadataBuffers();
        $meta = $this->meta()->toCData();

        // Normalize extra args: convert BackedEnum to their values
        $normalizedArgs = array_map(
            static fn ($arg) => $arg instanceof \BackedEnum ? $arg->value : $arg,
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

        return new NDArray($outHandle, new ArrayMetadata($shape), $dtype);
    }

    /**
     * Call a binary FFI function with this array as LHS and another {@see NDArray} as RHS.
     *
     * Signature: `(a_handle, a_meta, b_handle, b_meta, out_handle, out_dtype, out_ndim, out_shape, max_ndim)`.
     */
    protected function binaryOp(
        string $funcName,
        NDArray $other,
    ): NDArray {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');
        [$outDtypeBuf, $outNdimBuf, $outShapeBuf] = Lib::createOutputMetadataBuffers();

        $aMeta = $this->meta()->toCData();
        $bMeta = $other->meta()->toCData();
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

        return new NDArray($outHandle, new ArrayMetadata($shape), $dtype);
    }

    /**
     * Run a reduction that returns one element (sum, mean, argmax, …) and decode it in PHP.
     *
     * FFI: `(handle, metadata, ...$extraArgs, out_value, out_dtype)` — Rust writes the packed
     * scalar and dtype; this method passes a `uint8_t[16]` for `out_value` then {@see bufferToScalar}.
     *
     * @param string $funcName     FFI symbol (e.g. `ndarray_sum`)
     * @param mixed  ...$extraArgs Placed before `out_value` (e.g. `ddof` for var/std)
     */
    protected function scalarReductionOp(string $funcName, mixed ...$extraArgs): Complex|float|int
    {
        $ffi = Lib::get();
        $outValue = $ffi->new('uint8_t[16]');
        $outDtype = $ffi->new('uint8_t');

        $meta = $this->meta()->toCData();
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

        return $this->bufferToScalar($outValue, $dtype);
    }

    /**
     * Encode a PHP scalar as a typed FFI buffer plus {@see DType} for `*_scalar` entry points.
     *
     * Uses {@see DType::fromValue} to pick the dtype, allocates a one-element buffer (or two
     * floats/doubles for complex), and returns `[buffer, dtype]`. Spread into {@see unaryOp}:
     * `unaryOp('ndarray_add_scalar', ...$this->scalarToBuffer($x))` — `unaryOp` turns the enum
     * into its `uint8_t` FFI value.
     *
     * @return array{0: CData, 1: DType}
     */
    protected function scalarToBuffer(Complex|float|int $value): array
    {
        $dtype = DType::fromValue($value);
        $ffi = Lib::get();

        if ($dtype->isComplex()) {
            \assert($value instanceof Complex);
            $buffer = $ffi->new("{$dtype->ffiType()}[2]");
            $buffer[0] = $value->real;
            $buffer[1] = $value->imag;

            return [$buffer, $dtype];
        }

        $buffer = $ffi->new("{$dtype->ffiType()}[1]");
        $buffer[0] = $dtype->isBool() ? ($value ? 1 : 0) : $value;

        return [$buffer, $dtype];
    }

    /**
     * Decode a raw scalar buffer from Rust into a PHP `float`, `int`, or {@see Complex}.
     *
     * Inverse of {@see scalarToBuffer}: used when the native side writes `out_value` (up to 16 bytes,
     * e.g. complex128) and sets `out_dtype`. Reads layout according to `dtype` via FFI casts.
     *
     * @param CData $buffer memory Rust filled; typically `uint8_t[16]` from a scalar reduction
     */
    private function bufferToScalar(CData $buffer, DType $dtype): Complex|float|int
    {
        $ffi = Lib::get();
        $base = Lib::addr($buffer);

        return match ($dtype) {
            DType::Float64 => $ffi->cast('double*', $base)[0],
            DType::Float32 => $ffi->cast('float*', $base)[0],
            DType::Int64, DType::Int32, DType::Int16, DType::Int8 => $ffi->cast('int64_t*', $base)[0],
            DType::UInt64, DType::UInt32, DType::UInt16, DType::UInt8 => $ffi->cast('uint64_t*', $base)[0],
            DType::Complex64 => new Complex($ffi->cast('float*', $base)[0], $ffi->cast('float*', $base)[1]),
            DType::Complex128 => new Complex($ffi->cast('double*', $base)[0], $ffi->cast('double*', $base)[1]),
            default => 0,
        };
    }
}
