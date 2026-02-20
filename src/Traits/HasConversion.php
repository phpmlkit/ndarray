<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Traits;

use FFI;
use FFI\CData;
use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\Exceptions\ShapeException;
use PhpMlKit\NDArray\FFI\Lib;

/**
 * Conversion methods for transforming NDArray data into PHP types.
 *
 * Handles both root arrays and strided views using a unified FFI interface.
 */
trait HasConversion
{
    /**
     * Copy flattened C-order data into a caller-allocated C buffer.
     *
     * @param CData $dst Destination typed C buffer
     * @param int|null $maxElements Maximum elements the destination can hold
     * @return int Number of elements copied
     */
    public function copyToBuffer(CData $dst, ?int $maxElements = null): int
    {
        $size = $this->size;
        if ($size === 0) {
            return 0;
        }

        $maxElements ??= $size;
        if ($maxElements < $size) {
            throw new ShapeException(
                "Destination buffer too small: requires {$size} elements, got {$maxElements}"
            );
        }

        $ffi = Lib::get();
        $cShape = Lib::createShapeArray($this->shape);
        $cStrides = Lib::createShapeArray($this->strides);
        $outLen = Lib::createBox('size_t');

        $status = $ffi->ndarray_get_data(
            $this->handle,
            $this->offset,
            $cShape,
            $cStrides,
            $this->ndim,
            $dst,
            $maxElements,
            Lib::addr($outLen),
        );

        Lib::checkStatus($status);

        return (int) min((int) $outLen->cdata, $maxElements);
    }

    /**
     * Return raw bytes of the array/view in C-order.
     */
    public function tobytes(): string
    {
        $nbytes = $this->nbytes();
        if ($nbytes === 0) {
            return '';
        }

        $ffi = Lib::get();
        $buffer = $ffi->new("uint8_t[{$nbytes}]");
        $this->copyToBuffer($buffer, $this->size);

        return FFI::string($buffer, $nbytes);
    }

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
     * Convert to flat PHP data in C-order.
     *
     * For 0-dimensional arrays, returns a scalar.
     *
     * @return array|float|int|bool
     */
    public function toFlatArray(): array|float|int|bool
    {
        if ($this->ndim === 0) {
            return $this->toScalar();
        }

        return $this->fetchFlatData();
    }

    /**
     * Convert to nested PHP array shape.
     *
     * Uses flat typed extraction from Rust + iterative nesting in PHP.
     *
     * @return array|float|int|bool Returns array for N-dimensional arrays, scalar for 0-dimensional
     */
    public function toArray(): array|float|int|bool
    {
        if ($this->ndim === 0) {
            return $this->toScalar();
        }

        $flat = $this->fetchFlatData();
        if ($this->ndim === 1) {
            return $flat;
        }

        return $this->nestFromFlatIterative($flat, $this->shape);
    }

    /**
     * Fetch flattened view data from Rust using a single FFI call.
     *
     * @return array<int|float|bool>
     */
    private function fetchFlatData(): array
    {
        $ffi = Lib::get();
        $size = $this->size;
        $allocLen = max(1, $size);

        $cShape = Lib::createShapeArray($this->shape);
        $cStrides = Lib::createShapeArray($this->strides);
        $outLen = Lib::createBox('size_t');

        $ctype = match ($this->dtype) {
            DType::Bool => 'uint8_t',
            default => $this->dtype->ffiType(),
        };
        $buffer = $ffi->new("{$ctype}[{$allocLen}]");

        $status = $ffi->ndarray_get_data(
            $this->handle,
            $this->offset,
            $cShape,
            $cStrides,
            $this->ndim,
            $buffer,
            $size,
            Lib::addr($outLen),
        );

        Lib::checkStatus($status);

        $len = min((int) $outLen->cdata, $size);
        if ($len === 0) {
            return [];
        }

        $out = [];

        switch ($this->dtype) {
            case DType::Float64:
            case DType::Float32:
                for ($i = 0; $i < $len; $i++) {
                    $out[] = (float) $buffer[$i];
                }
                break;
            case DType::Bool:
                for ($i = 0; $i < $len; $i++) {
                    $out[] = ((int) $buffer[$i]) !== 0;
                }
                break;
            default:
                for ($i = 0; $i < $len; $i++) {
                    $out[] = (int) $buffer[$i];
                }
                break;
        }

        return $out;
    }

    /**
     * Build nested arrays from a flat C-order vector using iterative chunking.
     *
     * @param array<int|float|bool> $flat
     * @param array<int> $shape
     * @return array
     */
    private function nestFromFlatIterative(array $flat, array $shape): array
    {
        $level = $flat;
        $ndim = count($shape);

        for ($dim = $ndim - 1; $dim >= 1; $dim--) {
            $chunkSize = $shape[$dim];
            if ($chunkSize <= 0) {
                return [];
            }

            $next = [];
            $count = count($level);
            $idx = 0;
            while ($idx < $count) {
                $chunk = [];
                for ($j = 0; $j < $chunkSize; $j++) {
                    $chunk[] = $level[$idx++];
                }
                $next[] = $chunk;
            }
            $level = $next;
        }

        return $level;
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
