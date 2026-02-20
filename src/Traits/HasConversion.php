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
     * @param CData    $dst         Destination typed C buffer
     * @param null|int $maxElements Maximum elements the destination can hold
     *
     * @return int Number of elements copied
     */
    public function copyToBuffer(CData $dst, ?int $maxElements = null): int
    {
        $size = $this->size;
        if (0 === $size) {
            return 0;
        }

        $maxElements ??= $size;
        if ($maxElements < $size) {
            throw new ShapeException(
                "Destination buffer too small: requires {$size} elements, got {$maxElements}"
            );
        }

        $ffi = Lib::get();
        $meta = $this->viewMetadata()->toCData();
        $outLen = $ffi->new('size_t');

        $status = $ffi->ndarray_get_data(
            $this->handle,
            Lib::addr($meta),
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
        if (0 === $nbytes) {
            return '';
        }

        $ffi = Lib::get();
        $buffer = $ffi->new("uint8_t[{$nbytes}]");
        $this->copyToBuffer($buffer, $this->size);

        return \FFI::string($buffer, $nbytes);
    }

    /**
     * Convert 0-dimensional array to scalar.
     *
     * Returns float, int, or bool depending on the array's dtype.
     * Throws if the array is not 0-dimensional.
     */
    public function toScalar(): bool|float|int
    {
        if (0 !== $this->ndim()) {
            throw new ShapeException(
                'toScalar requires a 0-dimensional array, got array with '.$this->ndim().' dimensions'
            );
        }

        $ffi = Lib::get();
        $meta = $this->viewMetadata()->toCData();
        $out = $ffi->new($this->dtype->ffiType());

        $status = $ffi->ndarray_scalar(
            $this->handle,
            Lib::addr($meta),
            \FFI::addr($out),
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
     * @return array<bool|float|int>|bool|float|int
     */
    public function toFlatArray(): array|bool|float|int
    {
        if (0 === $this->ndim()) {
            return $this->toScalar();
        }

        return $this->fetchFlatData();
    }

    /**
     * Convert to nested PHP array shape.
     *
     * Uses flat typed extraction from Rust + iterative nesting in PHP.
     *
     * @return array<mixed>|bool|float|int Returns array for N-dimensional arrays, scalar for 0-dimensional
     */
    public function toArray(): array|bool|float|int
    {
        if (0 === $this->ndim()) {
            return $this->toScalar();
        }

        $flat = $this->fetchFlatData();
        if (1 === $this->ndim()) {
            return $flat;
        }

        return $this->nestFromFlatIterative($flat, $this->shape());
    }

    /**
     * Create a deep copy of the array (or view).
     *
     * The returned array is always C-contiguous and owns its data.
     */
    public function copy(): self
    {
        $ffi = Lib::get();

        $meta = $this->viewMetadata()->toCData();
        $outHandle = $ffi->new('struct NdArrayHandle*');

        $status = $ffi->ndarray_copy(
            $this->handle,
            Lib::addr($meta),
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new self($outHandle, $this->shape(), $this->dtype);
    }

    /**
     * Cast array to a different data type.
     *
     * Returns a new array with the specified dtype. If the target dtype
     * is the same as the current dtype, this is equivalent to copy().
     *
     * @param DType $dtype Target data type
     *
     * @return self New array with converted data
     */
    public function astype(DType $dtype): self
    {
        if ($this->dtype === $dtype) {
            return $this->copy();
        }

        $ffi = Lib::get();
        $meta = $this->viewMetadata()->toCData();
        $outHandle = $ffi->new('struct NdArrayHandle*');

        $status = $ffi->ndarray_astype(
            $this->handle,
            Lib::addr($meta),
            $dtype->value,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new self($outHandle, $this->shape(), $dtype);
    }

    /**
     * Fetch flattened view data from Rust using a single FFI call.
     *
     * @return array<bool|float|int>
     */
    private function fetchFlatData(): array
    {
        $ffi = Lib::get();
        $size = $this->size;
        $allocLen = max(1, $size);
        $meta = $this->viewMetadata()->toCData();
        $outLen = $ffi->new('size_t');

        $ctype = match ($this->dtype) {
            DType::Bool => 'uint8_t',
            default => $this->dtype->ffiType(),
        };
        $buffer = $ffi->new("{$ctype}[{$allocLen}]");

        $status = $ffi->ndarray_get_data(
            $this->handle,
            Lib::addr($meta),
            $buffer,
            $size,
            Lib::addr($outLen),
        );

        Lib::checkStatus($status);

        $len = min((int) $outLen->cdata, $size);
        if (0 === $len) {
            return [];
        }

        $out = [];

        switch ($this->dtype) {
            case DType::Float64:
            case DType::Float32:
                for ($i = 0; $i < $len; ++$i) {
                    $out[] = (float) $buffer[$i];
                }

                break;

            case DType::Bool:
                for ($i = 0; $i < $len; ++$i) {
                    $out[] = ((int) $buffer[$i]) !== 0;
                }

                break;

            default:
                for ($i = 0; $i < $len; ++$i) {
                    $out[] = (int) $buffer[$i];
                }

                break;
        }

        return $out;
    }

    /**
     * Build nested arrays from a flat C-order vector using iterative chunking.
     *
     * @param array<bool|float|int> $flat
     * @param array<int>            $shape
     *
     * @return array<mixed>
     */
    private function nestFromFlatIterative(array $flat, array $shape): array
    {
        $level = $flat;
        $ndim = \count($shape);

        for ($dim = $ndim - 1; $dim >= 1; --$dim) {
            $chunkSize = $shape[$dim];
            if ($chunkSize <= 0) {
                return [];
            }

            $next = [];
            $count = \count($level);
            $idx = 0;
            while ($idx < $count) {
                $chunk = [];
                for ($j = 0; $j < $chunkSize; ++$j) {
                    $chunk[] = $level[$idx++];
                }
                $next[] = $chunk;
            }
            $level = $next;
        }

        return $level;
    }
}
