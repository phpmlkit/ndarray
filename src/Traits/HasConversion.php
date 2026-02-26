<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Traits;

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
     * Return raw bytes of the array/view in C-order.
     */
    public function toBytes(): string
    {
        $nbytes = $this->nbytes();
        if (0 === $nbytes) {
            return '';
        }

        $ffi = Lib::get();
        $buffer = $ffi->new("uint8_t[{$nbytes}]");
        $this->intoBuffer($buffer, 0, $this->size);

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
        $out = $this->dtype->createCValue();

        $status = $ffi->ndarray_as_scalar(
            $this->handle,
            Lib::addr($meta),
            \FFI::addr($out),
        );

        Lib::checkStatus($status);

        return $this->dtype->castFromCValue($out->cdata);
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

        $flat = $this->getData(0, $this->size);
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
     * Copy flattened C-order data into a caller-allocated C buffer.
     *
     * @param CData    $buffer Destination typed C buffer
     * @param int      $start  Starting element offset (0-indexed)
     * @param int      $len    Number of elements to copy
     *
     * @return int Number of elements copied
     */
    public function intoBuffer(CData $buffer, int $start, int $len): int
    {
        if (0 === $len) {
            return 0;
        }

        $ffi = Lib::get();
        $meta = $this->viewMetadata()->toCData();
        $outLen = $ffi->new('size_t');

        $status = $ffi->ndarray_get_data(
            $this->handle,
            Lib::addr($meta),
            $start,
            $len,
            $buffer,
            Lib::addr($outLen),
        );

        Lib::checkStatus($status);

        return (int) $outLen->cdata;
    }

    /**
     * Fetch a range of flattened view data from Rust.
     *
     * @param int $start Starting element offset (0-indexed)
     * @param int $len   Number of elements to fetch
     *
     * @return array<bool|float|int>
     */
    public function getData(int $start, int $len): array
    {
        if ($len <= 0) {
            return [];
        }

        $buffer = $this->dtype->createCArray($len);

        $actualLen = $this->intoBuffer($buffer, $start, $len);

        $out = [];

        for ($i = 0; $i < $actualLen; ++$i) {
            $out[] = $this->dtype->castFromCValue($buffer[$i]);
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
