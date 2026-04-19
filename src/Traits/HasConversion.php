<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Traits;

use FFI\CData;
use PhpMlKit\NDArray\Complex;
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
     * Convert 0-dimensional array to scalar.
     *
     * Returns float, int, or bool depending on the array's dtype.
     * Throws if the array is not 0-dimensional.
     */
    public function toScalar(): bool|Complex|float|int
    {
        if (0 !== $this->ndim()) {
            throw new ShapeException(
                'toScalar requires a 0-dimensional array, got array with '.$this->ndim().' dimensions'
            );
        }

        $ffi = Lib::get();
        $meta = $this->meta()->toCData();
        $out = $this->dtype->createCValue();

        $status = $ffi->ndarray_as_scalar(
            $this->handle,
            Lib::addr($meta),
            \FFI::addr($out),
        );

        Lib::checkStatus($status);

        if ($this->dtype->isComplex()) {
            return $this->dtype->castFromCValue($out);
        }

        return $this->dtype->castFromCValue($out->cdata);
    }

    /**
     * Convert to nested PHP array shape.
     *
     * Uses flat typed extraction from Rust + iterative nesting in PHP.
     *
     * @return array<mixed>|bool|float|int Returns array for N-dimensional arrays, scalar for 0-dimensional
     */
    public function toArray(): array|bool|Complex|float|int
    {
        if (0 === $this->ndim()) {
            return $this->toScalar();
        }

        $flat = $this->getData(0, $this->size());
        if (1 === $this->ndim()) {
            return $flat;
        }

        return $this->nestFromFlatIterative($flat, $this->shape());
    }

    /**
     * Copy flattened C-order data into a C buffer.
     *
     * If no buffer is provided, allocates a new C array of the appropriate type.
     * The returned CData is owned by PHP and will be garbage collected when no
     * longer referenced.
     *
     * @param null|CData $buffer Optional destination typed C buffer. If null, a new buffer is allocated.
     * @param int        $start  Starting element offset (0-indexed). Default: 0
     * @param null|int   $len    Number of elements to copy. Default: null (copy remaining elements)
     *
     * @return CData The buffer (either provided or newly allocated)
     */
    public function toBuffer(?CData $buffer = null, int $start = 0, ?int $len = null): CData
    {
        if (null === $len) {
            $len = $this->size() - $start;
        }

        $ffi = Lib::get();

        if (0 === $len || $len < 0) {
            return $buffer ?? $this->dtype->createCArray(1);
        }

        $buffer ??= $this->dtype->createCArray($len);
        $meta = $this->meta()->toCData();
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

        return $buffer;
    }

    /**
     * Copy flattened C-order data into a caller-allocated C buffer.
     *
     * @deprecated Use toBuffer() instead. intoBuffer() will be removed in a future version.
     *
     * @param CData    $buffer Destination typed C buffer
     * @param int      $start  Starting element offset (0-indexed). Default: 0
     * @param null|int $len    Number of elements to copy. Default: null (copy remaining elements)
     *
     * @return int Number of elements copied
     */
    public function intoBuffer(CData $buffer, int $start = 0, ?int $len = null): int
    {
        if (null === $len) {
            $len = $this->size() - $start;
        }

        if (0 === $len || $len < 0) {
            return 0;
        }

        $ffi = Lib::get();
        $meta = $this->meta()->toCData();
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
     * Return raw bytes of the array/view in C-order.
     *
     * The returned string contains the raw binary data in little-endian format.
     */
    public function toBytes(): string
    {
        $nbytes = $this->nbytes();
        if (0 === $nbytes) {
            return '';
        }

        $buffer = $this->toBuffer();

        return \FFI::string($buffer, $nbytes);
    }

    /**
     * Fetch a range of flattened view data from Rust.
     *
     * @param int $start Starting element offset (0-indexed)
     * @param int $len   Number of elements to fetch
     *
     * @return array<bool|Complex|float|int>
     */
    public function getData(int $start, int $len): array
    {
        if ($len <= 0) {
            return [];
        }

        $buffer = $this->toBuffer(null, $start, $len);

        $out = [];

        if ($this->dtype->isComplex()) {
            for ($i = 0; $i < $len; ++$i) {
                $out[] = new Complex((float) $buffer[$i * 2], (float) $buffer[$i * 2 + 1]);
            }
        } else {
            for ($i = 0; $i < $len; ++$i) {
                $out[] = $this->dtype->castFromCValue($buffer[$i]);
            }
        }

        return $out;
    }

    /**
     * Build nested arrays from a flat C-order vector using iterative chunking.
     *
     * @param array<bool|Complex|float|int> $flat
     * @param array<int>                    $shape
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
