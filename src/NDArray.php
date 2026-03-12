<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray;

use FFI\CData;
use PhpMlKit\NDArray\FFI\Lib;
use PhpMlKit\NDArray\Traits\CanBePrinted;
use PhpMlKit\NDArray\Traits\CreatesArrays;
use PhpMlKit\NDArray\Traits\HasArrayAccess;
use PhpMlKit\NDArray\Traits\HasComparison;
use PhpMlKit\NDArray\Traits\HasConversion;
use PhpMlKit\NDArray\Traits\HasIndexing;
use PhpMlKit\NDArray\Traits\HasLinearAlgebra;
use PhpMlKit\NDArray\Traits\HasLogical;
use PhpMlKit\NDArray\Traits\HasMath;
use PhpMlKit\NDArray\Traits\HasOps;
use PhpMlKit\NDArray\Traits\HasReductions;
use PhpMlKit\NDArray\Traits\HasShapeOps;
use PhpMlKit\NDArray\Traits\HasSlicing;
use PhpMlKit\NDArray\Traits\HasStacking;

/**
 * N-dimensional array class with PHP-managed view metadata and Rust-managed data.
 *
 * Views share the same Rust handle as their parent. Only root arrays (where
 * $base is null) call ndarray_free() on destruction. Views keep their root
 * alive through PHP's reference counting via the $base chain.
 *
 * @implements \ArrayAccess<int|string, bool|float|int|self>
 * @implements \IteratorAggregate<int, bool|float|int|self>
 */
class NDArray implements \ArrayAccess, \Stringable, \IteratorAggregate
{
    use CanBePrinted;
    use CreatesArrays;
    use HasArrayAccess;
    use HasComparison;
    use HasConversion;
    use HasIndexing;
    use HasLinearAlgebra;
    use HasLogical;
    use HasMath;
    use HasOps;
    use HasReductions;
    use HasShapeOps;
    use HasSlicing;
    use HasStacking;

    /**
     * Private constructor — use factory methods.
     *
     * @param CData         $handle Opaque pointer to Rust NDArrayWrapper
     * @param ArrayMetadata $meta   View metadata
     * @param DType         $dtype  Data type
     * @param null|self     $base   Parent array if this is a view
     */
    protected function __construct(
        protected readonly CData $handle,
        protected readonly ArrayMetadata $meta,
        protected readonly DType $dtype,
        protected readonly ?self $base = null,
    ) {}

    /**
     * Destructor — only root arrays free Rust memory.
     */
    public function __destruct()
    {
        if (null === $this->base) {
            Lib::get()->ndarray_free($this->handle);
        }
    }

    /**
     * Get shape.
     *
     * @return array<int>
     */
    public function shape(): array
    {
        return $this->meta->shape;
    }

    /**
     * Get strides.
     *
     * @return array<int>
     */
    public function strides(): array
    {
        return $this->meta->strides;
    }

    /**
     * Get number of dimensions.
     */
    public function ndim(): int
    {
        return $this->meta->ndim;
    }

    /**
     * Get total number of elements.
     */
    public function size(): int
    {
        return $this->meta->size;
    }

    /**
     * Get data type.
     */
    public function dtype(): DType
    {
        return $this->dtype;
    }

    /**
     * Get item size in bytes.
     */
    public function itemsize(): int
    {
        return $this->dtype->itemSize();
    }

    /**
     * Get total bytes consumed.
     */
    public function nbytes(): int
    {
        return $this->meta->size * $this->itemsize();
    }

    /**
     * Get the internal handle (for advanced FFI usage).
     *
     * @internal
     */
    public function handle(): CData
    {
        return $this->handle;
    }

    /**
     * Get view metadata (shape, strides, offset, ndim).
     *
     * @internal
     */
    public function meta(): ArrayMetadata
    {
        return $this->meta;
    }

    /**
     * Get view offset relative to root storage.
     *
     * @internal
     */
    public function offset(): int
    {
        return $this->meta->offset;
    }

    /**
     * Whether this array is a view of another array.
     */
    public function isView(): bool
    {
        return null !== $this->base;
    }

    /**
     * Check if the array is C-contiguous (row-major).
     */
    public function isContiguous(): bool
    {
        $expected = ArrayMetadata::computeStrides($this->meta->shape);

        return $this->meta->strides === $expected;
    }

    /**
     * Get iterator for foreach loops.
     *
     * 1D arrays: yields scalar values
     * 2D+ arrays: yields row views (along first axis)
     *
     * @return \Generator<int, bool|float|int|self>
     */
    public function getIterator(): \Generator
    {
        if (1 === $this->ndim()) {
            foreach ($this->flat() as $value) {
                yield $value;
            }
        } else {
            $shape = $this->shape();
            for ($i = 0; $i < $shape[0]; ++$i) {
                yield $this->slice([$i]);
            }
        }
    }

    /**
     * Get a 1-D iterator over the array.
     *
     * Returns a FlatIterator that provides 1-D access to the array elements
     * in C-contiguous (row-major) order. Uses hybrid approach:
     * - Arrays < 100k elements: Batch extraction (fast)
     * - Arrays >= 100k elements: Chunked extraction (memory efficient)
     *
     * @return FlatIterator Iterator over flattened array
     */
    public function flat(): FlatIterator
    {
        return new FlatIterator($this);
    }
}
