<?php

declare(strict_types=1);

namespace NDArray;

use ArrayAccess;
use FFI\CData;
use NDArray\FFI\Lib;
use NDArray\Traits\CreatesArrays;
use NDArray\Traits\HasArrayAccess;
use NDArray\Traits\HasComparison;
use NDArray\Traits\HasConversion;
use NDArray\Traits\HasIndexing;
use NDArray\Traits\HasMath;
use NDArray\Traits\HasReductions;
use NDArray\Traits\HasShapeOps;
use NDArray\Traits\HasSlicing;
use NDArray\Traits\HasStringable;

/**
 * N-dimensional array class with PHP-managed view metadata and Rust-managed data.
 *
 * Views share the same Rust handle as their parent. Only root arrays (where
 * $base is null) call ndarray_free() on destruction. Views keep their root
 * alive through PHP's reference counting via the $base chain.
 *
 * @implements ArrayAccess<int|string, self|int|float|bool>
 */
class NDArray implements ArrayAccess
{
    use CreatesArrays;
    use HasIndexing;
    use HasArrayAccess;
    use HasConversion;
    use HasMath;
    use HasReductions;
    use HasShapeOps;
    use HasComparison;
    use HasStringable;
    use HasSlicing;

    /** Number of dimensions */
    protected int $ndim;

    /** Total number of elements in this array/view */
    protected int $size;

    /** Element strides per dimension (in element count, not bytes) */
    protected readonly array $strides;

    /**
     * Private constructor — use factory methods.
     *
     * @param CData $handle Opaque pointer to Rust NDArrayWrapper
     * @param array<int> $shape Shape of this array/view
     * @param DType $dtype Data type
     * @param array<int> $strides Element strides per dimension
     * @param int $offset Flat offset into root data
     * @param self|null $base Parent array if this is a view
     */
    protected function __construct(
        protected readonly CData $handle,
        protected array $shape,
        protected readonly DType $dtype,
        array $strides = [],
        protected readonly int $offset = 0,
        protected readonly ?self $base = null,
    ) {
        $this->ndim = count($shape);
        $this->size = (int) array_product($shape);
        $this->strides = $strides ?: self::computeStrides($shape);
    }

    /**
     * Destructor — only root arrays free Rust memory.
     */
    public function __destruct()
    {
        if ($this->base === null && isset($this->handle)) {
            Lib::get()->ndarray_free($this->handle);
        }
    }

    // =========================================================================
    // Properties
    // =========================================================================

    /**
     * Get shape.
     *
     * @return array<int>
     */
    public function shape(): array
    {
        return $this->shape;
    }

    /**
     * Get strides.
     *
     * @return array<int>
     */
    public function strides(): array
    {
        return $this->strides;
    }

    /**
     * Get number of dimensions.
     */
    public function ndim(): int
    {
        return $this->ndim;
    }

    /**
     * Get total number of elements.
     */
    public function size(): int
    {
        return $this->size;
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
        return $this->size * $this->itemsize();
    }

    /**
     * Get the internal handle (for advanced FFI usage).
     *
     * @internal
     */
    public function getHandle(): CData
    {
        return $this->handle;
    }

    /**
     * Whether this array is a view of another array.
     */
    public function isView(): bool
    {
        return $this->base !== null;
    }

    /**
     * Check if the array is C-contiguous (row-major).
     */
    public function isContiguous(): bool
    {
        $expected = self::computeStrides($this->shape);
        return $this->strides === $expected;
    }

    // =========================================================================
    // Private Helpers
    // =========================================================================

    /**
     * Compute row-major strides from shape.
     *
     * For shape [3, 4, 5], strides are [20, 5, 1].
     *
     * @param array<int> $shape
     * @return array<int>
     */
    private static function computeStrides(array $shape): array
    {
        $ndim = count($shape);
        if ($ndim === 0) {
            return [];
        }

        $strides = array_fill(0, $ndim, 1);
        for ($i = $ndim - 2; $i >= 0; $i--) {
            $strides[$i] = $strides[$i + 1] * $shape[$i + 1];
        }

        return $strides;
    }
}
