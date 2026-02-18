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
use NDArray\Traits\HasLinearAlgebra;
use NDArray\Traits\HasMath;
use NDArray\Traits\HasOps;
use NDArray\Traits\HasReductions;
use NDArray\Traits\HasShapeOps;
use NDArray\Traits\HasSlicing;
use NDArray\Traits\HasStacking;
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
    use HasLinearAlgebra;
    use HasOps;
    use HasMath;
    use HasReductions;
    use HasShapeOps;
    use HasStacking;
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
     * Get view offset relative to root storage.
     *
     * @internal
     */
    public function getOffset(): int
    {
        return $this->offset;
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

    /**
     * Select values from x and y based on a boolean condition.
     *
     * @param self|int|float|bool $condition Bool NDArray or scalar condition
     * @param self|int|float|bool $x Values where condition is true
     * @param self|int|float|bool $y Values where condition is false
     * @return self
     */
    public static function where(self|int|float|bool $condition, self|int|float|bool $x, self|int|float|bool $y): self
    {
        $condArray = self::coerceWhereOperand($condition, DType::Bool);
        $xArray = self::coerceWhereOperand($x, null);
        $yArray = self::coerceWhereOperand($y, null);

        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');
        [$outDtypeBuf, $outNdimBuf, $outShapeBuf] = Lib::createOutputMetadataBuffers();

        $status = $ffi->ndarray_where(
            $condArray->handle,
            $condArray->offset,
            Lib::createShapeArray($condArray->shape),
            Lib::createCArray('size_t', $condArray->strides),
            $condArray->ndim,
            $xArray->handle,
            $xArray->offset,
            Lib::createShapeArray($xArray->shape),
            Lib::createCArray('size_t', $xArray->strides),
            $xArray->ndim,
            $yArray->handle,
            $yArray->offset,
            Lib::createShapeArray($yArray->shape),
            Lib::createCArray('size_t', $yArray->strides),
            $yArray->ndim,
            Lib::addr($outHandle),
            Lib::addr($outDtypeBuf),
            Lib::addr($outNdimBuf),
            $outShapeBuf,
            Lib::MAX_NDIM
        );

        Lib::checkStatus($status);

        $dtype = DType::tryFrom((int) $outDtypeBuf->cdata);
        $ndim = (int) $outNdimBuf->cdata;
        $shape = Lib::extractShapeFromPointer($outShapeBuf, $ndim);
        if ($dtype === null) {
            throw new \RuntimeException('Invalid dtype returned from Rust for where()');
        }

        return new self($outHandle, $shape, $dtype);
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

    /**
     * @param self|int|float|bool $value
     */
    private static function coerceWhereOperand(self|int|float|bool $value, ?DType $forceDtype): self
    {
        if ($value instanceof self) {
            if ($forceDtype !== null && $value->dtype !== $forceDtype) {
                return $value->astype($forceDtype);
            }
            return $value;
        }

        $dtype = $forceDtype ?? DType::fromValue($value);
        return self::array([$value], $dtype);
    }
}
