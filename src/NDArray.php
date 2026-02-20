<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray;

use FFI\CData;
use PhpMlKit\NDArray\Exceptions\NDArrayException;
use PhpMlKit\NDArray\FFI\Lib;
use PhpMlKit\NDArray\Traits\CreatesArrays;
use PhpMlKit\NDArray\Traits\HasArrayAccess;
use PhpMlKit\NDArray\Traits\HasComparison;
use PhpMlKit\NDArray\Traits\HasConversion;
use PhpMlKit\NDArray\Traits\HasIndexing;
use PhpMlKit\NDArray\Traits\HasLinearAlgebra;
use PhpMlKit\NDArray\Traits\HasMath;
use PhpMlKit\NDArray\Traits\HasOps;
use PhpMlKit\NDArray\Traits\HasReductions;
use PhpMlKit\NDArray\Traits\HasShapeOps;
use PhpMlKit\NDArray\Traits\HasSlicing;
use PhpMlKit\NDArray\Traits\HasStacking;
use PhpMlKit\NDArray\Traits\HasStringable;

/**
 * N-dimensional array class with PHP-managed view metadata and Rust-managed data.
 *
 * Views share the same Rust handle as their parent. Only root arrays (where
 * $base is null) call ndarray_free() on destruction. Views keep their root
 * alive through PHP's reference counting via the $base chain.
 *
 * @implements \ArrayAccess<int|string, bool|float|int|self>
 */
class NDArray implements \ArrayAccess
{
    use CreatesArrays;
    use HasArrayAccess;
    use HasComparison;
    use HasConversion;
    use HasIndexing;
    use HasLinearAlgebra;
    use HasMath;
    use HasOps;
    use HasReductions;
    use HasShapeOps;
    use HasSlicing;
    use HasStacking;
    use HasStringable;

    /** View metadata (shape, strides, offset, ndim). */
    protected readonly ViewMetadata $viewMetadata;

    /** Total number of elements in this array/view */
    protected int $size;

    /**
     * Private constructor — use factory methods.
     *
     * @param CData      $handle  Opaque pointer to Rust NDArrayWrapper
     * @param array<int> $shape   Shape of this array/view
     * @param DType      $dtype   Data type
     * @param array<int> $strides Element strides per dimension
     * @param int        $offset  Flat offset into root data
     * @param null|self  $base    Parent array if this is a view
     */
    protected function __construct(
        protected readonly CData $handle,
        array $shape,
        protected readonly DType $dtype,
        array $strides = [],
        int $offset = 0,
        protected readonly ?self $base = null,
    ) {
        $this->viewMetadata = new ViewMetadata($shape, $strides, $offset);
        $this->size = (int) array_product($shape);
    }

    /**
     * Destructor — only root arrays free Rust memory.
     */
    public function __destruct()
    {
        if (null === $this->base) {
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
        return $this->viewMetadata->shape;
    }

    /**
     * Get strides.
     *
     * @return array<int>
     */
    public function strides(): array
    {
        return $this->viewMetadata->strides;
    }

    /**
     * Get number of dimensions.
     */
    public function ndim(): int
    {
        return $this->viewMetadata->ndim;
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
    public function handle(): CData
    {
        return $this->handle;
    }

    /**
     * Get view metadata (shape, strides, offset, ndim).
     *
     * @internal
     */
    public function viewMetadata(): ViewMetadata
    {
        return $this->viewMetadata;
    }

    /**
     * Get view offset relative to root storage.
     *
     * @internal
     */
    public function getOffset(): int
    {
        return $this->viewMetadata->offset;
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
        $expected = ViewMetadata::computeStrides($this->viewMetadata->shape);

        return $this->viewMetadata->strides === $expected;
    }

    /**
     * Select values from x and y based on a boolean condition.
     *
     * @param bool|float|int|self $condition Bool NDArray or scalar condition
     * @param bool|float|int|self $x         Values where condition is true
     * @param bool|float|int|self $y         Values where condition is false
     */
    public static function where(bool|float|int|self $condition, bool|float|int|self $x, bool|float|int|self $y): self
    {
        $condArray = self::coerceWhereOperand($condition, DType::Bool);
        $xArray = self::coerceWhereOperand($x, null);
        $yArray = self::coerceWhereOperand($y, null);

        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');
        [$outDtypeBuf, $outNdimBuf, $outShapeBuf] = Lib::createOutputMetadataBuffers();

        $condMeta = $condArray->viewMetadata()->toCData();
        $xMeta = $xArray->viewMetadata()->toCData();
        $yMeta = $yArray->viewMetadata()->toCData();
        $status = $ffi->ndarray_where(
            $condArray->handle,
            Lib::addr($condMeta),
            $xArray->handle,
            Lib::addr($xMeta),
            $yArray->handle,
            Lib::addr($yMeta),
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
        if (null === $dtype) {
            throw new NDArrayException('Invalid dtype returned from Rust for where()');
        }

        return new self($outHandle, $shape, $dtype);
    }

    private static function coerceWhereOperand(bool|float|int|self $value, ?DType $forceDtype): self
    {
        if ($value instanceof self) {
            if (null !== $forceDtype && $value->dtype !== $forceDtype) {
                return $value->astype($forceDtype);
            }

            return $value;
        }

        $dtype = $forceDtype ?? DType::fromValue($value);

        return self::array([$value], $dtype);
    }
}
