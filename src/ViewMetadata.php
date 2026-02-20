<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray;

use FFI\CData;
use PhpMlKit\NDArray\FFI\Lib;

/**
 * Holds shape, strides, offset and ndim for an array view.
 *
 * NDArray uses this instead of storing those fields directly. When FFI needs
 * a C ViewMetadata struct, call toCData() to build and fill it; use
 * Lib::addr($viewMetadata->toCData()) when passing to FFI.
 *
 * @internal this is an internal implementation detail, not part of the public API
 */
final class ViewMetadata
{
    public readonly int $ndim;

    /** @var array<int> */
    public readonly array $strides;

    /** Cached C struct and backing arrays so pointers stay valid. */
    private ?CData $cachedStruct = null;

    private ?CData $cachedShapeC = null;

    private ?CData $cachedStridesC = null;

    /**
     * @param array<int> $shape   Shape dimensions
     * @param array<int> $strides Element strides per dimension (row-major if empty)
     * @param int        $offset  Flat offset into root data
     */
    public function __construct(
        public readonly array $shape,
        array $strides = [],
        public readonly int $offset = 0,
    ) {
        $this->ndim = \count($shape);
        $this->strides = [] !== $strides ? $strides : self::computeStrides($shape);
    }

    /**
     * Build and return the C ViewMetadata struct. Caches the struct and
     * shape/strides arrays so the returned struct remains valid. Pass
     * Lib::addr($this->toCData()) when an FFI function expects a pointer.
     *
     * @internal
     */
    public function toCData(): CData
    {
        if (null !== $this->cachedStruct) {
            return $this->cachedStruct;
        }

        $ffi = Lib::get();
        $this->cachedShapeC = Lib::createShapeArray($this->shape);
        $this->cachedStridesC = Lib::createShapeArray($this->strides);
        $this->cachedStruct = $ffi->new('struct ViewMetadata', false);
        $this->cachedStruct->offset = $this->offset;
        $this->cachedStruct->shape = \FFI::addr($this->cachedShapeC[0]);
        $this->cachedStruct->strides = \FFI::addr($this->cachedStridesC[0]);
        $this->cachedStruct->ndim = $this->ndim;

        return $this->cachedStruct;
    }

    /**
     * Compute row-major strides from shape.
     *
     * @param array<int> $shape
     *
     * @return array<int>
     */
    public static function computeStrides(array $shape): array
    {
        $ndim = \count($shape);
        if (0 === $ndim) {
            return [];
        }
        $strides = array_fill(0, $ndim, 1);
        for ($i = $ndim - 2; $i >= 0; --$i) {
            $strides[$i] = $strides[$i + 1] * $shape[$i + 1];
        }

        return $strides;
    }
}
