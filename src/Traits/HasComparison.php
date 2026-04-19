<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Traits;

use PhpMlKit\NDArray\Complex;
use PhpMlKit\NDArray\NDArray;

/**
 * Element-wise comparison operations.
 *
 * Provides element-wise comparison operators for NDArray and scalar operands.
 */
trait HasComparison
{
    /**
     * Element-wise equal comparison. Returns Bool array.
     */
    public function eq(Complex|float|int|NDArray $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_eq', $other);
        }

        return $this->unaryOp('ndarray_eq_scalar', ...$this->scalarToBuffer($other));
    }

    /**
     * Element-wise not-equal comparison. Returns Bool array.
     */
    public function ne(Complex|float|int|NDArray $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_ne', $other);
        }

        return $this->unaryOp('ndarray_ne_scalar', ...$this->scalarToBuffer($other));
    }

    /**
     * Element-wise greater-than comparison. Returns Bool array.
     */
    public function gt(Complex|float|int|NDArray $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_gt', $other);
        }

        return $this->unaryOp('ndarray_gt_scalar', ...$this->scalarToBuffer($other));
    }

    /**
     * Element-wise greater-or-equal comparison. Returns Bool array.
     */
    public function gte(Complex|float|int|NDArray $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_gte', $other);
        }

        return $this->unaryOp('ndarray_gte_scalar', ...$this->scalarToBuffer($other));
    }

    /**
     * Element-wise less-than comparison. Returns Bool array.
     */
    public function lt(Complex|float|int|NDArray $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_lt', $other);
        }

        return $this->unaryOp('ndarray_lt_scalar', ...$this->scalarToBuffer($other));
    }

    /**
     * Element-wise less-or-equal comparison. Returns Bool array.
     */
    public function lte(Complex|float|int|NDArray $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_lte', $other);
        }

        return $this->unaryOp('ndarray_lte_scalar', ...$this->scalarToBuffer($other));
    }
}
