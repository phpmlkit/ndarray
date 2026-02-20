<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Traits;

use PhpMlKit\NDArray\DType;
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
    public function eq(NDArray|float|int $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_eq', $other, DType::Bool);
        }
        return $this->unaryOp('ndarray_eq_scalar', $other);
    }

    /**
     * Element-wise not-equal comparison. Returns Bool array.
     */
    public function ne(NDArray|float|int $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_ne', $other, DType::Bool);
        }
        return $this->unaryOp('ndarray_ne_scalar', $other);
    }

    /**
     * Element-wise greater-than comparison. Returns Bool array.
     */
    public function gt(NDArray|float|int $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_gt', $other, DType::Bool);
        }
        return $this->unaryOp('ndarray_gt_scalar', $other);
    }

    /**
     * Element-wise greater-or-equal comparison. Returns Bool array.
     */
    public function gte(NDArray|float|int $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_gte', $other, DType::Bool);
        }
        return $this->unaryOp('ndarray_gte_scalar', $other);
    }

    /**
     * Element-wise less-than comparison. Returns Bool array.
     */
    public function lt(NDArray|float|int $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_lt', $other, DType::Bool);
        }
        return $this->unaryOp('ndarray_lt_scalar', $other);
    }

    /**
     * Element-wise less-or-equal comparison. Returns Bool array.
     */
    public function lte(NDArray|float|int $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_lte', $other, DType::Bool);
        }
        return $this->unaryOp('ndarray_lte_scalar', $other);
    }
}
