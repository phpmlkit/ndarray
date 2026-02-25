<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Traits;

use PhpMlKit\NDArray\NDArray;

/**
 * Element-wise logical operations.
 *
 * Provides element-wise logical operators for NDArray operands.
 * All operations work on any input type (converts to bool first)
 * and always return Bool arrays.
 */
trait HasLogical
{
    /**
     * Element-wise logical AND. Returns Bool array.
     *
     * Works with any input type (converts to bool first).
     * Zero values are treated as false, non-zero as true.
     */
    public function and(NDArray $other): NDArray
    {
        return $this->binaryOp('ndarray_logical_and', $other);
    }

    /**
     * Element-wise logical OR. Returns Bool array.
     *
     * Works with any input type (converts to bool first).
     * Zero values are treated as false, non-zero as true.
     */
    public function or(NDArray $other): NDArray
    {
        return $this->binaryOp('ndarray_logical_or', $other);
    }

    /**
     * Element-wise logical NOT. Returns Bool array.
     *
     * Works with any input type (converts to bool first).
     * Zero values are treated as false, non-zero as true.
     */
    public function not(): NDArray
    {
        return $this->unaryOp('ndarray_logical_not');
    }

    /**
     * Element-wise logical XOR. Returns Bool array.
     *
     * Works with any input type (converts to bool first).
     * Zero values are treated as false, non-zero as true.
     */
    public function xor(NDArray $other): NDArray
    {
        return $this->binaryOp('ndarray_logical_xor', $other);
    }
}
