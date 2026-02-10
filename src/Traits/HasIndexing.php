<?php

declare(strict_types=1);

namespace NDArray\Traits;

use FFI;
use NDArray\DType;
use NDArray\Exceptions\IndexException;
use NDArray\FFI\Lib;

/**
 * Element indexing: get(), set(), and view creation.
 *
 * Full indices return a scalar via FFI. Partial indices return a
 * zero-cost PHP view sharing the same Rust handle.
 */
trait HasIndexing
{
    /**
     * Access elements by index.
     *
     * Full indices (count === ndim) return a scalar via FFI read.
     * Partial indices (count < ndim) return a view (pure PHP, zero FFI).
     *
     * @param int ...$indices One or more dimension indices
     * @return self|int|float|bool Scalar for full indexing, view for partial
     */
    public function get(int ...$indices): self|int|float|bool
    {
        $count = count($indices);

        if ($count === 0) {
            throw new IndexException('At least one index is required');
        }

        if ($count > $this->ndim) {
            throw new IndexException(
                "Too many indices: got $count for array with {$this->ndim} dimensions"
            );
        }

        // Validate each index
        foreach ($indices as $dim => $index) {
            $dimSize = $this->shape[$dim];
            if ($index < 0 || $index >= $dimSize) {
                throw new IndexException(
                    "Index $index is out of bounds for dimension $dim with size $dimSize"
                );
            }
        }

        if ($count === $this->ndim) {
            // Full indexing — return scalar via FFI
            $flatIndex = $this->calculateFlatIndex($indices);

            return $this->getScalar($flatIndex);
        }

        // Partial indexing — return a view (pure PHP, zero FFI)
        return $this->createView($indices);
    }

    /**
     * Set a scalar value at the given indices.
     *
     * Requires full indexing (count === ndim).
     *
     * @param array<int> $indices Indices for each dimension
     * @param int|float|bool $value Value to set
     */
    public function set(array $indices, int|float|bool $value): void
    {
        $count = count($indices);

        if ($count !== $this->ndim) {
            throw new IndexException(
                "set() requires exactly {$this->ndim} indices, got $count"
            );
        }

        // Validate each index
        foreach ($indices as $dim => $index) {
            $dimSize = $this->shape[$dim];
            if ($index < 0 || $index >= $dimSize) {
                throw new IndexException(
                    "Index $index is out of bounds for dimension $dim with size $dimSize"
                );
            }
        }

        $flatIndex = $this->calculateFlatIndex($indices);

        $this->setScalar($flatIndex, $value);
    }

    // =========================================================================
    // Private Helpers
    // =========================================================================

    /**
     * Calculate flat index from dimension indices using strides.
     *
     * flat_index = offset + sum(indices[i] * strides[i])
     *
     * @param array<int> $indices
     * @return int
     */
    private function calculateFlatIndex(array $indices): int
    {
        $flatIndex = $this->offset;
        foreach ($indices as $dim => $index) {
            $flatIndex += $index * $this->strides[$dim];
        }

        return $flatIndex;
    }

    /**
     * Create a view from partial indices (pure PHP, zero FFI).
     *
     * @param array<int> $indices Partial dimension indices
     * @return self
     */
    private function createView(array $indices): self
    {
        $count = count($indices);

        // Calculate new offset
        $newOffset = $this->offset;
        foreach ($indices as $dim => $index) {
            $newOffset += $index * $this->strides[$dim];
        }

        // Remaining dimensions become the view's shape/strides
        $newShape = array_slice($this->shape, $count);
        $newStrides = array_slice($this->strides, $count);

        // Base is the root array (follow the chain)
        $root = $this->base ?? $this;

        return new self(
            handle: $this->handle,
            shape: $newShape,
            dtype: $this->dtype,
            strides: $newStrides,
            offset: $newOffset,
            base: $root,
        );
    }

    /**
     * Read a scalar value at a flat index via type-specific FFI call.
     *
     * @param int $flatIndex
     * @return int|float|bool
     */
    private function getScalar(int $flatIndex): int|float|bool
    {
        $ffi = Lib::get();

        return match ($this->dtype) {
            DType::Int8 => $this->getTypedScalar($ffi, 'ndarray_get_element_int8', 'int8_t', $flatIndex),
            DType::Int16 => $this->getTypedScalar($ffi, 'ndarray_get_element_int16', 'int16_t', $flatIndex),
            DType::Int32 => $this->getTypedScalar($ffi, 'ndarray_get_element_int32', 'int32_t', $flatIndex),
            DType::Int64 => $this->getTypedScalar($ffi, 'ndarray_get_element_int64', 'int64_t', $flatIndex),
            DType::Uint8 => $this->getTypedScalar($ffi, 'ndarray_get_element_uint8', 'uint8_t', $flatIndex),
            DType::Uint16 => $this->getTypedScalar($ffi, 'ndarray_get_element_uint16', 'uint16_t', $flatIndex),
            DType::Uint32 => $this->getTypedScalar($ffi, 'ndarray_get_element_uint32', 'uint32_t', $flatIndex),
            DType::Uint64 => $this->getTypedScalar($ffi, 'ndarray_get_element_uint64', 'uint64_t', $flatIndex),
            DType::Float32 => $this->getTypedScalar($ffi, 'ndarray_get_element_float32', 'float', $flatIndex),
            DType::Float64 => $this->getTypedScalar($ffi, 'ndarray_get_element_float64', 'double', $flatIndex),
            DType::Bool => (bool) $this->getTypedScalar($ffi, 'ndarray_get_element_bool', 'uint8_t', $flatIndex),
        };
    }

    /**
     * Call a type-specific get_element FFI function.
     *
     * @param FFI $ffi
     * @param string $funcName FFI function name
     * @param string $cType C type for the output value
     * @param int $flatIndex
     * @return int|float
     */
    private function getTypedScalar(FFI $ffi, string $funcName, string $cType, int $flatIndex): int|float
    {
        $outValue = $ffi->new($cType);
        $status = $ffi->$funcName($this->handle, $flatIndex, Lib::addr($outValue));
        Lib::checkStatus($status);

        return $outValue->cdata;
    }

    /**
     * Write a scalar value at a flat index via type-specific FFI call.
     *
     * @param int $flatIndex
     * @param int|float|bool $value
     */
    private function setScalar(int $flatIndex, int|float|bool $value): void
    {
        $ffi = Lib::get();

        $funcName = match ($this->dtype) {
            DType::Int8 => 'ndarray_set_element_int8',
            DType::Int16 => 'ndarray_set_element_int16',
            DType::Int32 => 'ndarray_set_element_int32',
            DType::Int64 => 'ndarray_set_element_int64',
            DType::Uint8 => 'ndarray_set_element_uint8',
            DType::Uint16 => 'ndarray_set_element_uint16',
            DType::Uint32 => 'ndarray_set_element_uint32',
            DType::Uint64 => 'ndarray_set_element_uint64',
            DType::Float32 => 'ndarray_set_element_float32',
            DType::Float64 => 'ndarray_set_element_float64',
            DType::Bool => 'ndarray_set_element_bool',
        };

        if ($this->dtype === DType::Bool) {
            $value = $value ? 1 : 0;
        }

        $status = $ffi->$funcName($this->handle, $flatIndex, $value);
        Lib::checkStatus($status);
    }
}
