<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Traits;

use FFI;
use FFI\CData;
use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\Exceptions\IndexException;
use PhpMlKit\NDArray\FFI\Bindings;
use PhpMlKit\NDArray\FFI\Lib;
use PhpMlKit\NDArray\NDArray;

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
     * Supports negative indices: -1 refers to the last element, -2 to second-to-last, etc.
     *
     * @param int ...$indices One or more dimension indices
     *
     * @return bool|float|int|self Scalar for full indexing, view for partial
     */
    public function get(int ...$indices): bool|float|int|self
    {
        $count = \count($indices);

        if (0 === $count) {
            throw new IndexException('At least one index is required');
        }

        if ($count > $this->ndim) {
            throw new IndexException(
                "Too many indices: got {$count} for array with {$this->ndim} dimensions"
            );
        }

        // Normalize and validate each index
        $normalizedIndices = [];
        foreach ($indices as $dim => $index) {
            $dimSize = $this->shape[$dim];
            $normalizedIndex = $this->normalizeIndex($index, $dimSize, $dim);
            $normalizedIndices[] = $normalizedIndex;
        }

        if ($count === $this->ndim) {
            // Full indexing — return scalar via FFI
            $flatIndex = $this->calculateFlatIndex($normalizedIndices);

            return $this->getScalar($flatIndex);
        }

        // Partial indexing — return a view (pure PHP, zero FFI)
        return $this->createView($normalizedIndices);
    }

    /**
     * Set a scalar value at the given indices.
     *
     * Requires full indexing (count === ndim).
     *
     * Supports negative indices: -1 refers to the last element, -2 to second-to-last, etc.
     *
     * @param array<int>     $indices Indices for each dimension
     * @param bool|float|int $value   Value to set
     */
    public function set(array $indices, bool|float|int $value): void
    {
        $count = \count($indices);

        if ($count !== $this->ndim) {
            throw new IndexException(
                "set() requires exactly {$this->ndim} indices, got {$count}"
            );
        }

        // Normalize and validate each index
        $normalizedIndices = [];
        foreach ($indices as $dim => $index) {
            $dimSize = $this->shape[$dim];
            $normalizedIndices[] = $this->normalizeIndex($index, $dimSize, $dim);
        }

        $flatIndex = $this->calculateFlatIndex($normalizedIndices);

        $this->setScalar($flatIndex, $value);
    }

    /**
     * Set a scalar value using a logical flat index (C-order) for this array/view.
     *
     * Supports negative indices: -1 refers to the last logical element.
     *
     * @param int            $flatIndex Logical flat index into this array/view
     * @param bool|float|int $value     Value to set
     */
    public function setAt(int $flatIndex, bool|float|int $value): void
    {
        $normalized = $this->normalizeFlatIndex($flatIndex);
        $storageFlatIndex = $this->logicalFlatToStorageIndex($normalized);
        $this->setScalar($storageFlatIndex, $value);
    }

    /**
     * Gather values by indices.
     *
     * If axis is null, gathers from logical flattened view (C-order).
     * If axis is provided, delegates to takeAlongAxis semantics.
     *
     * @param array<array|int>|self $indices
     */
    public function take(array|self $indices, ?int $axis = null): self
    {
        if (null !== $axis) {
            $idxArray = $indices instanceof self ? $indices : NDArray::array($indices, DType::Int64);

            return $this->takeAlongAxis($idxArray, $axis);
        }

        [$flatIndices, $indicesShape] = $this->prepareIndicesInput($indices);
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

        $status = $ffi->ndarray_take_flat(
            $this->handle,
            $this->offset,
            Lib::createShapeArray($this->shape),
            Lib::createCArray('size_t', $this->strides),
            $this->ndim,
            Lib::createCArray('int64_t', $flatIndices),
            \count($flatIndices),
            Lib::createShapeArray($indicesShape),
            \count($indicesShape),
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new self($outHandle, $indicesShape, $this->dtype);
    }

    /**
     * Gather values along an axis using per-position indices.
     *
     * @param self $indices Int64 indices array
     */
    public function takeAlongAxis(self $indices, int $axis): self
    {
        if (DType::Int64 !== $indices->dtype) {
            throw new \InvalidArgumentException('takeAlongAxis indices must have Int64 dtype');
        }

        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

        $status = $ffi->ndarray_take_along_axis(
            $this->handle,
            $this->offset,
            Lib::createShapeArray($this->shape),
            Lib::createCArray('size_t', $this->strides),
            $this->ndim,
            $indices->handle,
            $indices->offset,
            Lib::createShapeArray($indices->shape),
            Lib::createCArray('size_t', $indices->strides),
            $indices->ndim,
            $axis,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new self($outHandle, $indices->shape, $this->dtype);
    }

    /**
     * Scatter values by flattened indices and return a mutated copy.
     *
     * @param array<array|int>|self $indices
     * @param string                $mode    Currently supports only 'raise'
     */
    public function put(array|self $indices, bool|float|int|self $values, string $mode = 'raise'): self
    {
        if ('raise' !== $mode) {
            throw new \InvalidArgumentException("put mode '{$mode}' is not supported yet. Use 'raise'.");
        }

        [$flatIndices] = $this->prepareIndicesInput($indices);
        [$valuesBuffer, $valuesLen, $scalarValue, $hasScalar] = $this->prepareValuesBuffer($values);

        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

        $status = $ffi->ndarray_put_flat(
            $this->handle,
            $this->offset,
            Lib::createShapeArray($this->shape),
            Lib::createCArray('size_t', $this->strides),
            $this->ndim,
            Lib::createCArray('int64_t', $flatIndices),
            \count($flatIndices),
            $valuesBuffer,
            $valuesLen,
            $scalarValue,
            $hasScalar,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new self($outHandle, $this->shape, $this->dtype);
    }

    /**
     * Scatter values along an axis and return a mutated copy.
     *
     * @param self $indices Int64 indices array
     */
    public function putAlongAxis(self $indices, bool|float|int|self $values, int $axis): self
    {
        if (DType::Int64 !== $indices->dtype) {
            throw new \InvalidArgumentException('putAlongAxis indices must have Int64 dtype');
        }

        [$valuesBuffer, $valuesLen, $scalarValue, $hasScalar] = $this->prepareValuesBuffer($values);
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

        $status = $ffi->ndarray_put_along_axis(
            $this->handle,
            $this->offset,
            Lib::createShapeArray($this->shape),
            Lib::createCArray('size_t', $this->strides),
            $this->ndim,
            $indices->handle,
            $indices->offset,
            Lib::createShapeArray($indices->shape),
            Lib::createCArray('size_t', $indices->strides),
            $indices->ndim,
            $axis,
            $valuesBuffer,
            $valuesLen,
            $scalarValue,
            $hasScalar,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new self($outHandle, $this->shape, $this->dtype);
    }

    /**
     * Add updates by flattened indices and return a mutated copy.
     *
     * @param array<array|int>|self $indices
     */
    public function scatterAdd(array|self $indices, bool|float|int|self $updates): self
    {
        [$flatIndices] = $this->prepareIndicesInput($indices);
        [$updatesBuffer, $updatesLen, $scalarUpdate, $hasScalar] = $this->prepareValuesBuffer($updates);

        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

        $status = $ffi->ndarray_scatter_add_flat(
            $this->handle,
            $this->offset,
            Lib::createShapeArray($this->shape),
            Lib::createCArray('size_t', $this->strides),
            $this->ndim,
            Lib::createCArray('int64_t', $flatIndices),
            \count($flatIndices),
            $updatesBuffer,
            $updatesLen,
            $scalarUpdate,
            $hasScalar,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new self($outHandle, $this->shape, $this->dtype);
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
     * Normalize a logical flat index (supports negative indexing).
     */
    private function normalizeFlatIndex(int $flatIndex): int
    {
        if ($flatIndex < 0) {
            $flatIndex += $this->size;
        }

        if ($flatIndex < 0 || $flatIndex >= $this->size) {
            throw new IndexException(
                "Flat index {$flatIndex} is out of bounds for array/view of size {$this->size}"
            );
        }

        return $flatIndex;
    }

    /**
     * Convert a logical C-order flat index to the underlying storage flat index.
     */
    private function logicalFlatToStorageIndex(int $logicalFlatIndex): int
    {
        if (1 === $this->ndim) {
            return $this->offset + ($logicalFlatIndex * $this->strides[0]);
        }

        $storageIndex = $this->offset;
        $remaining = $logicalFlatIndex;

        for ($dim = $this->ndim - 1; $dim >= 0; --$dim) {
            $dimSize = $this->shape[$dim];
            $idxInDim = $remaining % $dimSize;
            $remaining = intdiv($remaining, $dimSize);
            $storageIndex += $idxInDim * $this->strides[$dim];
        }

        return $storageIndex;
    }

    /**
     * Normalize an index to a positive value, handling negative indices.
     *
     * Negative indices count from the end: -1 is the last element,
     * -2 is the second-to-last, etc.
     *
     * @param int $index The index (may be negative)
     * @param int $size  The size of the dimension
     * @param int $dim   The dimension number (for error messages)
     *
     * @return int The normalized positive index
     *
     * @throws IndexException If the index is out of bounds
     */
    private function normalizeIndex(int $index, int $size, int $dim): int
    {
        if ($index < 0) {
            $index = $size + $index;
        }

        if ($index < 0 || $index >= $size) {
            throw new IndexException(
                'Index '.($index - $size)." is out of bounds for dimension {$dim} with size {$size}"
            );
        }

        return $index;
    }

    /**
     * Create a view from partial indices (pure PHP, zero FFI).
     *
     * @param array<int> $indices Partial dimension indices
     */
    private function createView(array $indices): self
    {
        $count = \count($indices);

        // Calculate new offset
        $newOffset = $this->offset;
        foreach ($indices as $dim => $index) {
            $newOffset += $index * $this->strides[$dim];
        }

        // Remaining dimensions become the view's shape/strides
        $newShape = \array_slice($this->shape, $count);
        $newStrides = \array_slice($this->strides, $count);

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
     * Read a scalar value at a flat index via FFI.
     */
    private function getScalar(int $flatIndex): bool|float|int
    {
        $ffi = Lib::get();

        return match ($this->dtype) {
            DType::Int8 => $this->getTypedScalar($ffi, 'int8_t', $flatIndex),
            DType::Int16 => $this->getTypedScalar($ffi, 'int16_t', $flatIndex),
            DType::Int32 => $this->getTypedScalar($ffi, 'int32_t', $flatIndex),
            DType::Int64 => $this->getTypedScalar($ffi, 'int64_t', $flatIndex),
            DType::Uint8 => $this->getTypedScalar($ffi, 'uint8_t', $flatIndex),
            DType::Uint16 => $this->getTypedScalar($ffi, 'uint16_t', $flatIndex),
            DType::Uint32 => $this->getTypedScalar($ffi, 'uint32_t', $flatIndex),
            DType::Uint64 => $this->getTypedScalar($ffi, 'uint64_t', $flatIndex),
            DType::Float32 => $this->getTypedScalar($ffi, 'float', $flatIndex),
            DType::Float64 => $this->getTypedScalar($ffi, 'double', $flatIndex),
            DType::Bool => (bool) $this->getTypedScalar($ffi, 'uint8_t', $flatIndex),
        };
    }

    /**
     * Get a scalar value via FFI using the unified ndarray_get_element function.
     *
     * @param Bindings&\FFI $ffi
     * @param string        $cType C type for the output value
     */
    private function getTypedScalar(\FFI $ffi, string $cType, int $flatIndex): float|int
    {
        $outValue = Lib::createBox($cType);
        $status = $ffi->ndarray_get_element($this->handle, $flatIndex, Lib::addr($outValue));
        Lib::checkStatus($status);

        return $outValue->cdata;
    }

    /**
     * @param array<array|int>|self $indices
     *
     * @return array{0: array<int>, 1: array<int>}
     */
    private function prepareIndicesInput(array|self $indices): array
    {
        if ($indices instanceof self) {
            if (!$indices->dtype->isInteger()) {
                throw new \InvalidArgumentException('indices NDArray must have an integer dtype');
            }
            $flat = $indices->toFlatArray();
            $flat = \is_array($flat) ? $flat : [$flat];

            return [array_map(static fn ($v) => (int) $v, $flat), $indices->shape];
        }

        $shape = $this->inferNestedShape($indices);
        $flat = $this->flattenNestedArray($indices);

        return [array_map(static fn ($v) => (int) $v, $flat), $shape];
    }

    /**
     * Build a typed values buffer for put/scatter operations.
     *
     * @return array{0: CData, 1: int, 2: float, 3: bool}
     */
    private function prepareValuesBuffer(bool|float|int|self $values): array
    {
        if (\is_int($values) || \is_float($values) || \is_bool($values)) {
            $dummy = Lib::createCArray($this->dtype->ffiType(), [0]);

            return [$dummy, 0, (float) $values, true];
        }

        $valuesNd = $values->dtype === $this->dtype ? $values : $values->astype($this->dtype);
        $flat = $valuesNd->toFlatArray();
        $flat = \is_array($flat) ? $flat : [$flat];
        if (DType::Bool === $this->dtype) {
            $flat = array_map(static fn ($v) => $v ? 1 : 0, $flat);
        }
        $buffer = Lib::createCArray($this->dtype->ffiType(), $flat);

        return [$buffer, \count($flat), 0.0, false];
    }

    /**
     * Infer shape from nested PHP arrays.
     *
     * @return array<int>
     */
    private function inferNestedShape(array $data): array
    {
        $shape = [];
        $current = $data;
        while (\is_array($current)) {
            $shape[] = \count($current);
            if (empty($current)) {
                break;
            }
            $current = $current[0];
        }

        return [] === $shape ? [\count($data)] : $shape;
    }

    /**
     * Flatten nested PHP arrays.
     *
     * @return array<bool|float|int>
     */
    private function flattenNestedArray(array $data): array
    {
        $out = [];
        array_walk_recursive($data, static function ($v) use (&$out): void {
            $out[] = $v;
        });

        return $out;
    }

    /**
     * Write a scalar value at a flat index via type-specific FFI call.
     */
    private function setScalar(int $flatIndex, bool|float|int $value): void
    {
        $ffi = Lib::get();

        // Create C value of appropriate type
        $cValue = match ($this->dtype) {
            DType::Int8 => $ffi->new('int8_t'),
            DType::Int16 => $ffi->new('int16_t'),
            DType::Int32 => $ffi->new('int32_t'),
            DType::Int64 => $ffi->new('int64_t'),
            DType::Uint8 => $ffi->new('uint8_t'),
            DType::Uint16 => $ffi->new('uint16_t'),
            DType::Uint32 => $ffi->new('uint32_t'),
            DType::Uint64 => $ffi->new('uint64_t'),
            DType::Float32 => $ffi->new('float'),
            DType::Float64 => $ffi->new('double'),
            DType::Bool => $ffi->new('uint8_t'),
        };

        // Set the value
        if (DType::Bool === $this->dtype) {
            $cValue->cdata = $value ? 1 : 0;
        } else {
            $cValue->cdata = $value;
        }

        $status = $ffi->ndarray_set_element($this->handle, $flatIndex, Lib::addr($cValue));
        Lib::checkStatus($status);
    }
}
