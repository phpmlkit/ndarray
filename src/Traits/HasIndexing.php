<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Traits;

use FFI\CData;
use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\Exceptions\IndexException;
use PhpMlKit\NDArray\Exceptions\NDArrayException;
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

        if ($count > $this->ndim()) {
            throw new IndexException(
                "Too many indices: got {$count} for array with {$this->ndim()} dimensions"
            );
        }

        $normalizedIndices = [];
        foreach ($indices as $dim => $index) {
            $dimSize = $this->shape()[$dim];
            $normalizedIndex = $this->normalizeIndex($index, $dimSize, $dim);
            $normalizedIndices[] = $normalizedIndex;
        }

        if ($count === $this->ndim()) {
            // Full indexing - return scalar
            $flatIndex = $this->calculateFlatIndex($normalizedIndices);

            return $this->getElement($flatIndex);
        }

        // Partial indexing - return a view
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

        if ($count !== $this->ndim()) {
            throw new IndexException(
                "set() requires exactly {$this->ndim()} indices, got {$count}"
            );
        }

        $normalizedIndices = [];
        foreach ($indices as $dim => $index) {
            $dimSize = $this->shape()[$dim];
            $normalizedIndices[] = $this->normalizeIndex($index, $dimSize, $dim);
        }

        $flatIndex = $this->calculateFlatIndex($normalizedIndices);

        $this->setElement($flatIndex, $value);
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
        $this->setElement($storageFlatIndex, $value);
    }

    /**
     * Gather values by indices.
     *
     * If axis is null, gathers from logical flattened view (C-order).
     * If axis is provided, delegates to takeAlongAxis semantics.
     *
     * @param array<array<int>|int>|self $indices
     */
    public function take(array|self $indices, ?int $axis = null): self
    {
        $indices = $indices instanceof self ? $indices : NDArray::array($indices, DType::Int64);

        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');
        [$outDtypeBuf, $outNdimBuf, $outShapeBuf] = Lib::createOutputMetadataBuffers();

        $meta = $this->viewMetadata()->toCData();
        $indicesMeta = $indices->viewMetadata()->toCData();

        if (null !== $axis) {
            $status = $ffi->ndarray_take_axis(
                $this->handle,
                Lib::addr($meta),
                $indices->handle(),
                Lib::addr($indicesMeta),
                $axis,
                Lib::addr($outHandle),
                Lib::addr($outDtypeBuf),
                Lib::addr($outNdimBuf),
                $outShapeBuf,
                Lib::MAX_NDIM
            );
        } else {
            $status = $ffi->ndarray_take(
                $this->handle,
                Lib::addr($meta),
                $indices->handle(),
                Lib::addr($indicesMeta),
                Lib::addr($outHandle),
                Lib::addr($outDtypeBuf),
                Lib::addr($outNdimBuf),
                $outShapeBuf,
                Lib::MAX_NDIM
            );
        }

        Lib::checkStatus($status);

        $dtype = DType::tryFrom((int) $outDtypeBuf->cdata);
        if (null === $dtype) {
            throw new NDArrayException('Invalid dtype returned from Rust');
        }

        $ndim = (int) $outNdimBuf->cdata;
        $outShape = Lib::extractShapeFromPointer($outShapeBuf, $ndim);

        return new self($outHandle, $outShape, $dtype);
    }

    /**
     * Gather values along an axis using per-position indices.
     *
     * @param array<array<int>|int>|self $indices
     */
    public function takeAlongAxis(array|self $indices, int $axis): self
    {
        $indices = $indices instanceof self ? $indices : NDArray::array($indices, DType::Int64);

        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');
        [$outDtypeBuf, $outNdimBuf, $outShapeBuf] = Lib::createOutputMetadataBuffers();

        $meta = $this->viewMetadata()->toCData();
        $indicesMeta = $indices->viewMetadata()->toCData();
        $status = $ffi->ndarray_take_along_axis(
            $this->handle,
            Lib::addr($meta),
            $indices->handle(),
            Lib::addr($indicesMeta),
            $axis,
            Lib::addr($outHandle),
            Lib::addr($outDtypeBuf),
            Lib::addr($outNdimBuf),
            $outShapeBuf,
            Lib::MAX_NDIM
        );

        Lib::checkStatus($status);

        $dtype = DType::tryFrom((int) $outDtypeBuf->cdata);
        if (null === $dtype) {
            throw new NDArrayException('Invalid dtype returned from Rust');
        }

        $ndim = (int) $outNdimBuf->cdata;
        $outShape = Lib::extractShapeFromPointer($outShapeBuf, $ndim);

        return new self($outHandle, $outShape, $dtype);
    }

    /**
     * Scatter values by flattened indices and return a mutated copy.
     *
     * @param array<array<int>|int>|self $indices
     * @param string                     $mode    Currently supports only 'raise'
     */
    public function put(array|self $indices, bool|float|int|self $values, string $mode = 'raise'): self
    {
        if ('raise' !== $mode) {
            throw new \InvalidArgumentException("put mode '{$mode}' is not supported yet. Use 'raise'.");
        }

        $indices = $indices instanceof self ? $indices : NDArray::array($indices, DType::Int64);

        [$valuesBuffer, $valuesLen, $scalarValue, $hasScalar] = $this->prepareValuesBuffer($values);

        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

        $meta = $this->viewMetadata()->toCData();
        $indicesMeta = $indices->viewMetadata()->toCData();
        $status = $ffi->ndarray_put(
            $this->handle,
            Lib::addr($meta),
            $indices->handle(),
            Lib::addr($indicesMeta),
            $valuesBuffer,
            $valuesLen,
            $scalarValue,
            $hasScalar,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new self($outHandle, $this->shape(), $this->dtype);
    }

    /**
     * Scatter values along an axis and return a mutated copy.
     *
     * @param array<array<int>|int>|self $indices
     */
    public function putAlongAxis(array|self $indices, bool|float|int|self $values, int $axis): self
    {
        $indices = $indices instanceof self ? $indices : NDArray::array($indices, DType::Int64);

        [$valuesBuffer, $valuesLen, $scalarValue, $hasScalar] = $this->prepareValuesBuffer($values);
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

        $meta = $this->viewMetadata()->toCData();
        $indicesMeta = $indices->viewMetadata()->toCData();
        $status = $ffi->ndarray_put_along_axis(
            $this->handle,
            Lib::addr($meta),
            $indices->handle(),
            Lib::addr($indicesMeta),
            $axis,
            $valuesBuffer,
            $valuesLen,
            $scalarValue,
            $hasScalar,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new self($outHandle, $this->shape(), $this->dtype);
    }

    /**
     * Add updates by flattened indices and return a mutated copy.
     *
     * @param array<array<int>|int>|self $indices
     */
    public function scatterAdd(array|self $indices, bool|float|int|self $updates): self
    {
        $indices = $indices instanceof self ? $indices : NDArray::array($indices, DType::Int64);

        [$updatesBuffer, $updatesLen, $scalarUpdate, $hasScalar] = $this->prepareValuesBuffer($updates);

        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');

        $meta = $this->viewMetadata()->toCData();
        $indicesMeta = $indices->viewMetadata()->toCData();
        $status = $ffi->ndarray_scatter_add_flat(
            $this->handle,
            Lib::addr($meta),
            $indices->handle(),
            Lib::addr($indicesMeta),
            $updatesBuffer,
            $updatesLen,
            $scalarUpdate,
            $hasScalar,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new self($outHandle, $this->shape(), $this->dtype);
    }

    /**
     * Calculate flat index from dimension indices using strides.
     *
     * flat_index = offset + sum(indices[i] * strides[i])
     *
     * @param array<int> $indices
     */
    private function calculateFlatIndex(array $indices): int
    {
        $flatIndex = $this->getOffset();
        foreach ($indices as $dim => $index) {
            $flatIndex += $index * $this->strides()[$dim];
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
        if (1 === $this->ndim()) {
            return $this->getOffset() + ($logicalFlatIndex * $this->strides()[0]);
        }

        $storageIndex = $this->getOffset();
        $remaining = $logicalFlatIndex;

        for ($dim = $this->ndim() - 1; $dim >= 0; --$dim) {
            $dimSize = $this->shape()[$dim];
            $idxInDim = $remaining % $dimSize;
            $remaining = intdiv($remaining, $dimSize);
            $storageIndex += $idxInDim * $this->strides()[$dim];
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
                "Index {$index} is out of bounds for dimension {$dim} with size {$size}"
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

        $newOffset = $this->getOffset();
        foreach ($indices as $dim => $index) {
            $newOffset += $index * $this->strides()[$dim];
        }

        $newShape = \array_slice($this->shape(), $count);
        $newStrides = \array_slice($this->strides(), $count);

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
     * Build a typed values buffer for put/scatter operations.
     *
     * @return array{0: CData, 1: int, 2: float, 3: bool}
     */
    private function prepareValuesBuffer(bool|float|int|self $values): array
    {
        if (\is_int($values) || \is_float($values) || \is_bool($values)) {
            $dummy = $this->dtype->createCArray(1, [0]);

            return [$dummy, 0, (float) $values, true];
        }

        $valuesNd = $values->dtype === $this->dtype ? $values : $values->astype($this->dtype);
        $flat = $valuesNd->toFlatArray();
        $flat = \is_array($flat) ? $flat : [$flat];
        $flat = $this->dtype->prepareArrayValues($flat);
        $buffer = $this->dtype->createCArray(\count($flat), $flat);

        return [$buffer, \count($flat), 0.0, false];
    }

    /**
     * Read an element at a flat index.
     */
    private function getElement(int $flatIndex): bool|float|int
    {
        $ffi = Lib::get();

        $outValue = $this->dtype->createCValue();
        $status = $ffi->ndarray_get_element($this->handle, $flatIndex, Lib::addr($outValue));
        Lib::checkStatus($status);

        return $this->dtype->castFromCValue($outValue->cdata);
    }

    /**
     * Write an element at a flat index.
     */
    private function setElement(int $flatIndex, bool|float|int $value): void
    {
        $ffi = Lib::get();

        $cValue = $this->dtype->createCValue($value);

        $status = $ffi->ndarray_set_element($this->handle, $flatIndex, Lib::addr($cValue));
        Lib::checkStatus($status);
    }
}
