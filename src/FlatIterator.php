<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray;

/**
 * 1-D iterator over an NDArray.
 *
 * Acts as a flattened view of the array, iterating in C-contiguous order.
 * Uses a hybrid approach:
 * - Arrays with < 100,000 elements: Batch extraction (fast)
 * - Arrays with >= 100,000 elements: Chunked extraction (memory efficient)
 *
 * Note: This is a snapshot - changes to the original array after creating
 * the iterator will not be reflected.
 */
class FlatIterator implements \Iterator, \ArrayAccess, \Countable
{
    /**
     * Threshold for switching from batch to chunked mode.
     * Arrays with size >= this value use chunked extraction.
     */
    public static int $chunkThreshold = 100000;

    /**
     * Size of each chunk when using chunked mode.
     */
    public static int $chunkSize = 10000;

    private int $totalSize;
    private int $position = 0;

    private ?array $batchElements = null;

    private array $currentChunk = [];
    private int $chunkStart = 0;

    public function __construct(public readonly NDArray $array)
    {
        $this->totalSize = $array->size();

        if ($this->totalSize < self::$chunkThreshold) {
            $this->batchElements = $array->getData(0, $this->totalSize);
        }
    }

    public function current(): float|int|bool
    {
        if ($this->batchElements !== null) {
            return $this->batchElements[$this->position];
        }

        $chunkIndex = $this->position - $this->chunkStart;

        if ($chunkIndex >= \count($this->currentChunk)) {
            $this->loadChunk();
            $chunkIndex = 0;
        }

        return $this->currentChunk[$chunkIndex];
    }

    public function key(): int
    {
        return $this->position;
    }

    public function next(): void
    {
        ++$this->position;
    }

    public function rewind(): void
    {
        $this->position = 0;

        if ($this->batchElements === null) {
            $this->currentChunk = [];
            $this->chunkStart = 0;
        }
    }

    public function valid(): bool
    {
        return $this->position < $this->totalSize;
    }

    public function offsetExists(mixed $offset): bool
    {
        if (!\is_int($offset)) {
            return false;
        }

        if ($offset < 0) {
            $offset = $this->totalSize + $offset;
        }

        return $offset >= 0 && $offset < $this->totalSize;
    }

    public function offsetGet(mixed $offset): float|int|bool
    {
        if (!\is_int($offset)) {
            throw new \OutOfBoundsException("Offset must be an integer");
        }

        if ($offset < 0) {
            $offset = $this->totalSize + $offset;
        }

        if ($offset < 0 || $offset >= $this->totalSize) {
            throw new \OutOfBoundsException("Offset {$offset} is out of bounds");
        }

        if ($this->batchElements !== null) {
            return $this->batchElements[$offset];
        }

        return $this->array->getAt($offset);
    }

    public function offsetSet(mixed $offset, mixed $value): void
    {
        if (!\is_int($offset)) {
            throw new \OutOfBoundsException("Offset must be an integer");
        }

        if ($offset < 0) {
            $offset = $this->totalSize + $offset;
        }

        if ($offset < 0 || $offset >= $this->totalSize) {
            throw new \OutOfBoundsException("Offset {$offset} is out of bounds");
        }

        $this->array->setAt($offset, $value);

        if ($this->batchElements !== null) {
            $this->batchElements[$offset] = $value;
        }
    }

    public function offsetUnset(mixed $offset): void
    {
        throw new \BadMethodCallException('Cannot unset elements in FlatIterator');
    }

    public function count(): int
    {
        return $this->totalSize;
    }

    /**
     * Convert to PHP array.
     */
    public function toArray(): array
    {
        if ($this->batchElements !== null) {
            return $this->batchElements;
        }

        return $this->array->getData(0, $this->totalSize);
    }

    /**
     * Load the next chunk of elements.
     */
    private function loadChunk(): void
    {
        $this->chunkStart = $this->position;
        $end = min($this->position + self::$chunkSize, $this->totalSize);
        $count = $end - $this->position;

        $this->currentChunk = $this->array->getData($this->chunkStart, $count);
    }
}
