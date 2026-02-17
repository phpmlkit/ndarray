<?php

declare(strict_types=1);

namespace NDArray;

/**
 * Sorting algorithm selection for sort/argsort operations.
 *
 * Integer values must stay in sync with rust sort kind parsing.
 */
enum SortKind: int
{
    case QuickSort = 0;
    case MergeSort = 1;
    case HeapSort = 2;
    case Stable = 3;
}
