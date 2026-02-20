<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray;

/**
 * Padding mode for NDArray::pad().
 *
 * Integer values must stay in sync with Rust PadMode in shape_ops helpers.
 */
enum PadMode: int
{
    case Constant = 0;
    case Symmetric = 1;
    case Reflect = 2;
}
