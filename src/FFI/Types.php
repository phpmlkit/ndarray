<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\FFI;

use FFI\CData;

/**
 * IDE Helper for FFI CData types.
 */
class Types
{
    // This class is a namespace holder
}

/**
 * Generic CData wrapper for scalar pointers.
 *
 * @property mixed $cdata The underlying C value or pointer.
 */
class Box extends CData {}

/**
 * Wrapper for NdArrayHandle pointer.
 */
class Handle extends CData {}
