<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray;

/**
 * FFT / DCT normalization (SciPy `norm` / NumPy `norm`).
 *
 * Integer codes from {@see self::toFfi()} match the Rust FFI (`0` … `3`).
 */
enum Normalization: string
{
    /** Default; same as NumPy / SciPy `norm='backward'`. */
    case Backward = 'backward';

    /** Alias of {@see self::Backward}. */
    case Default = 'default';

    /** Orthonormal (unitary) transform. */
    case Ortho = 'ortho';

    /** Alias of {@see self::Ortho}. */
    case Orthogonal = 'orthogonal';

    /** Forward normalization (inverse of `backward` scaling). */
    case Forward = 'forward';

    /** Unnormalized `rustfft` / `rustdct` output (no SciPy scaling). */
    case None = 'none';

    /** Alias of {@see self::None}. */
    case Unscaled = 'unscaled';

    /**
     * FFI norm code passed to Rust: `0` backward, `1` ortho, `2` forward, `3` none.
     */
    public function toFfi(): int
    {
        return match ($this) {
            self::Backward, self::Default => 0,
            self::Ortho, self::Orthogonal => 1,
            self::Forward => 2,
            self::None, self::Unscaled => 3,
        };
    }
}
