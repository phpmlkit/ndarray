<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray;

/**
 * Complex number value object for NDArray operations.
 *
 * Represents a complex number with real and imaginary parts.
 * Used for creating complex arrays and receiving complex scalar results.
 */
final class Complex
{
    public function __construct(
        public readonly float $real,
        public readonly float $imag = 0.0
    ) {}

    public function __toString(): string
    {
        if ($this->imag >= 0.0) {
            return "{$this->real}+{$this->imag}j";
        }

        return "{$this->real}{$this->imag}j";
    }

    /**
     * Create a Complex from a PHP array [real, imag] or scalar value.
     */
    public static function from(mixed $value): self
    {
        if ($value instanceof self) {
            return $value;
        }

        if (\is_array($value)) {
            return new self((float) ($value[0] ?? 0.0), (float) ($value[1] ?? 0.0));
        }

        return new self((float) $value);
    }

    /**
     * Convert to a flat array [real, imag] for FFI transmission.
     *
     * @return array{float, float}
     */
    public function toArray(): array
    {
        return [$this->real, $this->imag];
    }

    /**
     * Get the magnitude (absolute value) of the complex number.
     */
    public function magnitude(): float
    {
        return hypot($this->real, $this->imag);
    }

    /**
     * Get the phase angle in radians.
     */
    public function angle(): float
    {
        return atan2($this->imag, $this->real);
    }

    /**
     * Check if this complex number equals another within a tolerance.
     */
    public function equals(self $other, float $tol = 1e-10): bool
    {
        return abs($this->real - $other->real) <= $tol
            && abs($this->imag - $other->imag) <= $tol;
    }
}
