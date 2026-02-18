<?php

declare(strict_types=1);

namespace NDArray;

use InvalidArgumentException;

/**
 * Data type enumeration for NDArray elements.
 *
 * Integer values must stay in sync with rust/src/dtype.rs
 */
enum DType: int
{
    // Signed integers
    case Int8 = 0;
    case Int16 = 1;
    case Int32 = 2;
    case Int64 = 3;

    // Unsigned integers
    case Uint8 = 4;
    case Uint16 = 5;
    case Uint32 = 6;
    case Uint64 = 7;

    // Floating-point
    case Float32 = 8;
    case Float64 = 9;

    // Boolean
    case Bool = 10;

    /**
     * Get the C FFI type string for this dtype.
     */
    public function ffiType(): string
    {
        return match ($this) {
            self::Int8 => 'int8_t',
            self::Int16 => 'int16_t',
            self::Int32 => 'int32_t',
            self::Int64 => 'int64_t',
            self::Uint8 => 'uint8_t',
            self::Uint16 => 'uint16_t',
            self::Uint32 => 'uint32_t',
            self::Uint64 => 'uint64_t',
            self::Float32 => 'float',
            self::Float64 => 'double',
            self::Bool => 'uint8_t',
        };
    }

    /**
     * Get the size of each element in bytes.
     */
    public function itemSize(): int
    {
        return match ($this) {
            self::Int8, self::Uint8, self::Bool => 1,
            self::Int16, self::Uint16 => 2,
            self::Int32, self::Uint32, self::Float32 => 4,
            self::Int64, self::Uint64, self::Float64 => 8,
        };
    }

    /**
     * Check if this dtype is a signed integer type.
     */
    public function isSigned(): bool
    {
        return match ($this) {
            self::Int8, self::Int16, self::Int32, self::Int64 => true,
            default => false,
        };
    }

    /**
     * Check if this dtype is an unsigned integer type.
     */
    public function isUnsigned(): bool
    {
        return match ($this) {
            self::Uint8, self::Uint16, self::Uint32, self::Uint64 => true,
            default => false,
        };
    }

    /**
     * Check if this dtype is an integer type (signed or unsigned).
     */
    public function isInteger(): bool
    {
        return $this->isSigned() || $this->isUnsigned();
    }

    /**
     * Check if this dtype is a floating-point type.
     */
    public function isFloat(): bool
    {
        return match ($this) {
            self::Float32, self::Float64 => true,
            default => false,
        };
    }

    /**
     * Check if this dtype is a boolean type.
     */
    public function isBool(): bool
    {
        return $this === self::Bool;
    }

    /**
     * Check if this dtype is a numeric type (integer or float).
     */
    public function isNumeric(): bool
    {
        return $this->isInteger() || $this->isFloat();
    }

    /**
     * Get a human-readable name for this dtype.
     */
    public function name(): string
    {
        return match ($this) {
            self::Int8 => 'int8',
            self::Int16 => 'int16',
            self::Int32 => 'int32',
            self::Int64 => 'int64',
            self::Uint8 => 'uint8',
            self::Uint16 => 'uint16',
            self::Uint32 => 'uint32',
            self::Uint64 => 'uint64',
            self::Float32 => 'float32',
            self::Float64 => 'float64',
            self::Bool => 'bool',
        };
    }

    /**
     * Create DType from string name (NumPy-compatible names).
     */
    public static function fromString(string $name): self
    {
        return match (strtolower($name)) {
            'int8', 'i1' => self::Int8,
            'int16', 'i2' => self::Int16,
            'int32', 'i4', 'int' => self::Int32,
            'int64', 'i8' => self::Int64,
            'uint8', 'u1' => self::Uint8,
            'uint16', 'u2' => self::Uint16,
            'uint32', 'u4' => self::Uint32,
            'uint64', 'u8' => self::Uint64,
            'float32', 'f4', 'float' => self::Float32,
            'float64', 'f8', 'double' => self::Float64,
            'bool', 'b' => self::Bool,
            default => throw new InvalidArgumentException(
                "Unknown dtype: '$name'. Valid types: int8, int16, int32, int64, " .
                "uint8, uint16, uint32, uint64, float32, float64, bool"
            ),
        };
    }

    /**
     * Infer dtype from a PHP value.
     *
     * Uses NumPy conventions: bool -> Bool, int -> Int64, float -> Float64
     */
    public static function fromValue(mixed $value): self
    {
        if (is_bool($value)) {
            return self::Bool;
        }

        if (is_int($value)) {
            return self::Int64;
        }

        if (is_float($value)) {
            return self::Float64;
        }

        throw new InvalidArgumentException(
            sprintf(
                "Cannot infer dtype from value of type '%s'. Expected bool, int, or float.",
                get_debug_type($value)
            )
        );
    }

    /**
     * Infer dtype from a PHP array, checking all elements.
     *
     * Type promotion: float > int > bool. Empty arrays default to Float64.
     */
    public static function fromArray(array $data): self
    {
        if (empty($data)) {
            return self::Float64;
        }

        $hasFloat = false;
        $hasInt = false;
        $hasBool = false;

        $stack = [$data];
        while ($stack) {
            $current = array_pop($stack);
            foreach ($current as $item) {
                if (is_array($item)) {
                    $stack[] = $item;
                } elseif (is_float($item)) {
                    $hasFloat = true;
                } elseif (is_int($item)) {
                    $hasInt = true;
                } elseif (is_bool($item)) {
                    $hasBool = true;
                } else {
                    throw new InvalidArgumentException(
                        sprintf(
                            "Cannot infer dtype: array contains unsupported type '%s'.",
                            get_debug_type($item)
                        )
                    );
                }
            }
        }

        if ($hasFloat) {
            return self::Float64;
        }
        if ($hasInt) {
            return self::Int64;
        }
        if ($hasBool) {
            return self::Bool;
        }

        return self::Float64;
    }

    /**
     * Get the minimum value representable by this dtype.
     */
    public function minValue(): int|float
    {
        return match ($this) {
            self::Int8 => -128,
            self::Int16 => -32768,
            self::Int32 => -2147483648,
            self::Int64 => PHP_INT_MIN,
            self::Uint8, self::Uint16, self::Uint32, self::Uint64 => 0,
            self::Float32 => -3.4028235e+38,
            self::Float64 => -PHP_FLOAT_MAX,
            self::Bool => 0,
        };
    }

    /**
     * Get the maximum value representable by this dtype.
     */
    public function maxValue(): int|float
    {
        return match ($this) {
            self::Int8 => 127,
            self::Int16 => 32767,
            self::Int32 => 2147483647,
            self::Int64 => PHP_INT_MAX,
            self::Uint8 => 255,
            self::Uint16 => 65535,
            self::Uint32 => 4294967295,
            self::Uint64 => PHP_INT_MAX,
            self::Float32 => 3.4028235e+38,
            self::Float64 => PHP_FLOAT_MAX,
            self::Bool => 1,
        };
    }
}
