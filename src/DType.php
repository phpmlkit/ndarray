<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray;

use FFI\CData;
use PhpMlKit\NDArray\FFI\Lib;

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
    case UInt8 = 4;
    case UInt16 = 5;
    case UInt32 = 6;
    case UInt64 = 7;

    // Floating-point
    case Float32 = 8;
    case Float64 = 9;

    // Boolean
    case Bool = 10;

    // Complex numbers
    case Complex64 = 11;
    case Complex128 = 12;

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
            self::UInt8 => 'uint8_t',
            self::UInt16 => 'uint16_t',
            self::UInt32 => 'uint32_t',
            self::UInt64 => 'uint64_t',
            self::Float32 => 'float',
            self::Float64 => 'double',
            self::Bool => 'uint8_t',
            self::Complex64 => 'float',
            self::Complex128 => 'double',
        };
    }

    /**
     * Get the size of each element in bytes.
     */
    public function itemSize(): int
    {
        return match ($this) {
            self::Int8, self::UInt8, self::Bool => 1,
            self::Int16, self::UInt16 => 2,
            self::Int32, self::UInt32, self::Float32 => 4,
            self::Int64, self::UInt64, self::Float64 => 8,
            self::Complex64 => 8,
            self::Complex128 => 16,
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
            self::UInt8, self::UInt16, self::UInt32, self::UInt64 => true,
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
     * Check if this dtype is a complex number type.
     */
    public function isComplex(): bool
    {
        return match ($this) {
            self::Complex64, self::Complex128 => true,
            default => false,
        };
    }

    /**
     * Check if this dtype is a boolean type.
     */
    public function isBool(): bool
    {
        return self::Bool === $this;
    }

    /**
     * Infer dtype from a PHP value.
     *
     * Uses NumPy conventions: bool -> Bool, int -> Int64, float -> Float64
     */
    public static function fromValue(mixed $value): self
    {
        if ($value instanceof Complex) {
            return self::Complex128;
        }

        if (\is_bool($value)) {
            return self::Bool;
        }

        if (\is_int($value)) {
            return self::Int64;
        }

        if (\is_float($value)) {
            return self::Float64;
        }

        throw new \InvalidArgumentException(
            \sprintf(
                "Cannot infer dtype from value of type '%s'. Expected bool, int, float, or Complex.",
                get_debug_type($value)
            )
        );
    }

    /**
     * Infer dtype from a PHP array, checking all elements.
     *
     * Type promotion: float > int > bool. Empty arrays default to Float64.
     *
     * @param array<mixed> $data
     */
    public static function fromArray(array $data): self
    {
        if (empty($data)) {
            return self::Float64;
        }

        $hasComplex = false;
        $hasFloat = false;
        $hasInt = false;
        $hasBool = false;

        $stack = [$data];
        while ($stack) {
            $current = array_pop($stack);
            foreach ($current as $item) {
                if (\is_array($item)) {
                    $stack[] = $item;
                } elseif ($item instanceof Complex) {
                    $hasComplex = true;
                } elseif (\is_float($item)) {
                    $hasFloat = true;
                } elseif (\is_int($item)) {
                    $hasInt = true;
                } elseif (\is_bool($item)) {
                    $hasBool = true;
                } else {
                    throw new \InvalidArgumentException(
                        \sprintf(
                            "Cannot infer dtype: array contains unsupported type '%s'.",
                            get_debug_type($item)
                        )
                    );
                }
            }
        }

        if ($hasComplex) {
            return self::Complex128;
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
    public function minValue(): Complex|float|int
    {
        return match ($this) {
            self::Int8 => -128,
            self::Int16 => -32768,
            self::Int32 => -2147483648,
            self::Int64 => \PHP_INT_MIN,
            self::UInt8, self::UInt16, self::UInt32, self::UInt64 => 0,
            self::Float32 => -3.4028235e+38,
            self::Float64 => -\PHP_FLOAT_MAX,
            self::Bool => 0,
            self::Complex64 => new Complex(-3.4028235e+38, -3.4028235e+38),
            self::Complex128 => new Complex(-\PHP_FLOAT_MAX, -\PHP_FLOAT_MAX),
        };
    }

    /**
     * Get the maximum value representable by this dtype.
     */
    public function maxValue(): Complex|float|int
    {
        return match ($this) {
            self::Int8 => 127,
            self::Int16 => 32767,
            self::Int32 => 2147483647,
            self::Int64 => \PHP_INT_MAX,
            self::UInt8 => 255,
            self::UInt16 => 65535,
            self::UInt32 => 4294967295,
            self::UInt64 => \PHP_INT_MAX,
            self::Float32 => 3.4028235e+38,
            self::Float64 => \PHP_FLOAT_MAX,
            self::Bool => 1,
            self::Complex64 => new Complex(3.4028235e+38, 3.4028235e+38),
            self::Complex128 => new Complex(\PHP_FLOAT_MAX, \PHP_FLOAT_MAX),
        };
    }

    /**
     * Create an FFI C value container for this dtype.
     *
     * If a value is provided, it will be stored in the C value with appropriate
     * type handling (e.g., bool is converted to 1/0).
     *
     * @param null|bool|Complex|float|int $value Optional value to store
     *
     * @return CData FFI CData representing this dtype
     */
    public function createCValue(bool|Complex|float|int|null $value = null): CData
    {
        $ffi = Lib::get();

        if ($this->isComplex()) {
            $cArray = $ffi->new("{$this->ffiType()}[2]");
            if (null !== $value) {
                $complex = $value instanceof Complex ? $value : new Complex((float) $value);
                $cArray[0] = $complex->real;
                $cArray[1] = $complex->imag;
            }

            return $cArray;
        }

        $cValue = $ffi->new($this->ffiType());

        if (null !== $value) {
            $cValue->cdata = $this->isBool() ? ($value ? 1 : 0) : $value;
        }

        return $cValue;
    }

    /**
     * Create an FFI C array for this dtype.
     *
     * If $values is non-empty, the array will be initialized with those values;
     * otherwise, a zero-filled array of the given element count will be allocated.
     * For complex types, the array length is $elementCount * 2 (real, imag).
     *
     * @param int          $elementCount Number of logical elements (not primitives); e.g., 4 means 4 floats or 4 complex numbers.
     * @param array<mixed> $values       Optional values to initialize the array with. Ignored if empty.
     *
     * @return CData allocated FFI C array for this dtype (primitive count, not logical elements, for complex types)
     */
    public function createCArray(int $elementCount, array $values = []): CData
    {
        $ffi = Lib::get();

        if (!empty($values)) {
            $cArray = $ffi->new("{$this->ffiType()}[".\count($values).']');
            foreach ($values as $i => $value) {
                $cArray[$i] = $value;
            }

            return $cArray;
        }

        $primitiveCount = $this->isComplex() ? $elementCount * 2 : $elementCount;

        return $ffi->new("{$this->ffiType()}[{$primitiveCount}]");
    }

    /**
     * Cast an FFI C value to the appropriate PHP type for this dtype.
     *
     * Handles type conversion: bools are cast from integer representation,
     * floats are cast to float, integers to int, complex to Complex object.
     *
     * @param null|bool|CData|float|int|string $cValue FFI CData containing the value
     *
     * @return bool|Complex|float|int The value cast to the appropriate PHP type
     */
    public function castFromCValue(bool|CData|float|int|string|null $cValue): bool|Complex|float|int
    {
        return match ($this) {
            self::Float64, self::Float32 => (float) $cValue,
            self::Bool => (bool) $cValue,
            self::Complex64 => new Complex((float) $cValue[0], (float) $cValue[1]),
            self::Complex128 => new Complex((float) $cValue[0], (float) $cValue[1]),
            default => (int) $cValue,
        };
    }

    /**
     * Prepare array values for FFI by converting bools to 1/0 and complex to flat pairs.
     *
     * This is necessary because FFI bools are represented as uint8_t,
     * so boolean values must be converted to integers.
     *
     * @param array<bool|Complex|float|int> $values Array of values to prepare
     *
     * @return array<float|int> Prepared values with bools converted and complex flattened
     */
    public function prepareArrayValues(array $values): array
    {
        if ($this->isComplex()) {
            $result = [];
            foreach ($values as $value) {
                $complex = $value instanceof Complex ? $value : new Complex((float) $value);
                $result[] = $complex->real;
                $result[] = $complex->imag;
            }

            return $result;
        }

        if (!$this->isBool()) {
            return $values;
        }

        return array_map(static fn ($v) => $v ? 1 : 0, $values);
    }
}
