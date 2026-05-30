<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray {
    use FFI\CData;
    use PhpMlKit\NDArray\Exceptions\ShapeException;

    /**
     * Functional proxies for {@see NDArray} instance and static APIs, grouped by trait.
     */

    // =============================================================================
    // CreatesArrays — factories, buffers, random, sequences, copy / astype
    // =============================================================================

    /**
     * Create array from PHP array.
     *
     * @param array<mixed>    $data  Nested PHP array
     * @param null|array<int> $shape Optional shape. If null, inferred from data.
     * @param null|DType      $dtype Data type (auto-inferred if null)
     */
    function nd_array(array $data, ?array $shape = null, ?DType $dtype = null): NDArray
    {
        return NDArray::fromArray($data, $shape, $dtype);
    }

    /**
     * Create an array filled with zeros.
     *
     * @param array<int> $shape Array shape
     * @param DType      $dtype Data type (default: Float64)
     */
    function zeros(array $shape, DType $dtype = DType::Float64): NDArray
    {
        return NDArray::zeros($shape, $dtype);
    }

    /**
     * Create an array filled with ones.
     *
     * @param array<int> $shape Array shape
     * @param DType      $dtype Data type (default: Float64)
     */
    function ones(array $shape, DType $dtype = DType::Float64): NDArray
    {
        return NDArray::ones($shape, $dtype);
    }

    /**
     * Create an array filled with a specific value.
     *
     * @param bool|Complex|float|int $value Value to fill array with
     * @param array<int>             $shape Array shape
     * @param null|DType             $dtype Data type (default: inferred from value)
     */
    function full(bool|Complex|float|int $value, array $shape, ?DType $dtype = null): NDArray
    {
        return NDArray::full($value, $shape, $dtype);
    }

    /**
     * Create an array from an external C buffer pointer.
     *
     * This method copies data from an external FFI buffer into a new NDArray. The source buffer
     * remains owned by the caller - this method creates an independent copy.
     *
     * @param CData      $buffer Pointer to raw C data (e.g., float*, int16_t*)
     * @param array<int> $shape  Array shape
     * @param DType      $dtype  Data type of the buffer
     *
     * @throws ShapeException If buffer size doesn't match shape
     */
    function from_buffer(CData $buffer, array $shape, DType $dtype): NDArray
    {
        return NDArray::fromBuffer($buffer, $shape, $dtype);
    }

    /**
     * Create an array from a binary string.
     *
     * This method interprets a PHP binary string as raw array data in little-endian format.
     * The bytes are copied into a new NDArray with the specified shape and dtype.
     *
     * @param string     $bytes Binary string containing raw array data (little-endian)
     * @param array<int> $shape Array shape
     * @param DType      $dtype Data type of the buffer
     *
     * @throws ShapeException If buffer size doesn't match shape
     */
    function from_bytes(string $bytes, array $shape, DType $dtype): NDArray
    {
        return NDArray::fromBytes($bytes, $shape, $dtype);
    }

    /**
     * Create an array of zeros with the same shape as the input array.
     *
     * @param NDArray    $array Input array defining the output shape
     * @param null|DType $dtype Data type (default: same as input array)
     */
    function zeros_like(NDArray $array, ?DType $dtype = null): NDArray
    {
        return NDArray::zerosLike($array, $dtype);
    }

    /**
     * Create an array of ones with the same shape as the input array.
     *
     * @param NDArray    $array Input array defining the output shape
     * @param null|DType $dtype Data type (default: same as input array)
     */
    function ones_like(NDArray $array, ?DType $dtype = null): NDArray
    {
        return NDArray::onesLike($array, $dtype);
    }

    /**
     * Create an array filled with a specific value, with the same shape as the input array.
     *
     * @param NDArray                $array Input array defining the output shape
     * @param bool|Complex|float|int $value Value to fill array with
     * @param null|DType             $dtype Data type (default: inferred from value or same as input)
     */
    function full_like(NDArray $array, bool|Complex|float|int $value, ?DType $dtype = null): NDArray
    {
        return NDArray::fullLike($array, $value, $dtype);
    }

    /**
     * Create coordinate matrices from one-dimensional coordinate vectors.
     *
     * @param array<array<mixed>|NDArray> $arrays   One-dimensional coordinate vectors
     * @param string                      $indexing Cartesian ('xy') or matrix ('ij') indexing
     * @param bool                        $sparse   Whether to return sparse coordinate grids
     *
     * @return array<NDArray> Coordinate grids, one for each input vector
     */
    function meshgrid(array $arrays, string $indexing = 'xy', bool $sparse = false): array
    {
        return NDArray::meshgrid($arrays, $indexing, $sparse);
    }

    /**
     * Create a 2D identity matrix.
     *
     * @param int      $N     Number of rows
     * @param null|int $M     Number of columns (default: N)
     * @param int      $k     Diagonal index (0: main, >0: upper, <0: lower)
     * @param DType    $dtype Data type (default: Float64)
     */
    function eye(int $N, ?int $M = null, int $k = 0, DType $dtype = DType::Float64): NDArray
    {
        return NDArray::eye($N, $M, $k, $dtype);
    }

    /**
     * Create evenly spaced values within a given interval.
     *
     * @param float|int      $start Start of interval (inclusive)
     * @param null|float|int $stop  End of interval (exclusive)
     * @param float|int      $step  Spacing between values
     * @param null|DType     $dtype Data type
     */
    function arange(float|int $start, float|int|null $stop = null, float|int $step = 1, ?DType $dtype = null): NDArray
    {
        return NDArray::arange($start, $stop, $step, $dtype);
    }

    /**
     * Create evenly spaced numbers over a specified interval.
     *
     * @param float $start    The starting value of the sequence
     * @param float $stop     The end value of the sequence
     * @param int   $num      Number of samples to generate
     * @param bool  $endpoint If true, stop is the last sample
     * @param DType $dtype    Data type (default: Float64)
     */
    function linspace(
        float $start,
        float $stop,
        int $num = 50,
        bool $endpoint = true,
        DType $dtype = DType::Float64,
    ): NDArray {
        return NDArray::linspace($start, $stop, $num, $endpoint, $dtype);
    }

    /**
     * Create numbers spaced evenly on a log scale.
     *
     * @param float $start The starting exponent (base**start is the first value)
     * @param float $stop  The end exponent (base**stop is the final value)
     * @param int   $num   Number of samples to generate
     * @param float $base  The base of the log space
     * @param DType $dtype Data type (default: Float64)
     */
    function logspace(
        float $start,
        float $stop,
        int $num = 50,
        float $base = 10.0,
        DType $dtype = DType::Float64,
    ): NDArray {
        return NDArray::logspace($start, $stop, $num, $base, $dtype);
    }

    /**
     * Create numbers spaced geometrically from start to stop.
     *
     * @param float $start The starting value of the sequence
     * @param float $stop  The end value of the sequence
     * @param int   $num   Number of samples to generate
     * @param DType $dtype Data type (default: Float64)
     */
    function geomspace(
        float $start,
        float $stop,
        int $num = 50,
        DType $dtype = DType::Float64,
    ): NDArray {
        return NDArray::geomspace($start, $stop, $num, $dtype);
    }

    /**
     * Create random samples from a uniform distribution over [0, 1).
     *
     * @param array<int> $shape Output shape
     * @param null|DType $dtype Float dtype (default: Float64)
     * @param null|int   $seed  Optional seed for deterministic output
     */
    function random(array $shape, ?DType $dtype = null, ?int $seed = null): NDArray
    {
        return NDArray::random($shape, $dtype, $seed);
    }

    /**
     * Create random integer samples from [low, high).
     *
     * @param int        $low   Inclusive lower bound
     * @param int        $high  Exclusive upper bound
     * @param array<int> $shape Output shape
     * @param null|DType $dtype Integer dtype (default: Int64)
     * @param null|int   $seed  Optional seed for deterministic output
     */
    function random_int(int $low, int $high, array $shape, ?DType $dtype = null, ?int $seed = null): NDArray
    {
        return NDArray::randomInt($low, $high, $shape, $dtype, $seed);
    }

    /**
     * Create random samples from a standard normal distribution N(0, 1).
     *
     * @param array<int> $shape Output shape
     * @param null|DType $dtype Float dtype (default: Float64)
     * @param null|int   $seed  Optional seed for deterministic output
     */
    function randn(array $shape, ?DType $dtype = null, ?int $seed = null): NDArray
    {
        return NDArray::randn($shape, $dtype, $seed);
    }

    /**
     * Create random samples from a normal distribution N(mean, std).
     *
     * @param float      $mean  Mean of the distribution
     * @param float      $std   Standard deviation (must be > 0)
     * @param array<int> $shape Output shape
     * @param null|DType $dtype Float dtype (default: Float64)
     * @param null|int   $seed  Optional seed for deterministic output
     */
    function normal(float $mean, float $std, array $shape, ?DType $dtype = null, ?int $seed = null): NDArray
    {
        return NDArray::normal($mean, $std, $shape, $dtype, $seed);
    }

    /**
     * Create random samples from a uniform distribution over [low, high).
     *
     * @param float      $low   Inclusive lower bound
     * @param float      $high  Exclusive upper bound
     * @param array<int> $shape Output shape
     * @param null|DType $dtype Float dtype (default: Float64)
     * @param null|int   $seed  Optional seed for deterministic output
     */
    function uniform(float $low, float $high, array $shape, ?DType $dtype = null, ?int $seed = null): NDArray
    {
        return NDArray::uniform($low, $high, $shape, $dtype, $seed);
    }

    /**
     * Tile an array by repeating it along each axis.
     *
     * Static convenience method that accepts either a PHP array or NDArray.
     *
     * @param array<int>|NDArray     $a    Input array or NDArray
     * @param array<int>|int|NDArray $reps The number of repetitions of A along each axis
     *
     * @return NDArray The tiled output array
     */
    function tile_array(array|NDArray $a, array|int|NDArray $reps): NDArray
    {
        return NDArray::tileArray($a, $reps);
    }

    /**
     * Repeat elements of an array.
     *
     * Static convenience method that accepts either a PHP array or NDArray.
     *
     * @param array<mixed>|NDArray   $a       Input array or NDArray
     * @param array<int>|int|NDArray $repeats The number of repetitions for each element
     * @param null|int               $axis    The axis along which to repeat values. By default, use the flattened input array
     *
     * @return NDArray Output array which has the same shape as a, except along the given axis
     */
    function repeat_array(array|NDArray $a, array|int|NDArray $repeats, ?int $axis = null): NDArray
    {
        return NDArray::repeatArray($a, $repeats, $axis);
    }

    /**
     * Create a deep copy of the array (or view).
     *
     * The returned array is always C-contiguous and owns its data.
     */
    function copy(NDArray $a): NDArray
    {
        return $a->copy();
    }

    /**
     * Cast array to a different data type.
     *
     * Returns a new array with the specified dtype. If the target dtype
     * is the same as the current dtype, this is equivalent to copy().
     *
     * @param DType $dtype Target data type
     *
     * @return NDArray New array with converted data
     */
    function astype(NDArray $a, DType $dtype): NDArray
    {
        return $a->astype($dtype);
    }

    // =============================================================================
    // HasMath — element-wise arithmetic, ufuncs, bitwise, clamp, min/max, softmax
    // =============================================================================

    /**
     * Add another array or scalar to this array.
     *
     * @param Complex|float|int|NDArray $other Array or scalar to add
     *
     * @return NDArray New array with result
     */
    function add(NDArray $a, Complex|float|int|NDArray $other): NDArray
    {
        return $a->add($other);
    }

    /**
     * Subtract another array or scalar from this array.
     *
     * @param Complex|float|int|NDArray $other Array or scalar to subtract
     *
     * @return NDArray New array with result
     */
    function subtract(NDArray $a, Complex|float|int|NDArray $other): NDArray
    {
        return $a->subtract($other);
    }

    /**
     * Multiply this array by another array or scalar.
     *
     * @param Complex|float|int|NDArray $other Array or scalar to multiply by
     *
     * @return NDArray New array with result
     */
    function multiply(NDArray $a, Complex|float|int|NDArray $other): NDArray
    {
        return $a->multiply($other);
    }

    /**
     * Divide this array by another array or scalar.
     *
     * @param Complex|float|int|NDArray $other Array or scalar to divide by
     *
     * @return NDArray New array with result
     */
    function divide(NDArray $a, Complex|float|int|NDArray $other): NDArray
    {
        return $a->divide($other);
    }

    /**
     * Compute remainder (modulo) with another array or scalar.
     *
     * @param Complex|float|int|NDArray $other Array or scalar
     *
     * @return NDArray New array with result
     */
    function rem(NDArray $a, Complex|float|int|NDArray $other): NDArray
    {
        return $a->rem($other);
    }

    /**
     * Compute modulo with another array or scalar.
     *
     * Alias for rem().
     *
     * @param Complex|float|int|NDArray $other Array or scalar
     *
     * @return NDArray New array with result
     */
    function mod(NDArray $a, Complex|float|int|NDArray $other): NDArray
    {
        return $a->mod($other);
    }

    /**
     * Compute absolute value element-wise.
     */
    function abs(NDArray $a): NDArray
    {
        return $a->abs();
    }

    /**
     * Compute negation element-wise (-$a).
     * Not supported for unsigned integers or bool.
     */
    function negative(NDArray $a): NDArray
    {
        return $a->negative();
    }

    /**
     * Extract real part element-wise.
     *
     * For complex arrays, returns the real component as float.
     * For real arrays, returns a copy with the same dtype.
     */
    function real(NDArray $a): NDArray
    {
        return $a->real();
    }

    /**
     * Extract imaginary part element-wise.
     *
     * For complex arrays, returns the imaginary component as float.
     * For real arrays, returns zeros with the same dtype.
     */
    function imag(NDArray $a): NDArray
    {
        return $a->imag();
    }

    /**
     * Compute complex conjugate element-wise.
     *
     * For complex arrays, negates the imaginary part.
     * For real arrays, returns a copy with the same dtype.
     */
    function conjugate(NDArray $a): NDArray
    {
        return $a->conjugate();
    }

    /**
     * Alias for conjugate().
     */
    function conj(NDArray $a): NDArray
    {
        return $a->conj();
    }

    /**
     * Returns bool array: true where element has non-zero imaginary part.
     *
     * For complex arrays, checks if imag ≠ 0.
     * For real arrays, always returns false.
     */
    function iscomplex(NDArray $a): NDArray
    {
        return $a->iscomplex();
    }

    /**
     * Returns bool array: true where element has zero imaginary part.
     *
     * For complex arrays, checks if imag = 0.
     * For real arrays, always returns true.
     */
    function isreal(NDArray $a): NDArray
    {
        return $a->isreal();
    }

    /**
     * Compute phase angle element-wise.
     *
     * Returns the counterclockwise angle from the positive real axis.
     * Always returns Float64 regardless of input dtype.
     *
     * @param bool $deg if true, returns angle in degrees; otherwise radians
     */
    function angle(NDArray $a, bool $deg = false): NDArray
    {
        return $a->angle($deg);
    }

    /**
     * Compute square root element-wise.
     */
    function sqrt(NDArray $a): NDArray
    {
        return $a->sqrt();
    }

    /**
     * Compute exponential element-wise.
     */
    function exp(NDArray $a): NDArray
    {
        return $a->exp();
    }

    /**
     * Compute natural logarithm element-wise.
     */
    function log(NDArray $a): NDArray
    {
        return $a->log();
    }

    /**
     * Compute natural logarithm element-wise.
     *
     * Alias for log().
     */
    function ln(NDArray $a): NDArray
    {
        return $a->ln();
    }

    /**
     * Compute sine element-wise.
     */
    function sin(NDArray $a): NDArray
    {
        return $a->sin();
    }

    /**
     * Compute cosine element-wise.
     */
    function cos(NDArray $a): NDArray
    {
        return $a->cos();
    }

    /**
     * Compute tangent element-wise.
     */
    function tan(NDArray $a): NDArray
    {
        return $a->tan();
    }

    /**
     * Compute hyperbolic sine element-wise.
     */
    function sinh(NDArray $a): NDArray
    {
        return $a->sinh();
    }

    /**
     * Compute hyperbolic cosine element-wise.
     */
    function cosh(NDArray $a): NDArray
    {
        return $a->cosh();
    }

    /**
     * Compute hyperbolic tangent element-wise.
     */
    function tanh(NDArray $a): NDArray
    {
        return $a->tanh();
    }

    /**
     * Compute arc sine element-wise.
     */
    function asin(NDArray $a): NDArray
    {
        return $a->asin();
    }

    /**
     * Compute arc cosine element-wise.
     */
    function acos(NDArray $a): NDArray
    {
        return $a->acos();
    }

    /**
     * Compute arc tangent element-wise.
     */
    function atan(NDArray $a): NDArray
    {
        return $a->atan();
    }

    /**
     * Compute cube root element-wise.
     */
    function cbrt(NDArray $a): NDArray
    {
        return $a->cbrt();
    }

    /**
     * Compute ceiling element-wise.
     */
    function ceil(NDArray $a): NDArray
    {
        return $a->ceil();
    }

    /**
     * Compute base-2 exponential (2^x) element-wise.
     */
    function exp2(NDArray $a): NDArray
    {
        return $a->exp2();
    }

    /**
     * Compute floor element-wise.
     */
    function floor(NDArray $a): NDArray
    {
        return $a->floor();
    }

    /**
     * Compute base-2 logarithm element-wise.
     */
    function log2(NDArray $a): NDArray
    {
        return $a->log2();
    }

    /**
     * Compute base-10 logarithm element-wise.
     */
    function log10(NDArray $a): NDArray
    {
        return $a->log10();
    }

    /**
     * Compute x^2 (square) element-wise.
     */
    function pow2(NDArray $a): NDArray
    {
        return $a->pow2();
    }

    /**
     * Compute round element-wise.
     */
    function round(NDArray $a): NDArray
    {
        return $a->round();
    }

    /**
     * Compute signum element-wise.
     */
    function signum(NDArray $a): NDArray
    {
        return $a->signum();
    }

    /**
     * Compute reciprocal (1/x) element-wise.
     */
    function recip(NDArray $a): NDArray
    {
        return $a->recip();
    }

    /**
     * Compute ln(1+x) element-wise.
     *
     * More accurate than log(1+x) for small x.
     */
    function ln1p(NDArray $a): NDArray
    {
        return $a->ln1p();
    }

    /**
     * Convert radians to degrees element-wise.
     */
    function to_degrees(NDArray $a): NDArray
    {
        return $a->toDegrees();
    }

    /**
     * Convert degrees to radians element-wise.
     */
    function to_radians(NDArray $a): NDArray
    {
        return $a->toRadians();
    }

    /**
     * Compute x^n where n is an integer, element-wise.
     *
     * Generally faster than pow() for integer exponents.
     *
     * @param int $exp Integer exponent
     */
    function powi(NDArray $a, int $exp): NDArray
    {
        return $a->powi($exp);
    }

    /**
     * Compute x^y where y is a float, element-wise.
     *
     * @param float $exp Float exponent
     */
    function powf(NDArray $a, float $exp): NDArray
    {
        return $a->powf($exp);
    }

    /**
     * Compute hypotenuse element-wise.
     *
     * @param float $other Scalar value
     */
    function hypot(NDArray $a, float $other): NDArray
    {
        return $a->hypot($other);
    }

    /**
     * Bitwise AND with another array or scalar.
     *
     * @param int|NDArray $other Array or scalar
     *
     * @return NDArray New array with result
     */
    function bitand(NDArray $a, int|NDArray $other): NDArray
    {
        return $a->bitand($other);
    }

    /**
     * Bitwise OR with another array or scalar.
     *
     * @param int|NDArray $other Array or scalar
     *
     * @return NDArray New array with result
     */
    function bitor(NDArray $a, int|NDArray $other): NDArray
    {
        return $a->bitor($other);
    }

    /**
     * Bitwise XOR with another array or scalar.
     *
     * @param int|NDArray $other Array or scalar
     *
     * @return NDArray New array with result
     */
    function bitxor(NDArray $a, int|NDArray $other): NDArray
    {
        return $a->bitxor($other);
    }

    /**
     * Left shift by another array or scalar.
     *
     * @param int|NDArray $other Array or scalar (number of bits)
     *
     * @return NDArray New array with result
     */
    function left_shift(NDArray $a, int|NDArray $other): NDArray
    {
        return $a->leftShift($other);
    }

    /**
     * Right shift by another array or scalar.
     *
     * @param int|NDArray $other Array or scalar (number of bits)
     *
     * @return NDArray New array with result
     */
    function right_shift(NDArray $a, int|NDArray $other): NDArray
    {
        return $a->rightShift($other);
    }

    /**
     * Clamp (clip) array values to a specified range.
     *
     * Similar to NumPy's clip function. Values outside [min, max] are set
     * to the nearest boundary.
     *
     * @param float $min Minimum value
     * @param float $max Maximum value
     *
     * @throws \InvalidArgumentException If min > max
     */
    function clamp(NDArray $a, float|int $min, float|int $max): NDArray
    {
        return $a->clamp($min, $max);
    }

    /**
     * Clip array values to a specified range.
     *
     * Alias for clamp().
     *
     * @param float $min Minimum value
     * @param float $max Maximum value
     *
     * @throws \InvalidArgumentException If min > max
     */
    function clip(NDArray $a, float|int $min, float|int $max): NDArray
    {
        return $a->clip($min, $max);
    }

    /**
     * Element-wise minimum of two arrays.
     *
     * Compares two arrays element-wise and returns a new array containing
     * the smaller value at each position. Supports broadcasting.
     *
     * @param NDArray $other The array to compare with
     *
     * @return NDArray New array with element-wise minimum values
     */
    function minimum(NDArray $a, NDArray $other): NDArray
    {
        return $a->minimum($other);
    }

    /**
     * Element-wise maximum of two arrays.
     *
     * Compares two arrays element-wise and returns a new array containing
     * the larger value at each position. Supports broadcasting.
     *
     * @param NDArray $other The array to compare with
     *
     * @return NDArray New array with element-wise maximum values
     */
    function maximum(NDArray $a, NDArray $other): NDArray
    {
        return $a->maximum($other);
    }

    /**
     * Compute sigmoid element-wise: 1 / (1 + exp(-x)).
     */
    function sigmoid(NDArray $a): NDArray
    {
        return $a->sigmoid();
    }

    /**
     * Compute softmax along axis: exp(x - max) / sum(exp(x - max)).
     *
     * Numerically stable. Default axis -1 (last axis) for typical logits.
     *
     * @param int $axis Axis along which to compute softmax
     */
    function softmax(NDArray $a, int $axis = -1): NDArray
    {
        return $a->softmax($axis);
    }

    // =============================================================================
    // HasComparison — element-wise comparisons (Bool result)
    // =============================================================================

    /**
     * Element-wise equal comparison. Returns Bool array.
     */
    function eq(NDArray $a, Complex|float|int|NDArray $other): NDArray
    {
        return $a->eq($other);
    }

    /**
     * Element-wise not-equal comparison. Returns Bool array.
     */
    function ne(NDArray $a, Complex|float|int|NDArray $other): NDArray
    {
        return $a->ne($other);
    }

    /**
     * Element-wise greater-than comparison. Returns Bool array.
     */
    function gt(NDArray $a, Complex|float|int|NDArray $other): NDArray
    {
        return $a->gt($other);
    }

    /**
     * Element-wise greater-or-equal comparison. Returns Bool array.
     */
    function gte(NDArray $a, Complex|float|int|NDArray $other): NDArray
    {
        return $a->gte($other);
    }

    /**
     * Element-wise less-than comparison. Returns Bool array.
     */
    function lt(NDArray $a, Complex|float|int|NDArray $other): NDArray
    {
        return $a->lt($other);
    }

    /**
     * Element-wise less-or-equal comparison. Returns Bool array.
     */
    function lte(NDArray $a, Complex|float|int|NDArray $other): NDArray
    {
        return $a->lte($other);
    }

    // =============================================================================
    // HasLogical — element-wise logical (Bool result).
    // =============================================================================

    /**
     * Element-wise logical AND. Returns Bool array.
     *
     * Works with any input type (converts to bool first).
     * Zero values are treated as false, non-zero as true.
     *
     * @see NDArray::and()
     */
    function logical_and(NDArray $a, NDArray $other): NDArray
    {
        return $a->and($other);
    }

    /**
     * Element-wise logical OR. Returns Bool array.
     *
     * Works with any input type (converts to bool first).
     * Zero values are treated as false, non-zero as true.
     *
     * @see NDArray::or()
     */
    function logical_or(NDArray $a, NDArray $other): NDArray
    {
        return $a->or($other);
    }

    /**
     * Element-wise logical NOT. Returns Bool array.
     *
     * Works with any input type (converts to bool first).
     * Zero values are treated as false, non-zero as true.
     *
     * @see NDArray::not()
     */
    function logical_not(NDArray $a): NDArray
    {
        return $a->not();
    }

    /**
     * Element-wise logical XOR. Returns Bool array.
     *
     * Works with any input type (converts to bool first).
     * Zero values are treated as false, non-zero as true.
     *
     * @see NDArray::xor()
     */
    function logical_xor(NDArray $a, NDArray $other): NDArray
    {
        return $a->xor($other);
    }

    // =============================================================================
    // HasReductions — reductions, sort, topk, cumulative, bincount
    // =============================================================================

    /**
     * Sum of array elements over a given axis.
     *
     * @param null|int $axis     Axis along which to sum. If null, sum over all elements.
     * @param bool     $keepdims if true, the reduced axis is retained with size 1
     *
     * @return Complex|float|int|NDArray scalar if axis is null, otherwise an NDArray
     */
    function sum(NDArray $a, ?int $axis = null, bool $keepdims = false): Complex|float|int|NDArray
    {
        return $a->sum($axis, $keepdims);
    }

    /**
     * Mean of array elements over a given axis.
     *
     * @param null|int $axis     Axis along which to compute mean. If null, compute mean of all elements.
     * @param bool     $keepdims if true, the reduced axis is retained with size 1
     *
     * @return Complex|float|NDArray scalar if axis is null, otherwise an NDArray
     */
    function mean(NDArray $a, ?int $axis = null, bool $keepdims = false): Complex|float|NDArray
    {
        return $a->mean($axis, $keepdims);
    }

    /**
     * Minimum of array elements over a given axis.
     *
     * @param null|int $axis     Axis along which to find minimum. If null, find minimum of all elements.
     * @param bool     $keepdims if true, the reduced axis is retained with size 1
     *
     * @return Complex|float|int|NDArray scalar if axis is null, otherwise an NDArray
     *
     * @see NDArray::min() Named `amin` (NumPy-style) so this file does not shadow PHP's `min()`.
     */
    function amin(NDArray $a, ?int $axis = null, bool $keepdims = false): Complex|float|int|NDArray
    {
        return $a->min($axis, $keepdims);
    }

    /**
     * Maximum of array elements over a given axis.
     *
     * @param null|int $axis     Axis along which to find maximum. If null, find maximum of all elements.
     * @param bool     $keepdims if true, the reduced axis is retained with size 1
     *
     * @return Complex|float|int|NDArray scalar if axis is null, otherwise an NDArray
     *
     * @see NDArray::max() Named `amax` (NumPy-style) so this file does not shadow PHP's `max()`.
     */
    function amax(NDArray $a, ?int $axis = null, bool $keepdims = false): Complex|float|int|NDArray
    {
        return $a->max($axis, $keepdims);
    }

    /**
     * Index of minimum value over a given axis.
     *
     * @param null|int $axis     Axis along which to find argmin. If null, find argmin of flattened array.
     * @param bool     $keepdims if true, the reduced axis is retained with size 1
     *
     * @return int|NDArray scalar index if axis is null, otherwise an NDArray of indices
     */
    function argmin(NDArray $a, ?int $axis = null, bool $keepdims = false): int|NDArray
    {
        return $a->argmin($axis, $keepdims);
    }

    /**
     * Index of maximum value over a given axis.
     *
     * @param null|int $axis     Axis along which to find argmax. If null, find argmax of flattened array.
     * @param bool     $keepdims if true, the reduced axis is retained with size 1
     *
     * @return int|NDArray scalar index if axis is null, otherwise an NDArray of indices
     */
    function argmax(NDArray $a, ?int $axis = null, bool $keepdims = false): int|NDArray
    {
        return $a->argmax($axis, $keepdims);
    }

    /**
     * Return a sorted copy of the array.
     *
     * @param null|int $axis Axis along which to sort. If null, sort flattened data.
     * @param SortKind $kind sorting algorithm
     */
    function sort(NDArray $a, ?int $axis = -1, SortKind $kind = SortKind::QuickSort): NDArray
    {
        return $a->sort($axis, $kind);
    }

    /**
     * Return indices that would sort the array.
     *
     * @param null|int $axis Axis along which to argsort. If null, argsort flattened data.
     * @param SortKind $kind sorting algorithm
     *
     * @return NDArray int64 indices array
     */
    function argsort(NDArray $a, ?int $axis = -1, SortKind $kind = SortKind::QuickSort): NDArray
    {
        return $a->argsort($axis, $kind);
    }

    /**
     * Return top-k values and indices like PyTorch topk.
     *
     * @param int      $k       Number of elements to select
     * @param null|int $axis    Axis along which to select. If null, flatten first.
     * @param bool     $largest If true, select largest values; otherwise smallest values
     * @param bool     $sorted  If true, keep selected values sorted by rank
     * @param SortKind $kind    Sorting algorithm
     *
     * @return array{values: NDArray, indices: NDArray}
     */
    function topk(
        NDArray $a,
        int $k,
        ?int $axis = -1,
        bool $largest = true,
        bool $sorted = true,
        SortKind $kind = SortKind::QuickSort,
    ): array {
        return $a->topk($k, $axis, $largest, $sorted, $kind);
    }

    /**
     * Product of array elements over a given axis.
     *
     * @param null|int $axis     Axis along which to compute product. If null, compute product of all elements.
     * @param bool     $keepdims if true, the reduced axis is retained with size 1
     *
     * @return Complex|float|int|NDArray scalar if axis is null, otherwise an NDArray
     */
    function product(NDArray $a, ?int $axis = null, bool $keepdims = false): Complex|float|int|NDArray
    {
        return $a->product($axis, $keepdims);
    }

    /**
     * Cumulative sum of array elements.
     *
     * @param null|int $axis Axis along which to compute cumulative sum. If null, flatten and return 1D.
     */
    function cumsum(NDArray $a, ?int $axis = null): NDArray
    {
        return $a->cumsum($axis);
    }

    /**
     * Cumulative product of array elements.
     *
     * @param null|int $axis Axis along which to compute cumulative product. If null, flatten and return 1D.
     */
    function cumprod(NDArray $a, ?int $axis = null): NDArray
    {
        return $a->cumprod($axis);
    }

    /**
     * Variance of array elements over a given axis.
     *
     * @param null|int $axis     Axis along which to compute variance. If null, compute variance of all elements.
     * @param int      $ddof     delta degrees of freedom (0 for population, 1 for sample)
     * @param bool     $keepdims if true, the reduced axis is retained with size 1
     *
     * @return float|NDArray scalar if axis is null, otherwise an NDArray
     */
    function variance(NDArray $a, ?int $axis = null, int $ddof = 0, bool $keepdims = false): float|NDArray
    {
        return $a->var($axis, $ddof, $keepdims);
    }

    /**
     * Standard deviation of array elements over a given axis.
     *
     * @param null|int $axis     Axis along which to compute std. If null, compute std of all elements.
     * @param int      $ddof     delta degrees of freedom (0 for population, 1 for sample)
     * @param bool     $keepdims if true, the reduced axis is retained with size 1
     *
     * @return float|NDArray scalar if axis is null, otherwise an NDArray
     */
    function std(NDArray $a, ?int $axis = null, int $ddof = 0, bool $keepdims = false): float|NDArray
    {
        return $a->std($axis, $ddof, $keepdims);
    }

    /**
     * Count occurrences of non-negative integer values in flattened input.
     *
     * @param null|int $minlength Minimum output length
     *
     * @return NDArray Int64 counts array
     */
    function bincount(NDArray $a, ?int $minlength = null): NDArray
    {
        return $a->bincount($minlength);
    }

    // =============================================================================
    // HasShapeOps — shape, views, pad, tile, repeat
    // =============================================================================

    /**
     * Pad an array.
     *
     * @param array<array{int,int}|int>|array{int,int}|int $padWidth       Number of elements to pad on each side of each axis.
     *                                                                     - int: pad same amount on all sides of every axis
     *                                                                     - array{int,int}: pad [before, after] for all axes
     *                                                                     - array<int|array{int,int}>: per-axis pad (int or [before, after] per axis)
     * @param PadMode                                      $mode           padding mode
     * @param array<bool|float|int>|bool|float|int         $constantValues constant value to pad with (used for PadMode::Constant)
     *
     * @return NDArray padded array
     */
    function pad(NDArray $a, array|int $padWidth, PadMode $mode = PadMode::Constant, array|bool|float|int $constantValues = 0): NDArray
    {
        return $a->pad($padWidth, $mode, $constantValues);
    }

    /**
     * Reshape the array to a new shape.
     *
     * Returns a new array with the specified shape.
     * Supports both C-order (row-major, order='C') and F-order (column-major, order='F').
     *
     * For contiguous arrays, this returns a zero-copy view with updated metadata.
     * For non-contiguous arrays, data is copied to make it contiguous first.
     *
     * @param array<int> $newShape New shape
     * @param string     $order    Memory layout: 'C' for row-major, 'F' for column-major
     */
    function reshape(NDArray $a, array $newShape, string $order = 'C'): NDArray
    {
        return $a->reshape($newShape, $order);
    }

    /**
     * Transpose the array.
     *
     * For 2D arrays, swaps rows and columns.
     * For nD arrays, reverses the order of all axes.
     *
     * For contiguous arrays, this is a zero-copy operation returning a view.
     * For non-contiguous arrays, data is copied first.
     */
    function transpose(NDArray $a): NDArray
    {
        return $a->transpose();
    }

    /**
     * Swap two axes of the array.
     *
     * This is a zero-copy operation that returns a view with swapped
     * shape and stride metadata. The underlying data is shared.
     *
     * @param int $axis1 First axis to swap
     * @param int $axis2 Second axis to swap
     */
    function swapaxes(NDArray $a, int $axis1, int $axis2): NDArray
    {
        return $a->swapaxes($axis1, $axis2);
    }

    /**
     * Permute axes of the array.
     *
     * Reorders the axes according to the given permutation.
     * For example, permute(1, 0) on a 2D array is equivalent to transpose().
     *
     * @param int ...$axes New order of axes
     */
    function permute(NDArray $a, int ...$axes): NDArray
    {
        return $a->permute(...$axes);
    }

    /**
     * Merge axes by combining take into into.
     *
     * Merges the axis 'take' into axis 'into' by folding the dimensions.
     * The 'take' axis is removed and its size is multiplied into the 'into' axis.
     * This is a zero-copy operation that returns a view with updated metadata.
     *
     * For example, on an array with shape [2, 3, 4]:
     * - mergeaxes(1, 0) merges axis 1 into axis 0, resulting in shape [6, 4]
     * - mergeaxes(0, 1) merges axis 0 into axis 1, resulting in shape [3, 8]
     *
     * @param int $take Axis to merge from (will be removed)
     * @param int $into Axis to merge into (size will be multiplied)
     */
    function mergeaxes(NDArray $a, int $take, int $into): NDArray
    {
        return $a->mergeaxes($take, $into);
    }

    /**
     * Reverse the order of elements in an array along the given axis or axes.
     *
     * @param null|array<int>|int $axes Axis or axes to flip. If null, flip over all axes.
     */
    function flip(NDArray $a, array|int|null $axes = null): NDArray
    {
        return $a->flip($axes);
    }

    /**
     * Insert a new axis at the specified position.
     *
     * The new axis always has length 1. This is a zero-copy operation that returns
     * a view with updated metadata.
     *
     * @param int $axis Position where new axis is inserted
     */
    function insertaxis(NDArray $a, int $axis): NDArray
    {
        return $a->insertaxis($axis);
    }

    /**
     * Flatten the array to 1D.
     *
     * Always returns a copy in C-order (row-major).
     */
    function flatten(NDArray $a): NDArray
    {
        return $a->flatten();
    }

    /**
     * Ravel the array to 1D.
     *
     * Similar to flatten() but returns a view if the array is contiguous.
     * For non-contiguous arrays, data is copied to make it contiguous.
     *
     * @param string $order Memory layout: 'C' for row-major, 'F' for column-major
     */
    function ravel(NDArray $a, string $order = 'C'): NDArray
    {
        return $a->ravel($order);
    }

    /**
     * Remove axes of length 1 from the array.
     *
     * If no axes are specified, removes all length-1 axes (NumPy behavior).
     * This is a zero-copy operation that returns a view with updated metadata.
     *
     * @param null|array<int> $axes Specific axes to squeeze (null for all)
     */
    function squeeze(NDArray $a, ?array $axes = null): NDArray
    {
        return $a->squeeze($axes);
    }

    /**
     * Expand dimensions by inserting a new axis.
     *
     * Alias for insertaxis().
     *
     * @param int $axis Position where new axis is inserted
     */
    function expand_dims(NDArray $a, int $axis): NDArray
    {
        return $a->expandDims($axis);
    }

    /**
     * Construct an array by repeating A the number of times given by reps.
     *
     * @param array<int>|int|NDArray $reps the number of repetitions of A along each axis
     *
     * @return NDArray the tiled output array
     */
    function tile(NDArray $a, array|int|NDArray $reps): NDArray
    {
        return $a->tile($reps);
    }

    /**
     * Repeat elements of an array.
     *
     * @param array<int>|int|NDArray $repeats the number of repetitions for each element
     * @param null|int               $axis    The axis along which to repeat values. By default, use the flattened input array.
     *
     * @return NDArray output array which has the same shape as a, except along the given axis
     */
    function repeat(NDArray $a, array|int|NDArray $repeats, ?int $axis = null): NDArray
    {
        return $a->repeat($repeats, $axis);
    }

    // =============================================================================
    // HasStacking — concatenate, stack, split
    // =============================================================================

    /**
     * Join arrays along an existing axis.
     *
     * All arrays must have the same shape except for the dimension along axis.
     *
     * @param array<NDArray> $arrays Arrays to concatenate
     * @param int            $axis   Axis along which to join (default 0)
     */
    function concatenate(array $arrays, int $axis = 0): NDArray
    {
        return NDArray::concatenate($arrays, $axis);
    }

    /**
     * Stack arrays along a new axis.
     *
     * All arrays must have identical shapes.
     *
     * @param array<NDArray> $arrays Arrays to stack
     * @param int            $axis   Axis in the result at which the arrays are stacked
     */
    function stack(array $arrays, int $axis = 0): NDArray
    {
        return NDArray::stack($arrays, $axis);
    }

    /**
     * Stack arrays vertically (along axis 0).
     *
     * Equivalent to concatenate(arrays, axis=0).
     *
     * @param array<NDArray> $arrays Arrays to stack
     */
    function vstack(array $arrays): NDArray
    {
        return NDArray::vstack($arrays);
    }

    /**
     * Stack arrays horizontally (along axis 1).
     *
     * Equivalent to concatenate(arrays, axis=1).
     *
     * @param array<NDArray> $arrays Arrays to stack
     */
    function hstack(array $arrays): NDArray
    {
        return NDArray::hstack($arrays);
    }

    /**
     * Split array along axis.
     *
     * If $indicesOrSections is an integer N, split into N equal parts (axis length must be divisible by N).
     * If it is an array of indices, split at those positions.
     *
     * @param array<int>|int $indicesOrSections Number of equal parts, or array of split indices
     * @param int            $axis              Axis along which to split
     *
     * @return array<NDArray> List of sub-arrays (views)
     */
    function split(NDArray $a, array|int $indicesOrSections, int $axis = 0): array
    {
        return $a->split($indicesOrSections, $axis);
    }

    /**
     * Split array vertically (along axis 0).
     *
     * @param array<int>|int $indicesOrSections Number of equal parts or split indices
     *
     * @return array<NDArray>
     */
    function vsplit(NDArray $a, array|int $indicesOrSections): array
    {
        return $a->vsplit($indicesOrSections);
    }

    /**
     * Split array horizontally (along axis 1).
     *
     * @param array<int>|int $indicesOrSections Number of equal parts or split indices
     *
     * @return array<NDArray>
     */
    function hsplit(NDArray $a, array|int $indicesOrSections): array
    {
        return $a->hsplit($indicesOrSections);
    }

    // =============================================================================
    // HasIndexing — get/set, take/put, scatter, where
    // =============================================================================

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
     * @return bool|float|int|NDArray Scalar for full indexing, view for partial
     */
    function get(NDArray $a, int ...$indices): bool|Complex|float|int|NDArray
    {
        return $a->get(...$indices);
    }

    /**
     * Set a scalar value at the given indices.
     *
     * Requires full indexing (count === ndim).
     *
     * Supports negative indices: -1 refers to the last element, -2 to second-to-last, etc.
     *
     * @param array<int>             $indices Indices for each dimension
     * @param bool|Complex|float|int $value   Value to set
     */
    function set(NDArray $a, array $indices, bool|Complex|float|int $value): void
    {
        $a->set($indices, $value);
    }

    /**
     * Set a scalar value using a logical flat index (C-order) for this array/view.
     *
     * Supports negative indices: -1 refers to the last logical element.
     *
     * @param int            $flatIndex Logical flat index into this array/view
     * @param bool|float|int $value     Value to set
     */
    function set_at(NDArray $a, int $flatIndex, bool|Complex|float|int $value): void
    {
        $a->setAt($flatIndex, $value);
    }

    /**
     * Get a scalar value using a logical flat index (C-order) for this array/view.
     *
     * Supports negative indices: -1 refers to the last logical element.
     *
     * @param int $flatIndex Logical flat index into this array/view
     *
     * @return bool|float|int The value at the specified flat index
     */
    function get_at(NDArray $a, int $flatIndex): bool|Complex|float|int
    {
        return $a->getAt($flatIndex);
    }

    /**
     * Gather values by indices.
     *
     * If axis is null, gathers from logical flattened view (C-order).
     * If axis is provided, delegates to takeAlongAxis semantics.
     *
     * @param array<array<int>|int>|NDArray $indices
     */
    function take(NDArray $a, array|NDArray $indices, ?int $axis = null): NDArray
    {
        return $a->take($indices, $axis);
    }

    /**
     * Gather values along an axis using per-position indices.
     *
     * @param array<array<int>|int>|NDArray $indices
     */
    function take_along_axis(NDArray $a, array|NDArray $indices, int $axis): NDArray
    {
        return $a->takeAlongAxis($indices, $axis);
    }

    /**
     * Scatter values by flattened indices and return a mutated copy.
     *
     * @param array<array<int>|int>|NDArray $indices
     * @param string                        $mode    Currently supports only 'raise'
     */
    function put(NDArray $a, array|NDArray $indices, bool|float|int|NDArray $values, string $mode = 'raise'): NDArray
    {
        return $a->put($indices, $values, $mode);
    }

    /**
     * Scatter values along an axis and return a mutated copy.
     *
     * @param array<array<int>|int>|NDArray $indices
     */
    function put_along_axis(NDArray $a, array|NDArray $indices, bool|float|int|NDArray $values, int $axis): NDArray
    {
        return $a->putAlongAxis($indices, $values, $axis);
    }

    /**
     * Add updates by flattened indices and return a mutated copy.
     *
     * @param array<array<int>|int>|NDArray $indices
     */
    function scatter_add(NDArray $a, array|NDArray $indices, bool|float|int|NDArray $updates): NDArray
    {
        return $a->scatterAdd($indices, $updates);
    }

    /**
     * Select values from x and y based on a boolean condition.
     *
     * @param bool|float|int|NDArray $condition Bool NDArray or scalar condition
     * @param bool|float|int|NDArray $x         Values where condition is true
     * @param bool|float|int|NDArray $y         Values where condition is false
     */
    function where(bool|float|int|NDArray $condition, bool|float|int|NDArray $x, bool|float|int|NDArray $y): NDArray
    {
        return NDArray::where($condition, $x, $y);
    }

    // =============================================================================
    // HasSlicing — slice views, assign
    // =============================================================================

    /**
     * Create a view of the array using slice syntax.
     *
     * Selectors can be integers (to reduce dimensions) or slice strings
     * (e.g., "0:5", ":", "::2") to subset dimensions.
     *
     * @param array<int|string> $selection List of selectors for each dimension
     *
     * @return NDArray A new view sharing the same data
     */
    function slice(NDArray $a, array $selection): NDArray
    {
        return $a->slice($selection);
    }

    /**
     * Assign values to the current array/view.
     *
     * Supports scalar assignment (fill) or NDArray assignment (rhs is broadcast to this view’s shape when compatible).
     *
     * @param bool|Complex|float|int|NDArray $value Scalar value or NDArray
     */
    function assign(NDArray $a, bool|Complex|float|int|NDArray $value): void
    {
        $a->assign($value);
    }
}

namespace PhpMlKit\NDArray\Linalg {
    use PhpMlKit\NDArray\Complex;
    use PhpMlKit\NDArray\NDArray;

    /**
     * Compute vector or matrix norm.
     *
     * Supported orders:
     * - 1
     * - 2
     * - INF
     * - -INF
     * - 'fro' (matrix only, axis=null)
     *
     * @param null|float|int|string $ord      Norm order
     * @param null|int              $axis     Reduction axis. If null, reduces all elements
     * @param bool                  $keepdims Keep reduced axis with size 1 (axis mode only)
     */
    function norm(NDArray $a, float|int|string|null $ord = null, ?int $axis = null, bool $keepdims = false): float|NDArray
    {
        return $a->norm($ord, $axis, $keepdims);
    }

    /**
     * Generalized dot product for 1D and 2D operands.
     *
     * Operand dtypes are promoted to a common type before the operation. Shape rules:
     * - **1D × 1D**: inner product → scalar
     * - **2D × 2D**: matrix product → 2D array
     * - **1D × 2D** or **2D × 1D**: vector–matrix product → 1D array
     *
     * @param NDArray $other The other array
     *
     * @return Complex|float|int|NDArray scalar when the result is 0-D, otherwise an NDArray
     */
    function dot(NDArray $a, NDArray $other): Complex|float|int|NDArray
    {
        return $a->dot($other);
    }

    /**
     * Matrix multiplication (`@`) for 1D and 2D operands.
     *
     * Operand dtypes are promoted to a common type before the operation. Supported shapes:
     * - **2D × 2D**: matrix × matrix → 2D array
     * - **2D × 1D** or **1D × 2D**: matrix × vector → 1D array
     * - **1D × 1D**: inner product → scalar (0-D result unpacked to a PHP scalar or `Complex`)
     *
     * Operands with more than two dimensions are not supported.
     *
     * @param NDArray $other The other array
     *
     * @return Complex|float|int|NDArray scalar when the result is 0-D, otherwise an NDArray
     */
    function matmul(NDArray $a, NDArray $other): Complex|float|int|NDArray
    {
        return $a->matmul($other);
    }

    /**
     * Extract diagonal elements from a 2D array.
     *
     * Returns a 1D array containing the diagonal.
     *
     * @param int $offset Diagonal offset. 0 = main diagonal, positive = upper diagonal, negative = lower diagonal
     */
    function diagonal(NDArray $a, int $offset = 0): NDArray
    {
        return $a->diagonal($offset);
    }

    /**
     * Extract a diagonal or construct a diagonal array.
     *
     * If the input is 1D: returns a 2D array with the input as the diagonal.
     * If the input is 2D: returns a 1D array containing the diagonal.
     *
     * @param int $offset Diagonal offset. 0 = main diagonal, positive = upper diagonal, negative = lower diagonal
     */
    function diag(NDArray $a, int $offset = 0): NDArray
    {
        return $a->diag($offset);
    }

    /**
     * Compute trace (sum of diagonal elements).
     *
     * @return Complex|float|int scalar value
     */
    function trace(NDArray $a): Complex|float|int
    {
        return $a->trace();
    }

    /**
     * Solve a linear system A * x = b.
     *
     * A must be a 2D square matrix. b can be 1D or 2D.
     *
     * @param NDArray $b Right-hand side array
     */
    function solve(NDArray $a, NDArray $b): NDArray
    {
        return $a->solve($b);
    }

    /**
     * Compute the inverse of a square matrix.
     *
     * Requires a 2D square matrix.
     */
    function inv(NDArray $a): NDArray
    {
        return $a->inv();
    }

    /**
     * Compute the determinant of a square matrix.
     *
     * Requires a 2D square matrix.
     */
    function det(NDArray $a): Complex|float|int
    {
        return $a->det();
    }

    /**
     * Compute Singular Value Decomposition (SVD).
     *
     * Decomposes matrix A into U * S * V^T where:
     * - U is an orthogonal matrix (left singular vectors)
     * - S is a diagonal matrix of singular values (returned as 1D array)
     * - V^T is an orthogonal matrix (right singular vectors transposed)
     *
     * @param bool $computeUv If true, compute U and V^T matrices. If false, only compute singular values.
     *
     * @return ($computeUv is true ? array{0: NDArray, 1: NDArray, 2: NDArray} : NDArray)
     */
    function svd(NDArray $a, bool $computeUv = true): array|NDArray
    {
        return $a->svd($computeUv);
    }

    /**
     * Compute QR decomposition.
     *
     * Decomposes matrix A into Q * R where Q is orthogonal and R is upper triangular.
     *
     * @return array{0: NDArray, 1: NDArray} [Q, R]
     */
    function qr(NDArray $a): array
    {
        return $a->qr();
    }

    /**
     * Compute eigenvalue decomposition.
     *
     * Decomposes square matrix A into eigenvalues and right eigenvectors:
     * A * v_i = lambda_i * v_i
     *
     * For real input matrices, eigenvalues and eigenvectors are complex.
     * For complex input matrices, output type matches input type.
     *
     * @return array{0: NDArray, 1: NDArray} [eigenvalues, eigenvectors]
     */
    function eig(NDArray $a): array
    {
        return $a->eig();
    }

    /**
     * Compute eigenvalues only (no eigenvectors) for a general matrix.
     *
     * For real input matrices, eigenvalues are complex.
     * For complex input matrices, output type matches input type.
     */
    function eigvals(NDArray $a): NDArray
    {
        return $a->eigvals();
    }

    /**
     * Compute eigenvalue decomposition for a Hermitian/symmetric matrix.
     *
     * Eigenvalues are always real. Eigenvectors have the same type as input.
     *
     * @param bool $upper If true, use upper triangular part. If false, use lower.
     *
     * @return array{0: NDArray, 1: NDArray} [eigenvalues, eigenvectors]
     */
    function eigh(NDArray $a, bool $upper = false): array
    {
        return $a->eigh($upper);
    }

    /**
     * Compute eigenvalues only (no eigenvectors) for a Hermitian/symmetric matrix.
     *
     * Eigenvalues are always real.
     *
     * @param bool $upper If true, use upper triangular part. If false, use lower.
     */
    function eigvalsh(NDArray $a, bool $upper = false): NDArray
    {
        return $a->eigvalsh($upper);
    }

    /**
     * Compute Cholesky decomposition.
     *
     * For a Hermitian positive-definite matrix A:
     * - upper=false (default): returns L such that A = L * L^H
     * - upper=true: returns U such that A = U^H * U
     */
    function cholesky(NDArray $a, bool $upper = false): NDArray
    {
        return $a->cholesky($upper);
    }

    /**
     * Solve a least-squares problem min ||Ax - b||_2.
     *
     * Returns [x, residuals, rank, s] where:
     * - x is the least-squares solution
     * - residuals is the sum of residuals (or null if not applicable)
     * - rank is the effective rank of A
     * - s is the singular values of A
     *
     * @return array{0: NDArray, 1: null|NDArray, 2: int, 3: NDArray}
     */
    function lstsq(NDArray $a, NDArray $b): array
    {
        return $a->lstsq($b);
    }

    /**
     * Alias for lstsq().
     *
     * @return array{0: NDArray, 1: null|NDArray, 2: int, 3: NDArray}
     */
    function least_squares(NDArray $a, NDArray $b): array
    {
        return $a->leastSquares($b);
    }

    /**
     * Compute the Moore-Penrose pseudo-inverse of a matrix.
     *
     * @param null|float $rcond Cutoff for small singular values. Default uses machine precision.
     */
    function pinv(NDArray $a, ?float $rcond = null): NDArray
    {
        return $a->pinv($rcond);
    }

    /**
     * Compute the 2-norm condition number of a matrix.
     */
    function cond(NDArray $a): Complex|float|int
    {
        return $a->cond();
    }

    /**
     * Compute the rank of a matrix using SVD.
     *
     * @param null|float $tol threshold below which SVD values are considered zero
     */
    function rank(NDArray $a, ?float $tol = null): int
    {
        return $a->rank($tol);
    }
}

namespace PhpMlKit\NDArray\Fft {
    use PhpMlKit\NDArray\NDArray;
    use PhpMlKit\NDArray\Normalization;

    /**
     * Complex discrete Fourier transform along one axis. Real inputs are promoted to complex.
     *
     * @param null|int $n Transform length along `axis` (null = current size)
     */
    function fft(NDArray $a, ?int $n = null, int $axis = -1, Normalization $norm = Normalization::Backward): NDArray
    {
        return $a->fft($n, $axis, $norm);
    }

    /**
     * Inverse complex FFT along one axis (expects complex dtype).
     */
    function ifft(NDArray $a, ?int $n = null, int $axis = -1, Normalization $norm = Normalization::Backward): NDArray
    {
        return $a->ifft($n, $axis, $norm);
    }

    /**
     * N-dimensional complex FFT. `axes` null or empty transforms all axes in order.
     *
     * @param null|array<int> $axes Axis indices (negative indices allowed)
     */
    function fftn(NDArray $a, ?array $axes = null, Normalization $norm = Normalization::Backward): NDArray
    {
        return $a->fftn($axes, $norm);
    }

    /**
     * N-dimensional inverse complex FFT.
     *
     * @param null|array<int> $axes Axis indices (negative indices allowed)
     */
    function ifftn(NDArray $a, ?array $axes = null, Normalization $norm = Normalization::Backward): NDArray
    {
        return $a->ifftn($axes, $norm);
    }

    /**
     * Real-input FFT along `axis`; result is complex with length `n//2+1` on that axis.
     */
    function rfft(NDArray $a, ?int $n = null, int $axis = -1, Normalization $norm = Normalization::Backward): NDArray
    {
        return $a->rfft($n, $axis, $norm);
    }

    /**
     * Inverse real FFT: Hermitian spectrum → real. `n` is the real length along `axis` (null = inferred).
     */
    function irfft(NDArray $a, ?int $n = null, int $axis = -1, Normalization $norm = Normalization::Backward): NDArray
    {
        return $a->irfft($n, $axis, $norm);
    }

    /**
     * 2-D FFT on the last two axes (requires `ndim >= 2`).
     */
    function fft2(NDArray $a, Normalization $norm = Normalization::Backward): NDArray
    {
        return $a->fft2($norm);
    }

    /**
     * 2-D inverse FFT on the last two axes (requires `ndim >= 2`).
     */
    function ifft2(NDArray $a, Normalization $norm = Normalization::Backward): NDArray
    {
        return $a->ifft2($norm);
    }

    /**
     * Real discrete cosine transform along one axis.
     *
     * @param int      $type DCT-I … DCT-IV (`1` … `4`). Default `2` matches SciPy `dct(..., type=2)`.
     * @param null|int $n    Length along `axis` (null = current size)
     */
    function dct(NDArray $a, int $type = 2, ?int $n = null, int $axis = -1, Normalization $norm = Normalization::Backward): NDArray
    {
        return $a->dct($type, $n, $axis, $norm);
    }

    /**
     * Inverse DCT along one axis (pairs with {@see NDArray::dct()} for the same `$type`).
     */
    function idct(NDArray $a, int $type = 2, ?int $n = null, int $axis = -1, Normalization $norm = Normalization::Backward): NDArray
    {
        return $a->idct($type, $n, $axis, $norm);
    }

    /**
     * N-dimensional DCT. `axes` null or empty applies along every axis in order.
     *
     * @param null|array<int> $axes Axis indices (negative indices allowed)
     */
    function dctn(NDArray $a, ?array $axes = null, int $type = 2, Normalization $norm = Normalization::Backward): NDArray
    {
        return $a->dctn($axes, $type, $norm);
    }

    /**
     * N-dimensional inverse DCT.
     *
     * @param null|array<int> $axes Axis indices (negative indices allowed)
     */
    function idctn(NDArray $a, ?array $axes = null, int $type = 2, Normalization $norm = Normalization::Backward): NDArray
    {
        return $a->idctn($axes, $type, $norm);
    }

    /**
     * 2-D DCT on the last two axes (requires `ndim >= 2`).
     */
    function dct2(NDArray $a, int $type = 2, Normalization $norm = Normalization::Backward): NDArray
    {
        return $a->dct2($type, $norm);
    }

    /**
     * 2-D inverse DCT on the last two axes (requires `ndim >= 2`).
     */
    function idct2(NDArray $a, int $type = 2, Normalization $norm = Normalization::Backward): NDArray
    {
        return $a->idct2($type, $norm);
    }
}

namespace PhpMlKit\NDArray\Windows {
    use PhpMlKit\NDArray\NDArray;

    /**
     * Bartlett (triangular) window — delegates to {@see NDArray::bartlett()}.
     *
     * Piecewise linear, peaks at 1.0 at the center, 0.0 at both ends for `m > 1`.
     *
     * @param int  $m        Number of samples. `m = 0` → empty array; `m = 1` → `[1.0]`.
     * @param bool $periodic if `true`, periodic variant; if `false`, symmetric (default)
     *
     * @return NDArray float64, shape `[m]`
     */
    function bartlett(int $m, bool $periodic = false): NDArray
    {
        return NDArray::bartlett($m, $periodic);
    }

    /**
     * Blackman window — delegates to {@see NDArray::blackman()}.
     *
     * Cosine-sum window; stronger sidelobes suppression than Hann/Hamming, wider main lobe.
     * For `m > 1` and `0 ≤ n ≤ m-1`: {@code w(n) = 0.42 - 0.5*cos(2πn/(m-1)) + 0.08*cos(4πn/(m-1))}.
     *
     * @param int  $m        Number of samples. `m = 0` → empty array; `m = 1` → `[1.0]`.
     * @param bool $periodic if `true`, periodic variant; if `false`, symmetric (default)
     *
     * @return NDArray float64, shape `[m]`
     */
    function blackman(int $m, bool $periodic = false): NDArray
    {
        return NDArray::blackman($m, $periodic);
    }

    /**
     * Bohman window — delegates to {@see NDArray::bohman()}.
     *
     * Let {@code x = |2n/(m-1) - 1|}. For `m > 1`: {@code w(n) = (1-x)*cos(πx) + sin(πx)/π}.
     *
     * @param int  $m        Number of samples. `m = 0` → empty array; `m = 1` → `[1.0]`.
     * @param bool $periodic if `true`, periodic variant; if `false`, symmetric (default)
     *
     * @return NDArray float64, shape `[m]`
     */
    function bohman(int $m, bool $periodic = false): NDArray
    {
        return NDArray::bohman($m, $periodic);
    }

    /**
     * Boxcar (rectangular) window — delegates to {@see NDArray::boxcar()}.
     *
     * All ones; no tapering.
     *
     * @param int  $m        Number of samples. `m = 0` → empty array; `m = 1` → `[1.0]`.
     * @param bool $periodic if `true`, periodic variant; if `false`, symmetric (default)
     *
     * @return NDArray float64, shape `[m]`
     */
    function boxcar(int $m, bool $periodic = false): NDArray
    {
        return NDArray::boxcar($m, $periodic);
    }

    /**
     * Hamming window — delegates to {@see NDArray::hamming()}.
     *
     * For `m > 1` and `0 ≤ n ≤ m-1`: {@code w(n) = 0.54 - 0.46*cos(2πn/(m-1))}.
     *
     * @param int  $m        Number of samples. `m = 0` → empty array; `m = 1` → `[1.0]`.
     * @param bool $periodic if `true`, periodic variant; if `false`, symmetric (default)
     *
     * @return NDArray float64, shape `[m]`
     */
    function hamming(int $m, bool $periodic = false): NDArray
    {
        return NDArray::hamming($m, $periodic);
    }

    /**
     * Hanning (Hann) window — delegates to {@see NDArray::hanning()}.
     *
     * For `m > 1` and `0 ≤ n ≤ m-1`: {@code w(n) = 0.5 - 0.5*cos(2πn/(m-1))}.
     *
     * @param int  $m        Number of samples. `m = 0` → empty array; `m = 1` → `[1.0]`.
     * @param bool $periodic if `true`, periodic variant; if `false`, symmetric (default)
     *
     * @return NDArray float64, shape `[m]`
     *
     * @see hann()
     */
    function hanning(int $m, bool $periodic = false): NDArray
    {
        return NDArray::hanning($m, $periodic);
    }

    /**
     * Alias of {@see hanning()} — delegates to {@see NDArray::hann()}.
     *
     * @param int  $m        Number of samples. `m = 0` → empty array; `m = 1` → `[1.0]`.
     * @param bool $periodic if `true`, periodic variant; if `false`, symmetric (default)
     *
     * @return NDArray float64, shape `[m]`
     */
    function hann(int $m, bool $periodic = false): NDArray
    {
        return NDArray::hann($m, $periodic);
    }

    /**
     * Kaiser window — delegates to {@see NDArray::kaiser()}.
     *
     * Larger `beta` improves sidelobe attenuation at the cost of a wider main lobe.
     *
     * @param int   $m        Number of samples. `m = 0` → empty array; `m = 1` → `[1.0]`.
     * @param float $beta     shape parameter (main lobe vs sidelobe trade-off)
     * @param bool  $periodic if `true`, periodic variant; if `false`, symmetric (default)
     *
     * @return NDArray float64, shape `[m]`
     */
    function kaiser(int $m, float $beta, bool $periodic = false): NDArray
    {
        return NDArray::kaiser($m, $beta, $periodic);
    }

    /**
     * Lanczos window — delegates to {@see NDArray::lanczos()}.
     *
     * Let {@code x = 2n/(m-1) - 1}. For `m > 1`: {@code w(n) = sinc(x)} with {@code sinc(0)=1},
     * {@code sinc(x) = sin(πx)/(πx)} for {@code x ≠ 0}.
     *
     * @param int  $m        Number of samples. `m = 0` → empty array; `m = 1` → `[1.0]`.
     * @param bool $periodic if `true`, periodic variant; if `false`, symmetric (default)
     *
     * @return NDArray float64, shape `[m]`
     */
    function lanczos(int $m, bool $periodic = false): NDArray
    {
        return NDArray::lanczos($m, $periodic);
    }

    /**
     * Triangular window — delegates to {@see NDArray::triang()}.
     *
     * Similar to Bartlett; endpoints are not forced to zero the same way.
     *
     * @param int  $m        Number of samples. `m = 0` → empty array; `m = 1` → `[1.0]`.
     * @param bool $periodic if `true`, periodic variant; if `false`, symmetric (default)
     *
     * @return NDArray float64, shape `[m]`
     */
    function triang(int $m, bool $periodic = false): NDArray
    {
        return NDArray::triang($m, $periodic);
    }
}
