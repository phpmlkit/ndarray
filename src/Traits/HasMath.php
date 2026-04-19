<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Traits;

use PhpMlKit\NDArray\Complex;
use PhpMlKit\NDArray\NDArray;

/**
 * Mathematical operations trait for NDArray.
 *
 * Provides element-wise arithmetic operations with broadcasting support
 * for both array-array and array-scalar operations.
 */
trait HasMath
{
    /**
     * Add another array or scalar to this array.
     *
     * @param Complex|float|int|NDArray $other Array or scalar to add
     *
     * @return NDArray New array with result
     */
    public function add(Complex|float|int|NDArray $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_add', $other);
        }

        return $this->unaryOp('ndarray_add_scalar', ...$this->scalarToBuffer($other));
    }

    /**
     * Subtract another array or scalar from this array.
     *
     * @param Complex|float|int|NDArray $other Array or scalar to subtract
     *
     * @return NDArray New array with result
     */
    public function subtract(Complex|float|int|NDArray $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_sub', $other);
        }

        return $this->unaryOp('ndarray_sub_scalar', ...$this->scalarToBuffer($other));
    }

    /**
     * Multiply this array by another array or scalar.
     *
     * @param Complex|float|int|NDArray $other Array or scalar to multiply by
     *
     * @return NDArray New array with result
     */
    public function multiply(Complex|float|int|NDArray $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_mul', $other);
        }

        return $this->unaryOp('ndarray_mul_scalar', ...$this->scalarToBuffer($other));
    }

    /**
     * Divide this array by another array or scalar.
     *
     * @param Complex|float|int|NDArray $other Array or scalar to divide by
     *
     * @return NDArray New array with result
     */
    public function divide(Complex|float|int|NDArray $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_div', $other);
        }

        return $this->unaryOp('ndarray_div_scalar', ...$this->scalarToBuffer($other));
    }

    /**
     * Compute remainder (modulo) with another array or scalar.
     *
     * @param Complex|float|int|NDArray $other Array or scalar
     *
     * @return NDArray New array with result
     */
    public function rem(Complex|float|int|NDArray $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_rem', $other);
        }

        return $this->unaryOp('ndarray_rem_scalar', ...$this->scalarToBuffer($other));
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
    public function mod(Complex|float|int|NDArray $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_rem', $other);
        }

        return $this->unaryOp('ndarray_rem_scalar', ...$this->scalarToBuffer($other));
    }

    /**
     * Compute absolute value element-wise.
     */
    public function abs(): NDArray
    {
        return $this->unaryOp('ndarray_abs');
    }

    /**
     * Compute negation element-wise (-$a).
     * Not supported for unsigned integers or bool.
     */
    public function negative(): NDArray
    {
        return $this->unaryOp('ndarray_neg');
    }

    /**
     * Extract real part element-wise.
     *
     * For complex arrays, returns the real component as float.
     * For real arrays, returns a copy with the same dtype.
     */
    public function real(): NDArray
    {
        return $this->unaryOp('ndarray_real');
    }

    /**
     * Extract imaginary part element-wise.
     *
     * For complex arrays, returns the imaginary component as float.
     * For real arrays, returns zeros with the same dtype.
     */
    public function imag(): NDArray
    {
        return $this->unaryOp('ndarray_imag');
    }

    /**
     * Compute complex conjugate element-wise.
     *
     * For complex arrays, negates the imaginary part.
     * For real arrays, returns a copy with the same dtype.
     */
    public function conjugate(): NDArray
    {
        return $this->unaryOp('ndarray_conjugate');
    }

    /**
     * Alias for conjugate().
     */
    public function conj(): NDArray
    {
        return $this->conjugate();
    }

    /**
     * Returns bool array: true where element has non-zero imaginary part.
     *
     * For complex arrays, checks if imag ≠ 0.
     * For real arrays, always returns false.
     */
    public function iscomplex(): NDArray
    {
        return $this->unaryOp('ndarray_iscomplex');
    }

    /**
     * Returns bool array: true where element has zero imaginary part.
     *
     * For complex arrays, checks if imag = 0.
     * For real arrays, always returns true.
     */
    public function isreal(): NDArray
    {
        return $this->unaryOp('ndarray_isreal');
    }

    /**
     * Compute phase angle element-wise.
     *
     * Returns the counterclockwise angle from the positive real axis.
     * Always returns Float64 regardless of input dtype.
     *
     * @param bool $deg if true, returns angle in degrees; otherwise radians
     */
    public function angle(bool $deg = false): NDArray
    {
        return $this->unaryOp('ndarray_angle', $deg ? 1 : 0);
    }

    /**
     * Compute square root element-wise.
     */
    public function sqrt(): NDArray
    {
        return $this->unaryOp('ndarray_sqrt');
    }

    /**
     * Compute exponential element-wise.
     */
    public function exp(): NDArray
    {
        return $this->unaryOp('ndarray_exp');
    }

    /**
     * Compute natural logarithm element-wise.
     */
    public function log(): NDArray
    {
        return $this->unaryOp('ndarray_log');
    }

    /**
     * Compute natural logarithm element-wise.
     *
     * Alias for log().
     */
    public function ln(): NDArray
    {
        return $this->unaryOp('ndarray_ln');
    }

    /**
     * Compute sine element-wise.
     */
    public function sin(): NDArray
    {
        return $this->unaryOp('ndarray_sin');
    }

    /**
     * Compute cosine element-wise.
     */
    public function cos(): NDArray
    {
        return $this->unaryOp('ndarray_cos');
    }

    /**
     * Compute tangent element-wise.
     */
    public function tan(): NDArray
    {
        return $this->unaryOp('ndarray_tan');
    }

    /**
     * Compute hyperbolic sine element-wise.
     */
    public function sinh(): NDArray
    {
        return $this->unaryOp('ndarray_sinh');
    }

    /**
     * Compute hyperbolic cosine element-wise.
     */
    public function cosh(): NDArray
    {
        return $this->unaryOp('ndarray_cosh');
    }

    /**
     * Compute hyperbolic tangent element-wise.
     */
    public function tanh(): NDArray
    {
        return $this->unaryOp('ndarray_tanh');
    }

    /**
     * Compute arc sine element-wise.
     */
    public function asin(): NDArray
    {
        return $this->unaryOp('ndarray_asin');
    }

    /**
     * Compute arc cosine element-wise.
     */
    public function acos(): NDArray
    {
        return $this->unaryOp('ndarray_acos');
    }

    /**
     * Compute arc tangent element-wise.
     */
    public function atan(): NDArray
    {
        return $this->unaryOp('ndarray_atan');
    }

    /**
     * Compute cube root element-wise.
     */
    public function cbrt(): NDArray
    {
        return $this->unaryOp('ndarray_cbrt');
    }

    /**
     * Compute ceiling element-wise.
     */
    public function ceil(): NDArray
    {
        return $this->unaryOp('ndarray_ceil');
    }

    /**
     * Compute base-2 exponential (2^x) element-wise.
     */
    public function exp2(): NDArray
    {
        return $this->unaryOp('ndarray_exp2');
    }

    /**
     * Compute floor element-wise.
     */
    public function floor(): NDArray
    {
        return $this->unaryOp('ndarray_floor');
    }

    /**
     * Compute base-2 logarithm element-wise.
     */
    public function log2(): NDArray
    {
        return $this->unaryOp('ndarray_log2');
    }

    /**
     * Compute base-10 logarithm element-wise.
     */
    public function log10(): NDArray
    {
        return $this->unaryOp('ndarray_log10');
    }

    /**
     * Compute x^2 (square) element-wise.
     */
    public function pow2(): NDArray
    {
        return $this->unaryOp('ndarray_pow2');
    }

    /**
     * Compute round element-wise.
     */
    public function round(): NDArray
    {
        return $this->unaryOp('ndarray_round');
    }

    /**
     * Compute signum element-wise.
     */
    public function signum(): NDArray
    {
        return $this->unaryOp('ndarray_signum');
    }

    /**
     * Compute reciprocal (1/x) element-wise.
     */
    public function recip(): NDArray
    {
        return $this->unaryOp('ndarray_recip');
    }

    /**
     * Compute ln(1+x) element-wise.
     *
     * More accurate than log(1+x) for small x.
     */
    public function ln1p(): NDArray
    {
        return $this->unaryOp('ndarray_ln_1p');
    }

    /**
     * Convert radians to degrees element-wise.
     */
    public function toDegrees(): NDArray
    {
        return $this->unaryOp('ndarray_to_degrees');
    }

    /**
     * Convert degrees to radians element-wise.
     */
    public function toRadians(): NDArray
    {
        return $this->unaryOp('ndarray_to_radians');
    }

    /**
     * Compute x^n where n is an integer, element-wise.
     *
     * Generally faster than pow() for integer exponents.
     *
     * @param int $exp Integer exponent
     */
    public function powi(int $exp): NDArray
    {
        return $this->unaryOp('ndarray_powi', $exp);
    }

    /**
     * Compute x^y where y is a float, element-wise.
     *
     * @param float $exp Float exponent
     */
    public function powf(float $exp): NDArray
    {
        return $this->unaryOp('ndarray_powf', $exp);
    }

    /**
     * Compute hypotenuse element-wise.
     *
     * @param float $other Scalar value
     */
    public function hypot(float $other): NDArray
    {
        return $this->unaryOp('ndarray_hypot', $other);
    }

    /**
     * Bitwise AND with another array or scalar.
     *
     * @param int|NDArray $other Array or scalar
     *
     * @return NDArray New array with result
     */
    public function bitand(int|NDArray $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_bitand', $other);
        }

        return $this->unaryOp('ndarray_bitand_scalar', ...$this->scalarToBuffer($other));
    }

    /**
     * Bitwise OR with another array or scalar.
     *
     * @param int|NDArray $other Array or scalar
     *
     * @return NDArray New array with result
     */
    public function bitor(int|NDArray $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_bitor', $other);
        }

        return $this->unaryOp('ndarray_bitor_scalar', ...$this->scalarToBuffer($other));
    }

    /**
     * Bitwise XOR with another array or scalar.
     *
     * @param int|NDArray $other Array or scalar
     *
     * @return NDArray New array with result
     */
    public function bitxor(int|NDArray $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_bitxor', $other);
        }

        return $this->unaryOp('ndarray_bitxor_scalar', ...$this->scalarToBuffer($other));
    }

    /**
     * Left shift by another array or scalar.
     *
     * @param int|NDArray $other Array or scalar (number of bits)
     *
     * @return NDArray New array with result
     */
    public function leftShift(int|NDArray $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_left_shift', $other);
        }

        return $this->unaryOp('ndarray_left_shift_scalar', ...$this->scalarToBuffer($other));
    }

    /**
     * Right shift by another array or scalar.
     *
     * @param int|NDArray $other Array or scalar (number of bits)
     *
     * @return NDArray New array with result
     */
    public function rightShift(int|NDArray $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_right_shift', $other);
        }

        return $this->unaryOp('ndarray_right_shift_scalar', ...$this->scalarToBuffer($other));
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
    public function clamp(float|int $min, float|int $max): NDArray
    {
        if ($min > $max) {
            throw new \InvalidArgumentException('Clamp requires min <= max');
        }

        return $this->unaryOp('ndarray_clamp', $min, $max);
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
    public function clip(float|int $min, float|int $max): NDArray
    {
        return $this->clamp($min, $max);
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
    public function minimum(NDArray $other): NDArray
    {
        return $this->binaryOp('ndarray_minimum', $other);
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
    public function maximum(NDArray $other): NDArray
    {
        return $this->binaryOp('ndarray_maximum', $other);
    }

    /**
     * Compute sigmoid element-wise: 1 / (1 + exp(-x)).
     */
    public function sigmoid(): NDArray
    {
        return $this->unaryOp('ndarray_sigmoid');
    }

    /**
     * Compute softmax along axis: exp(x - max) / sum(exp(x - max)).
     *
     * Numerically stable. Default axis -1 (last axis) for typical logits.
     *
     * @param int $axis Axis along which to compute softmax
     */
    public function softmax(int $axis = -1): NDArray
    {
        return $this->unaryOp('ndarray_softmax', $axis);
    }
}
