<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Traits;

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
     * @param float|int|NDArray $other Array or scalar to add
     *
     * @return NDArray New array with result
     */
    public function add(float|int|NDArray $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_add', $other);
        }

        return $this->unaryOp('ndarray_add_scalar', $other);
    }

    /**
     * Subtract another array or scalar from this array.
     *
     * @param float|int|NDArray $other Array or scalar to subtract
     *
     * @return NDArray New array with result
     */
    public function subtract(float|int|NDArray $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_sub', $other);
        }

        return $this->unaryOp('ndarray_sub_scalar', $other);
    }

    /**
     * Multiply this array by another array or scalar.
     *
     * @param float|int|NDArray $other Array or scalar to multiply by
     *
     * @return NDArray New array with result
     */
    public function multiply(float|int|NDArray $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_mul', $other);
        }

        return $this->unaryOp('ndarray_mul_scalar', $other);
    }

    /**
     * Divide this array by another array or scalar.
     *
     * @param float|int|NDArray $other Array or scalar to divide by
     *
     * @return NDArray New array with result
     */
    public function divide(float|int|NDArray $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_div', $other);
        }

        return $this->unaryOp('ndarray_div_scalar', $other);
    }

    /**
     * Compute remainder (modulo) with another array or scalar.
     *
     * @param float|int|NDArray $other Array or scalar
     *
     * @return NDArray New array with result
     */
    public function rem(float|int|NDArray $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_rem', $other);
        }

        return $this->unaryOp('ndarray_rem_scalar', $other);
    }

    /**
     * Compute modulo with another array or scalar.
     *
     * Alias for rem().
     *
     * @param float|int|NDArray $other Array or scalar
     *
     * @return NDArray New array with result
     */
    public function mod(float|int|NDArray $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_rem', $other);
        }

        return $this->unaryOp('ndarray_rem_scalar', $other);
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

        return $this->unaryOp('ndarray_bitand_scalar', $other);
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

        return $this->unaryOp('ndarray_bitor_scalar', $other);
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

        return $this->unaryOp('ndarray_bitxor_scalar', $other);
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

        return $this->unaryOp('ndarray_left_shift_scalar', $other);
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

        return $this->unaryOp('ndarray_right_shift_scalar', $other);
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
