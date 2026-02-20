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
     * @param NDArray|float|int $other Array or scalar to add
     * @return NDArray New array with result
     */
    public function add(NDArray|float|int $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_add', $other);
        }
        return $this->unaryOp('ndarray_add_scalar', $other);
    }

    /**
     * Subtract another array or scalar from this array.
     *
     * @param NDArray|float|int $other Array or scalar to subtract
     * @return NDArray New array with result
     */
    public function subtract(NDArray|float|int $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_sub', $other);
        }
        return $this->unaryOp('ndarray_sub_scalar', $other);
    }

    /**
     * Multiply this array by another array or scalar.
     *
     * @param NDArray|float|int $other Array or scalar to multiply by
     * @return NDArray New array with result
     */
    public function multiply(NDArray|float|int $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_mul', $other);
        }
        return $this->unaryOp('ndarray_mul_scalar', $other);
    }

    /**
     * Divide this array by another array or scalar.
     *
     * @param NDArray|float|int $other Array or scalar to divide by
     * @return NDArray New array with result
     */
    public function divide(NDArray|float|int $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_div', $other);
        }
        return $this->unaryOp('ndarray_div_scalar', $other);
    }

    /**
     * Compute remainder (modulo) with another array or scalar.
     *
     * @param NDArray|float|int $other Array or scalar
     * @return NDArray New array with result
     */
    public function rem(NDArray|float|int $other): NDArray
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
     * @param NDArray|float|int $other Array or scalar
     * @return NDArray New array with result
     */
    public function mod(NDArray|float|int $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_rem', $other);
        }
        return $this->unaryOp('ndarray_rem_scalar', $other);
    }

    /**
     * Compute absolute value element-wise.
     *
     * @return NDArray
     */
    public function abs(): NDArray
    {
        return $this->unaryOp('ndarray_abs');
    }

    /**
     * Compute negation element-wise (-$a).
     * Not supported for unsigned integers or bool.
     *
     * @return NDArray
     */
    public function negative(): NDArray
    {
        return $this->unaryOp('ndarray_neg');
    }

    /**
     * Compute square root element-wise.
     *
     * @return NDArray
     */
    public function sqrt(): NDArray
    {
        return $this->unaryOp('ndarray_sqrt');
    }

    /**
     * Compute exponential element-wise.
     *
     * @return NDArray
     */
    public function exp(): NDArray
    {
        return $this->unaryOp('ndarray_exp');
    }

    /**
     * Compute natural logarithm element-wise.
     *
     * @return NDArray
     */
    public function log(): NDArray
    {
        return $this->unaryOp('ndarray_log');
    }

    /**
     * Compute natural logarithm element-wise.
     *
     * Alias for log().
     *
     * @return NDArray
     */
    public function ln(): NDArray
    {
        return $this->unaryOp('ndarray_ln');
    }

    /**
     * Compute sine element-wise.
     *
     * @return NDArray
     */
    public function sin(): NDArray
    {
        return $this->unaryOp('ndarray_sin');
    }

    /**
     * Compute cosine element-wise.
     *
     * @return NDArray
     */
    public function cos(): NDArray
    {
        return $this->unaryOp('ndarray_cos');
    }

    /**
     * Compute tangent element-wise.
     *
     * @return NDArray
     */
    public function tan(): NDArray
    {
        return $this->unaryOp('ndarray_tan');
    }

    /**
     * Compute hyperbolic sine element-wise.
     *
     * @return NDArray
     */
    public function sinh(): NDArray
    {
        return $this->unaryOp('ndarray_sinh');
    }

    /**
     * Compute hyperbolic cosine element-wise.
     *
     * @return NDArray
     */
    public function cosh(): NDArray
    {
        return $this->unaryOp('ndarray_cosh');
    }

    /**
     * Compute hyperbolic tangent element-wise.
     *
     * @return NDArray
     */
    public function tanh(): NDArray
    {
        return $this->unaryOp('ndarray_tanh');
    }



    /**
     * Compute arc sine element-wise.
     *
     * @return NDArray
     */
    public function asin(): NDArray
    {
        return $this->unaryOp('ndarray_asin');
    }

    /**
     * Compute arc cosine element-wise.
     *
     * @return NDArray
     */
    public function acos(): NDArray
    {
        return $this->unaryOp('ndarray_acos');
    }

    /**
     * Compute arc tangent element-wise.
     *
     * @return NDArray
     */
    public function atan(): NDArray
    {
        return $this->unaryOp('ndarray_atan');
    }

    /**
     * Compute cube root element-wise.
     *
     * @return NDArray
     */
    public function cbrt(): NDArray
    {
        return $this->unaryOp('ndarray_cbrt');
    }

    /**
     * Compute ceiling element-wise.
     *
     * @return NDArray
     */
    public function ceil(): NDArray
    {
        return $this->unaryOp('ndarray_ceil');
    }

    /**
     * Compute base-2 exponential (2^x) element-wise.
     *
     * @return NDArray
     */
    public function exp2(): NDArray
    {
        return $this->unaryOp('ndarray_exp2');
    }

    /**
     * Compute floor element-wise.
     *
     * @return NDArray
     */
    public function floor(): NDArray
    {
        return $this->unaryOp('ndarray_floor');
    }

    /**
     * Compute base-2 logarithm element-wise.
     *
     * @return NDArray
     */
    public function log2(): NDArray
    {
        return $this->unaryOp('ndarray_log2');
    }

    /**
     * Compute base-10 logarithm element-wise.
     *
     * @return NDArray
     */
    public function log10(): NDArray
    {
        return $this->unaryOp('ndarray_log10');
    }

    /**
     * Compute x^2 (square) element-wise.
     *
     * @return NDArray
     */
    public function pow2(): NDArray
    {
        return $this->unaryOp('ndarray_pow2');
    }

    /**
     * Compute round element-wise.
     *
     * @return NDArray
     */
    public function round(): NDArray
    {
        return $this->unaryOp('ndarray_round');
    }

    /**
     * Compute signum element-wise.
     *
     * @return NDArray
     */
    public function signum(): NDArray
    {
        return $this->unaryOp('ndarray_signum');
    }

    /**
     * Compute reciprocal (1/x) element-wise.
     *
     * @return NDArray
     */
    public function recip(): NDArray
    {
        return $this->unaryOp('ndarray_recip');
    }

    /**
     * Compute ln(1+x) element-wise.
     *
     * More accurate than log(1+x) for small x.
     *
     * @return NDArray
     */
    public function ln1p(): NDArray
    {
        return $this->unaryOp('ndarray_ln_1p');
    }

    /**
     * Convert radians to degrees element-wise.
     *
     * @return NDArray
     */
    public function toDegrees(): NDArray
    {
        return $this->unaryOp('ndarray_to_degrees');
    }

    /**
     * Convert degrees to radians element-wise.
     *
     * @return NDArray
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
     * @return NDArray
     */
    public function powi(int $exp): NDArray
    {
        return $this->unaryOp('ndarray_powi', $exp);
    }

    /**
     * Compute x^y where y is a float, element-wise.
     *
     * @param float $exp Float exponent
     * @return NDArray
     */
    public function powf(float $exp): NDArray
    {
        return $this->unaryOp('ndarray_powf', $exp);
    }

    /**
     * Compute hypotenuse element-wise.
     *
     * @param float $other Scalar value
     * @return NDArray
     */
    public function hypot(float $other): NDArray
    {
        return $this->unaryOp('ndarray_hypot', $other);
    }

    /**
     * Bitwise AND with another array or scalar.
     *
     * @param NDArray|int $other Array or scalar
     * @return NDArray New array with result
     */
    public function bitand(NDArray|int $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_bitand', $other);
        }
        return $this->unaryOp('ndarray_bitand_scalar', $other);
    }

    /**
     * Bitwise OR with another array or scalar.
     *
     * @param NDArray|int $other Array or scalar
     * @return NDArray New array with result
     */
    public function bitor(NDArray|int $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_bitor', $other);
        }
        return $this->unaryOp('ndarray_bitor_scalar', $other);
    }

    /**
     * Bitwise XOR with another array or scalar.
     *
     * @param NDArray|int $other Array or scalar
     * @return NDArray New array with result
     */
    public function bitxor(NDArray|int $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_bitxor', $other);
        }
        return $this->unaryOp('ndarray_bitxor_scalar', $other);
    }

    /**
     * Left shift by another array or scalar.
     *
     * @param NDArray|int $other Array or scalar (number of bits)
     * @return NDArray New array with result
     */
    public function leftShift(NDArray|int $other): NDArray
    {
        if ($other instanceof NDArray) {
            return $this->binaryOp('ndarray_left_shift', $other);
        }
        return $this->unaryOp('ndarray_left_shift_scalar', $other);
    }

    /**
     * Right shift by another array or scalar.
     *
     * @param NDArray|int $other Array or scalar (number of bits)
     * @return NDArray New array with result
     */
    public function rightShift(NDArray|int $other): NDArray
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
     * @return NDArray
     * @throws \InvalidArgumentException If min > max
     */
    public function clamp(float|int $min, float|int $max): NDArray
    {
        if ($min > $max) {
            throw new \InvalidArgumentException("Clamp requires min <= max");
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
     * @return NDArray
     * @throws \InvalidArgumentException If min > max
     */
    public function clip(float|int $min, float|int $max): NDArray
    {
        return $this->clamp($min, $max);
    }

     /**
     * Compute sigmoid element-wise: 1 / (1 + exp(-x)).
     *
     * @return NDArray
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
     * @return NDArray
     */
    public function softmax(int $axis = -1): NDArray
    {
        return $this->unaryOp('ndarray_softmax', $axis);
    }
}
