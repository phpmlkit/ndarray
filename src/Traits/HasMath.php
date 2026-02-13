<?php

declare(strict_types=1);

namespace NDArray\Traits;

use NDArray\DType;
use NDArray\NDArray;
use NDArray\FFI\Lib;

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
        return $this->scalarOp('ndarray_add_scalar', (float) $other);
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
        return $this->scalarOp('ndarray_sub_scalar', (float) $other);
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
        return $this->scalarOp('ndarray_mul_scalar', (float) $other);
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
        return $this->scalarOp('ndarray_div_scalar', (float) $other);
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
        return $this->scalarOp('ndarray_powi', $exp);
    }

    /**
     * Compute x^y where y is a float, element-wise.
     *
     * @param float $exp Float exponent
     * @return NDArray
     */
    public function powf(float $exp): NDArray
    {
        return $this->scalarOp('ndarray_powf', $exp);
    }

    /**
     * Compute hypotenuse sqrt(a^2 + b^2) element-wise.
     *
     * @param NDArray $other Other array
     * @return NDArray
     */
    public function hypot(NDArray $other): NDArray
    {
        return $this->binaryOp('ndarray_hypot', $other);
    }

    /**
     * Perform a binary operation with another array.
     *
     * @param string $funcName FFI function name
     * @param NDArray $other Other array
     * @return NDArray
     */
    private function binaryOp(string $funcName, NDArray $other): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new("struct NdArrayHandle*");

        // Get view metadata for both arrays
        $aShape = Lib::createShapeArray($this->shape);
        $aStrides = Lib::createShapeArray($this->strides);
        $bShape = Lib::createShapeArray($other->shape);
        $bStrides = Lib::createShapeArray($other->strides);

        // Use the maximum ndim for broadcasting
        $ndim = max(count($this->shape), count($other->shape));

        $status = $ffi->$funcName(
            $this->handle,
            $this->offset,
            $aShape,
            $aStrides,
            $other->handle,
            $other->offset,
            $bShape,
            $bStrides,
            $ndim,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        // Determine output shape based on broadcasting rules
        $outShape = $this->broadcastShapes($this->shape, $other->shape);

        // Determine output dtype via type promotion
        $outDtype = DType::promote($this->dtype, $other->dtype);

        return new NDArray($outHandle, $outShape, $outDtype);
    }

    /**
     * Perform a scalar operation.
     *
     * @param string $funcName FFI function name
     * @param float $scalar Scalar value
     * @return NDArray
     */
    private function scalarOp(string $funcName, float $scalar): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new("struct NdArrayHandle*");

        $aShape = Lib::createShapeArray($this->shape);
        $aStrides = Lib::createShapeArray($this->strides);

        $status = $ffi->$funcName(
            $this->handle,
            $this->offset,
            $aShape,
            $aStrides,
            count($this->shape),
            $scalar,
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new NDArray($outHandle, $this->shape, $this->dtype);
    }

    /**
     * Perform a unary operation.
     *
     * @param string $funcName FFI function name
     * @return NDArray
     */
    private function unaryOp(string $funcName): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new("struct NdArrayHandle*");

        $aShape = Lib::createShapeArray($this->shape);
        $aStrides = Lib::createShapeArray($this->strides);

        $status = $ffi->$funcName(
            $this->handle,
            $this->offset,
            $aShape,
            $aStrides,
            count($this->shape),
            Lib::addr($outHandle)
        );

        Lib::checkStatus($status);

        return new NDArray($outHandle, $this->shape, $this->dtype);
    }

    /**
     * Compute the broadcasted shape of two arrays.
     *
     * @param array<int> $shapeA
     * @param array<int> $shapeB
     * @return array<int>
     */
    private function broadcastShapes(array $shapeA, array $shapeB): array
    {
        $ndimA = count($shapeA);
        $ndimB = count($shapeB);
        $maxNdim = max($ndimA, $ndimB);

        // Pad shorter shape with 1s on the left
        $paddedA = array_merge(array_fill(0, $maxNdim - $ndimA, 1), $shapeA);
        $paddedB = array_merge(array_fill(0, $maxNdim - $ndimB, 1), $shapeB);

        $result = [];
        for ($i = 0; $i < $maxNdim; $i++) {
            $dimA = $paddedA[$i];
            $dimB = $paddedB[$i];

            if ($dimA === 1) {
                $result[] = $dimB;
            } elseif ($dimB === 1) {
                $result[] = $dimA;
            } elseif ($dimA === $dimB) {
                $result[] = $dimA;
            } else {
                throw new \NDArray\Exceptions\ShapeException(
                    "Cannot broadcast shapes " . json_encode($shapeA) . " and " . json_encode($shapeB)
                );
            }
        }

        return $result;
    }

    // =========================================================================
    // Static Methods
    // =========================================================================

    /**
     * Add two arrays element-wise.
     *
     * @param NDArray $a
     * @param NDArray|float|int $b
     * @return NDArray
     */
    public static function addArrays(NDArray $a, NDArray|float|int $b): NDArray
    {
        return $a->add($b);
    }

    /**
     * Subtract two arrays element-wise.
     *
     * @param NDArray $a
     * @param NDArray|float|int $b
     * @return NDArray
     */
    public static function subtractArrays(NDArray $a, NDArray|float|int $b): NDArray
    {
        return $a->subtract($b);
    }

    /**
     * Multiply two arrays element-wise.
     *
     * @param NDArray $a
     * @param NDArray|float|int $b
     * @return NDArray
     */
    public static function multiplyArrays(NDArray $a, NDArray|float|int $b): NDArray
    {
        return $a->multiply($b);
    }

    /**
     * Divide two arrays element-wise.
     *
     * @param NDArray $a
     * @param NDArray|float|int $b
     * @return NDArray
     */
    public static function divideArrays(NDArray $a, NDArray|float|int $b): NDArray
    {
        return $a->divide($b);
    }
}
