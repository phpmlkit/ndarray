<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\FFI;

use Codewithkyrian\PlatformPackageInstaller\Platform;
use FFI\CData;
use PhpMlKit\NDArray\Exceptions\AllocationException;
use PhpMlKit\NDArray\Exceptions\DTypeException;
use PhpMlKit\NDArray\Exceptions\IndexException;
use PhpMlKit\NDArray\Exceptions\MathException;
use PhpMlKit\NDArray\Exceptions\NDArrayException;
use PhpMlKit\NDArray\Exceptions\PanicException;
use PhpMlKit\NDArray\Exceptions\ShapeException;

/**
 * Singleton wrapper for the Rust NDArray FFI library.
 *
 * All C-level ndarray_* functions are declared via @method annotations
 * and forwarded to the underlying \FFI instance through __call().
 *
 * @method int   ndarray_get_last_error(CData $buf, int $len)
 * @method int   ndarray_to_string(CData $handle, CData $meta, CData $buf, int $buf_size, int $threshold, int $edgeitems, int $precision)
 * @method int   ndarray_create(CData $data, int $len, CData $shape, int $ndim, int $dtype, CData $out_handle)
 * @method int   ndarray_copy(CData $handle, CData $meta, CData $out_handle)
 * @method int   ndarray_free(CData $handle)
 * @method int   ndarray_zeros(CData $shape, int $ndim, int $dtype, CData $out_handle)
 * @method int   ndarray_ones(CData $shape, int $ndim, int $dtype, CData $out_handle)
 * @method int   ndarray_full(CData $shape, int $ndim, CData $value, int $dtype, CData $out_handle)
 * @method int   ndarray_eye(int $n, int $m, int $k, int $dtype, CData $out_handle)
 * @method int   ndarray_arange(float $start, float $stop, float $step, int $dtype, CData $out_handle)
 * @method int   ndarray_linspace(float $start, float $stop, int $num, bool $endpoint, int $dtype, CData $out_handle)
 * @method int   ndarray_logspace(float $start, float $stop, int $num, float $base, int $dtype, CData $out_handle)
 * @method int   ndarray_geomspace(float $start, float $stop, int $num, int $dtype, CData $out_handle)
 * @method int   ndarray_random(CData $shape, int $ndim, int $dtype, bool $has_seed, int $seed, CData $out_handle)
 * @method int   ndarray_random_int(int $low, int $high, CData $shape, int $ndim, int $dtype, bool $has_seed, int $seed, CData $out_handle)
 * @method int   ndarray_randn(CData $shape, int $ndim, int $dtype, bool $has_seed, int $seed, CData $out_handle)
 * @method int   ndarray_normal(float $mean, float $std, CData $shape, int $ndim, int $dtype, bool $has_seed, int $seed, CData $out_handle)
 * @method int   ndarray_uniform(float $low, float $high, CData $shape, int $ndim, int $dtype, bool $has_seed, int $seed, CData $out_handle)
 * @method int   ndarray_get_element(CData $handle, int $flat_index, CData $out_value)
 * @method int   ndarray_set_element(CData $handle, int $flat_index, CData $value)
 * @method int   ndarray_as_scalar(CData $handle, CData $meta, CData $out_value)
 * @method int   ndarray_get_data(CData $handle, CData $meta, int $start, int $len, CData $out_data, CData $out_len)
 * @method int   ndarray_take(CData $handle, CData $meta, CData $indices_handle, CData $indices_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_take_axis(CData $handle, CData $meta, CData $indices_handle, CData $indices_meta, int $axis, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_take_along_axis(CData $handle, CData $meta, CData $indices_handle, CData $indices_meta, int $axis, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_put(CData $handle, CData $meta, CData $indices_handle, CData $indices_meta, CData $values, int $values_len, float $scalar_value, bool $has_scalar, CData $out_handle)
 * @method int   ndarray_put_along_axis(CData $handle, CData $meta, CData $indices_handle, CData $indices_meta, int $axis, CData $values, int $values_len, float $scalar_value, bool $has_scalar, CData $out_handle)
 * @method int   ndarray_where(CData $cond_handle, CData $cond_meta, CData $x_handle, CData $x_meta, CData $y_handle, CData $y_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_scatter_add_flat(CData $handle, CData $meta, CData $indices_handle, CData $indices_meta, CData $updates, int $updates_len, float $scalar_update, bool $has_scalar, CData $out_handle)
 * @method int   ndarray_fill(CData $handle, CData $meta, CData $value)
 * @method int   ndarray_assign(CData $dst, CData $dst_meta, CData $src, CData $src_meta)
 * @method int   ndarray_add(CData $a, CData $a_meta, CData $b, CData $b_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_add_scalar(CData $a, CData $a_meta, CData $scalar, int $scalar_dtype, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_sub_scalar(CData $a, CData $a_meta, CData $scalar, int $scalar_dtype, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_mul_scalar(CData $a, CData $a_meta, CData $scalar, int $scalar_dtype, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_div_scalar(CData $a, CData $a_meta, CData $scalar, int $scalar_dtype, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_rem_scalar(CData $a, CData $a_meta, CData $scalar, int $scalar_dtype, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_hypot(CData $a, CData $a_meta, float $b, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_minimum(CData $a, CData $a_meta, CData $b, CData $b_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_maximum(CData $a, CData $a_meta, CData $b, CData $b_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_maximum_scalar(CData $a, CData $a_meta, float $scalar, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_minimum_scalar(CData $a, CData $a_meta, float $scalar, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_eq(CData $a, CData $a_meta, CData $b, CData $b_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_eq_scalar(CData $a, CData $a_meta, CData $scalar, int $scalar_dtype, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_ne_scalar(CData $a, CData $a_meta, CData $scalar, int $scalar_dtype, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_gt_scalar(CData $a, CData $a_meta, CData $scalar, int $scalar_dtype, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_gte_scalar(CData $a, CData $a_meta, CData $scalar, int $scalar_dtype, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_lt_scalar(CData $a, CData $a_meta, CData $scalar, int $scalar_dtype, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_lte_scalar(CData $a, CData $a_meta, CData $scalar, int $scalar_dtype, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_bitand(CData $a, CData $a_meta, CData $b, CData $b_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_bitand_scalar(CData $a, CData $a_meta, CData $scalar, int $scalar_dtype, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_bitor_scalar(CData $a, CData $a_meta, CData $scalar, int $scalar_dtype, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_bitxor_scalar(CData $a, CData $a_meta, CData $scalar, int $scalar_dtype, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_left_shift_scalar(CData $a, CData $a_meta, CData $scalar, int $scalar_dtype, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_right_shift_scalar(CData $a, CData $a_meta, CData $scalar, int $scalar_dtype, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_logical_and(CData $a, CData $a_meta, CData $b, CData $b_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_logical_or(CData $a, CData $a_meta, CData $b, CData $b_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_logical_not(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_logical_xor(CData $a, CData $a_meta, CData $b, CData $b_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_abs(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_sqrt(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_exp(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_log(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_ln(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_sin(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_cos(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_tan(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_sinh(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_cosh(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_tanh(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_asin(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_acos(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_atan(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_cbrt(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_ceil(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_exp2(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_floor(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_log2(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_log10(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_pow2(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_round(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_signum(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_recip(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_ln_1p(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_neg(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_to_degrees(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_to_radians(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_real(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_imag(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_conjugate(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_angle(CData $a, CData $a_meta, int $deg, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_iscomplex(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_isreal(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_powi(CData $a, CData $a_meta, int $exp, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_powf(CData $a, CData $a_meta, float $exp, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_clamp(CData $a, CData $a_meta, float $min_val, float $max_val, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_sigmoid(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_softmax(CData $handle, CData $meta, int $axis, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_sum(CData $handle, CData $meta, CData $out_value, CData $out_dtype_ptr)
 * @method int   ndarray_sum_axis(CData $handle, CData $meta, int $axis, bool $keepdims, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_mean(CData $handle, CData $meta, CData $out_value, CData $out_dtype_ptr)
 * @method int   ndarray_mean_axis(CData $handle, CData $meta, int $axis, bool $keepdims, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_min(CData $handle, CData $meta, CData $out_value, CData $out_dtype_ptr)
 * @method int   ndarray_min_axis(CData $handle, CData $meta, int $axis, bool $keepdims, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_max(CData $handle, CData $meta, CData $out_value, CData $out_dtype_ptr)
 * @method int   ndarray_max_axis(CData $handle, CData $meta, int $axis, bool $keepdims, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_argmin(CData $handle, CData $meta, CData $out_value, CData $out_dtype_ptr)
 * @method int   ndarray_argmin_axis(CData $handle, CData $meta, int $axis, bool $keepdims, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_argmax(CData $handle, CData $meta, CData $out_value, CData $out_dtype_ptr)
 * @method int   ndarray_argmax_axis(CData $handle, CData $meta, int $axis, bool $keepdims, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_product(CData $handle, CData $meta, CData $out_value, CData $out_dtype_ptr)
 * @method int   ndarray_product_axis(CData $handle, CData $meta, int $axis, bool $keepdims, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_cumsum(CData $handle, CData $meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_cumsum_axis(CData $handle, CData $meta, int $axis, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_cumprod(CData $handle, CData $meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_cumprod_axis(CData $handle, CData $meta, int $axis, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_var(CData $handle, CData $meta, float $ddof, CData $out_value, CData $out_dtype_ptr)
 * @method int   ndarray_var_axis(CData $handle, CData $meta, int $axis, bool $keepdims, float $ddof, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_std(CData $handle, CData $meta, float $ddof, CData $out_value, CData $out_dtype_ptr)
 * @method int   ndarray_std_axis(CData $handle, CData $meta, int $axis, bool $keepdims, float $ddof, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_any(CData $handle, CData $meta, CData $out_value, CData $out_dtype_ptr)
 * @method int   ndarray_any_axis(CData $handle, CData $meta, int $axis, bool $keepdims, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_all(CData $handle, CData $meta, CData $out_value, CData $out_dtype_ptr)
 * @method int   ndarray_all_axis(CData $handle, CData $meta, int $axis, bool $keepdims, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_bincount(CData $handle, CData $meta, int $minlength, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_sort_axis(CData $handle, CData $meta, int $axis, int $kind, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_sort_flat(CData $handle, CData $meta, int $kind, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_argsort_axis(CData $handle, CData $meta, int $axis, int $kind, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_argsort_flat(CData $handle, CData $meta, int $kind, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_topk_axis(CData $handle, CData $meta, int $axis, int $k, bool $largest, bool $sorted, int $kind, CData $out_values, CData $out_indices, CData $out_shape, int $max_ndim)
 * @method int   ndarray_topk_flat(CData $handle, CData $meta, int $k, bool $largest, bool $sorted, int $kind, CData $out_values, CData $out_indices, CData $out_shape)
 * @method int   ndarray_astype(CData $handle, CData $meta, int $target_dtype, CData $out_handle)
 * @method int   ndarray_reshape(CData $handle, CData $meta, CData $new_shape, int $new_ndim, int $order, CData $out_handle)
 * @method int   ndarray_transpose(CData $handle, CData $meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_flip(CData $handle, CData $meta, CData $axes, int $num_axes, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_permute(CData $handle, CData $meta, CData $axes, int $num_axes, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_flatten(CData $handle, CData $meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_ravel(CData $handle, CData $meta, int $order, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_pad(CData $handle, CData $meta, CData $pad_width, int $mode, CData $constant_values, int $constant_values_len, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_dot(CData $a, CData $a_meta, CData $b, CData $b_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_matmul(CData $a, CData $a_meta, CData $b, CData $b_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_diagonal(CData $handle, CData $meta, int $offset, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_from_diag(CData $handle, CData $meta, int $offset, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_trace(CData $handle, CData $meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_norm(CData $handle, CData $meta, int $ord, CData $out_value, CData $out_dtype_ptr)
 * @method int   ndarray_norm_axis(CData $handle, CData $meta, int $axis, bool $keepdims, int $ord, CData $out_handle)
 * @method int   ndarray_solve(CData $a, CData $a_meta, CData $b, CData $b_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_inv(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_det(CData $a, CData $a_meta, CData $out_value, CData $out_dtype_ptr)
 * @method int   ndarray_svd(CData $a, CData $a_meta, int $calc_u, int $calc_vt, CData $out_u, CData $out_dtype_u, CData $out_ndim_u, CData $out_shape_u, int $max_ndim, CData $out_s, CData $out_dtype_s, CData $out_ndim_s, CData $out_shape_s, CData $out_vt, CData $out_dtype_vt, CData $out_ndim_vt, CData $out_shape_vt)
 * @method int   ndarray_qr(CData $a, CData $a_meta, CData $out_q, CData $out_dtype_q, CData $out_ndim_q, CData $out_shape_q, int $max_ndim, CData $out_r, CData $out_dtype_r, CData $out_ndim_r, CData $out_shape_r)
 * @method int   ndarray_eig(CData $a, CData $a_meta, CData $out_eigvals, CData $out_dtype_eigvals, CData $out_ndim_eigvals, CData $out_shape_eigvals, int $max_ndim, CData $out_eigvecs, CData $out_dtype_eigvecs, CData $out_ndim_eigvecs, CData $out_shape_eigvecs)
 * @method int   ndarray_eigvals(CData $a, CData $a_meta, CData $out_eigvals, CData $out_dtype_eigvals, CData $out_ndim_eigvals, CData $out_shape_eigvals, int $max_ndim)
 * @method int   ndarray_eigh(CData $a, CData $a_meta, int $uplo, CData $out_eigvals, CData $out_dtype_eigvals, CData $out_ndim_eigvals, CData $out_shape_eigvals, int $max_ndim, CData $out_eigvecs, CData $out_dtype_eigvecs, CData $out_ndim_eigvecs, CData $out_shape_eigvecs)
 * @method int   ndarray_eigvalsh(CData $a, CData $a_meta, int $uplo, CData $out_eigvals, CData $out_dtype_eigvals, CData $out_ndim_eigvals, CData $out_shape_eigvals, int $max_ndim)
 * @method int   ndarray_cholesky(CData $a, CData $a_meta, int $upper, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_lstsq(CData $a, CData $a_meta, CData $b, CData $b_meta, CData $out_solution, CData $out_residuals, CData $out_rank, CData $out_s, CData $out_dtype_sol, CData $out_ndim_sol, CData $out_shape_sol, CData $out_dtype_res, CData $out_ndim_res, CData $out_shape_res, CData $out_dtype_s, CData $out_ndim_s, CData $out_shape_s, int $max_ndim)
 * @method int   ndarray_pinv(CData $a, CData $a_meta, CData $rcond, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_cond(CData $a, CData $a_meta, CData $out_value, CData $out_dtype_ptr)
 * @method int   ndarray_rank(CData $a, CData $a_meta, CData $tol, CData $out_rank)
 * @method int   ndarray_einsum(CData $a, CData $a_meta, ?CData $b, ?CData $b_meta, CData $subscripts, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_fft(CData $handle, CData $meta, int $axis, int $n, int $norm, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_ifft(CData $handle, CData $meta, int $axis, int $n, int $norm, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_fftn(CData $handle, CData $meta, ?CData $axes, int $n_axes, int $norm, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_ifftn(CData $handle, CData $meta, ?CData $axes, int $n_axes, int $norm, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_rfft(CData $handle, CData $meta, int $axis, int $n, int $norm, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_irfft(CData $handle, CData $meta, int $axis, int $n, int $norm, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_dct(CData $handle, CData $meta, int $axis, int $n, int $dct_type, int $norm, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_idct(CData $handle, CData $meta, int $axis, int $n, int $dct_type, int $norm, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_dctn(CData $handle, CData $meta, ?CData $axes, int $n_axes, int $dct_type, int $norm, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_idctn(CData $handle, CData $meta, ?CData $axes, int $n_axes, int $dct_type, int $norm, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_bartlett(int $m, bool $periodic, CData $out_handle)
 * @method int   ndarray_blackman(int $m, bool $periodic, CData $out_handle)
 * @method int   ndarray_bohman(int $m, bool $periodic, CData $out_handle)
 * @method int   ndarray_boxcar(int $m, bool $periodic, CData $out_handle)
 * @method int   ndarray_hamming(int $m, bool $periodic, CData $out_handle)
 * @method int   ndarray_hanning(int $m, bool $periodic, CData $out_handle)
 * @method int   ndarray_kaiser(int $m, float $beta, bool $periodic, CData $out_handle)
 * @method int   ndarray_lanczos(int $m, bool $periodic, CData $out_handle)
 * @method int   ndarray_triang(int $m, bool $periodic, CData $out_handle)
 * @method int   ndarray_concatenate(CData $handles, CData $handles_meta, int $num_arrays, int $axis, CData $out_handle, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_stack(CData $handles, CData $handles_meta, int $num_arrays, int $axis, CData $out_handle, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_split(CData $handle, CData $meta, int $axis, CData $indices, int $num_indices, CData $out_offsets, CData $out_shapes, CData $out_strides)
 * @method int   ndarray_tile(CData $handle, CData $meta, CData $reps, int $reps_len, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method int   ndarray_repeat(CData $handle, CData $meta, CData $repeats, int $repeats_len, int $axis, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim)
 * @method CData new(string $type, bool $owned = true)
 * @method CData cast(string $type, CData $ptr)
 */
final class Lib
{
    /** Maximum number of dimensions for output shape buffer (caller-allocated). */
    public const MAX_NDIM = 32;

    /**
     * Platform-specific library configurations.
     *
     * @var array<string, array{directory: string, libraryTemplate: string}>
     */
    private const PLATFORM_CONFIGS = [
        'linux-x86_64' => [
            'directory' => 'linux-x86_64',
            'libraryTemplate' => 'libndarray_php.so.*',
        ],
        'linux-arm64' => [
            'directory' => 'linux-arm64',
            'libraryTemplate' => 'libndarray_php.so.*',
        ],
        'darwin-x86_64' => [
            'directory' => 'darwin-x86_64',
            'libraryTemplate' => 'libndarray_php-*.dylib',
        ],
        'darwin-arm64' => [
            'directory' => 'darwin-arm64',
            'libraryTemplate' => 'libndarray_php-*.dylib',
        ],
        'windows-64' => [
            'directory' => 'windows-64',
            'libraryTemplate' => 'ndarray_php*.dll',
        ],
    ];

    private static ?self $instance = null;

    private static ?string $libraryPath = null;

    private \FFI $ffi;

    private function __construct() {}

    /**
     * Forward any undefined method call to the underlying \FFI instance.
     *
     * @param array<mixed> $args
     */
    public function __call(string $name, array $args): mixed
    {
        return $this->ffi->{$name}(...$args);
    }

    /**
     * Initialize FFI with the Rust library.
     */
    public static function init(?string $libraryPath = null): void
    {
        if (null !== self::$instance) {
            return;
        }

        self::$libraryPath = $libraryPath ?? self::findLibrary();

        $headerPath = \dirname(__DIR__, 2).'/include/ndarray_php.h';

        if (!file_exists($headerPath)) {
            throw new NDArrayException(
                "Header file not found: {$headerPath}. Did you build the Rust library?"
            );
        }

        if (!file_exists(self::$libraryPath)) {
            throw new NDArrayException(
                'Library not found: '.self::$libraryPath
            );
        }

        $instance = new self();
        $instance->ffi = \FFI::cdef(
            file_get_contents($headerPath),
            self::$libraryPath
        );

        self::$instance = $instance;
    }

    /**
     * Get the singleton Lib instance.
     */
    public static function get(): self
    {
        if (null === self::$instance) {
            self::init();
        }

        return self::$instance;
    }

    /**
     * Reset the singleton instance (mainly for testing).
     */
    public static function reset(): void
    {
        self::$instance = null;
        self::$libraryPath = null;
    }

    /**
     * Get the last error message from Rust.
     */
    public function getLastError(): string
    {
        $buffer = $this->new('char[1024]');
        $len = $this->ndarray_get_last_error($buffer, 1024);

        if (0 === $len) {
            return 'Unknown error';
        }

        return \FFI::string($buffer, $len);
    }

    /**
     * Check the status code returned by a C function.
     *
     * @throws NDArrayException
     */
    public function checkStatus(int $code): void
    {
        if (0 === $code) {
            return;
        }

        $message = $this->getLastError();

        match ($code) {
            1 => throw new NDArrayException($message),
            2 => throw new ShapeException($message),
            3 => throw new DTypeException($message),
            4 => throw new AllocationException($message),
            5 => throw new PanicException($message),
            6 => throw new IndexException($message),
            7 => throw new MathException($message),
            default => throw new NDArrayException($message),
        };
    }

    /**
     * Create a C array from a PHP array with the given type.
     *
     * @param string                $type C type (e.g., 'double', 'int64_t', 'size_t')
     * @param array<bool|float|int> $data PHP array of values
     *
     * @return CData The allocated C array
     */
    public function createCArray(string $type, array $data): CData
    {
        $count = \count($data);

        if (0 === $count) {
            return $this->new("{$type}[1]");
        }

        $cArray = $this->new("{$type}[{$count}]");

        foreach ($data as $i => $value) {
            $cArray[$i] = $value;
        }

        return $cArray;
    }

    /**
     * Read N elements from a size_t C array and return them as a PHP int array.
     *
     * @param CData $ptr    Pointer to size_t array from FFI (or buffer)
     * @param int   $length Number of elements to read
     *
     * @return array<int>
     */
    public function readSizeTArray(CData $ptr, int $length): array
    {
        $result = [];
        for ($i = 0; $i < $length; ++$i) {
            $result[] = (int) $ptr[$i];
        }

        return $result;
    }

    /**
     * Get the address of a CData value.
     */
    public static function addr(CData $value): CData
    {
        return \FFI::addr($value);
    }

    /**
     * Find the library file based on platform and architecture.
     */
    private static function findLibrary(): string
    {
        $platformConfig = Platform::findBestMatch(self::PLATFORM_CONFIGS);

        if (false === $platformConfig) {
            $current = Platform::current();

            throw new NDArrayException(
                "Unsupported platform: {$current['os']}-{$current['arch']}. "
                .'Supported platforms: '.implode(', ', array_keys(self::PLATFORM_CONFIGS))
            );
        }

        $baseDir = \dirname(__DIR__, 2).'/lib';
        $platformDir = "{$baseDir}/{$platformConfig['directory']}";

        if (!is_dir($platformDir)) {
            throw new NDArrayException(
                "Platform directory not found: {$platformDir}. Did you build the Rust library?"
            );
        }

        $template = $platformConfig['libraryTemplate'];

        $pattern = "{$platformDir}/{$template}";
        $files = glob($pattern);

        if (empty($files)) {
            throw new NDArrayException(
                "Library not found: {$pattern}. Did you build the Rust library?"
            );
        }

        return $files[0];
    }
}
