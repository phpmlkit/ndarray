<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\FFI;

use FFI\CData;

/**
 * Raw C-API interface for IDE autocompletion.
 *
 * @internal this interface is a type-hinting helper for FFI and should not be implemented
 */
interface Bindings
{
    // =========================================================================
    // DType Utilities
    // =========================================================================

    public function dtype_item_size(int $dtype): int;

    public function dtype_is_valid(int $dtype): bool;

    public function dtype_is_signed(int $dtype): bool;

    public function dtype_is_unsigned(int $dtype): bool;

    public function dtype_is_integer(int $dtype): bool;

    public function dtype_is_float(int $dtype): bool;

    public function dtype_is_bool(int $dtype): bool;

    public function dtype_promote(int $a, int $b): int;

    // =========================================================================
    // Error Handling
    // =========================================================================

    public function ndarray_get_last_error(CData $buf, int $len): int;

    // =========================================================================
    // Array Lifecycle
    // =========================================================================

    public function ndarray_create(CData $data, int $len, CData $shape, int $ndim, int $dtype, CData $out_handle): int;

    public function ndarray_copy(CData $handle, CData $meta, CData $out_handle): int;

    public function ndarray_free(CData $handle): int;

    // =========================================================================
    // Array Generators
    // =========================================================================

    public function ndarray_zeros(CData $shape, int $ndim, int $dtype, CData $out_handle): int;

    public function ndarray_ones(CData $shape, int $ndim, int $dtype, CData $out_handle): int;

    public function ndarray_full(CData $shape, int $ndim, CData $value, int $dtype, CData $out_handle): int;

    public function ndarray_eye(int $n, int $m, int $k, int $dtype, CData $out_handle): int;

    public function ndarray_arange(float $start, float $stop, float $step, int $dtype, CData $out_handle): int;

    public function ndarray_linspace(float $start, float $stop, int $num, bool $endpoint, int $dtype, CData $out_handle): int;

    public function ndarray_logspace(float $start, float $stop, int $num, float $base, int $dtype, CData $out_handle): int;

    public function ndarray_geomspace(float $start, float $stop, int $num, int $dtype, CData $out_handle): int;

    public function ndarray_random(CData $shape, int $ndim, int $dtype, bool $has_seed, int $seed, CData $out_handle): int;

    public function ndarray_random_int(int $low, int $high, CData $shape, int $ndim, int $dtype, bool $has_seed, int $seed, CData $out_handle): int;

    public function ndarray_randn(CData $shape, int $ndim, int $dtype, bool $has_seed, int $seed, CData $out_handle): int;

    public function ndarray_normal(float $mean, float $std, CData $shape, int $ndim, int $dtype, bool $has_seed, int $seed, CData $out_handle): int;

    public function ndarray_uniform(float $low, float $high, CData $shape, int $ndim, int $dtype, bool $has_seed, int $seed, CData $out_handle): int;

    // =========================================================================
    // Element Access & Mutation
    // =========================================================================

    public function ndarray_get_element(CData $handle, int $flat_index, CData $out_value): int;

    public function ndarray_set_element(CData $handle, int $flat_index, CData $value): int;

    public function ndarray_get_data(CData $handle, CData $meta, CData $out_data, int $max_len, CData $out_len): int;

    public function ndarray_take_flat(CData $handle, CData $meta, CData $indices, int $indices_len, CData $indices_shape, int $indices_ndim, CData $out_handle): int;

    public function ndarray_take_along_axis(CData $handle, CData $meta, CData $indices_handle, CData $indices_meta, int $axis, CData $out_handle): int;

    public function ndarray_put_flat(CData $handle, CData $meta, CData $indices, int $indices_len, CData $values, int $values_len, float $scalar_value, bool $has_scalar, CData $out_handle): int;

    public function ndarray_put_along_axis(CData $handle, CData $meta, CData $indices_handle, CData $indices_meta, int $axis, CData $values, int $values_len, float $scalar_value, bool $has_scalar, CData $out_handle): int;

    public function ndarray_where(CData $cond_handle, CData $cond_meta, CData $x_handle, CData $x_meta, CData $y_handle, CData $y_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_scatter_add_flat(CData $handle, CData $meta, CData $indices, int $indices_len, CData $updates, int $updates_len, float $scalar_update, bool $has_scalar, CData $out_handle): int;

    // =========================================================================
    // Slice Operations
    // =========================================================================

    public function ndarray_fill(CData $handle, CData $meta, CData $value): int;

    public function ndarray_assign(CData $dst, CData $dst_meta, CData $src, CData $src_meta): int;

    // =========================================================================
    // Arithmetic Operations
    // =========================================================================

    public function ndarray_add(CData $a, CData $a_meta, CData $b, CData $b_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_add_scalar(CData $a, CData $a_meta, float $scalar, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_sub(CData $a, CData $a_meta, CData $b, CData $b_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_sub_scalar(CData $a, CData $a_meta, float $scalar, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_mul(CData $a, CData $a_meta, CData $b, CData $b_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_mul_scalar(CData $a, CData $a_meta, float $scalar, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_div(CData $a, CData $a_meta, CData $b, CData $b_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_div_scalar(CData $a, CData $a_meta, float $scalar, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_rem(CData $a, CData $a_meta, CData $b, CData $b_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_rem_scalar(CData $a, CData $a_meta, float $scalar, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_hypot(CData $a, CData $a_meta, float $b, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    // =========================================================================
    // Comparison Operations
    // =========================================================================

    public function ndarray_eq(CData $a, CData $a_meta, CData $b, CData $b_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_eq_scalar(CData $a, CData $a_meta, float $scalar, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_ne(CData $a, CData $a_meta, CData $b, CData $b_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_ne_scalar(CData $a, CData $a_meta, float $scalar, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_gt(CData $a, CData $a_meta, CData $b, CData $b_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_gt_scalar(CData $a, CData $a_meta, float $scalar, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_gte(CData $a, CData $a_meta, CData $b, CData $b_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_gte_scalar(CData $a, CData $a_meta, float $scalar, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_lt(CData $a, CData $a_meta, CData $b, CData $b_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_lt_scalar(CData $a, CData $a_meta, float $scalar, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_lte(CData $a, CData $a_meta, CData $b, CData $b_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_lte_scalar(CData $a, CData $a_meta, float $scalar, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    // =========================================================================
    // Bitwise Operations
    // =========================================================================

    public function ndarray_bitand(CData $a, CData $a_meta, CData $b, CData $b_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_bitand_scalar(CData $a, CData $a_meta, int $scalar, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_bitor(CData $a, CData $a_meta, CData $b, CData $b_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_bitor_scalar(CData $a, CData $a_meta, int $scalar, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_bitxor(CData $a, CData $a_meta, CData $b, CData $b_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_bitxor_scalar(CData $a, CData $a_meta, int $scalar, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_left_shift(CData $a, CData $a_meta, CData $b, CData $b_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_left_shift_scalar(CData $a, CData $a_meta, int $scalar, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_right_shift(CData $a, CData $a_meta, CData $b, CData $b_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_right_shift_scalar(CData $a, CData $a_meta, int $scalar, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    // =========================================================================
    // Math Operations
    // =========================================================================

    // Unary operations (all return metadata: out_dtype, out_ndim, out_shape, max_ndim)
    public function ndarray_abs(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_sqrt(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_exp(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_log(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_ln(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_sin(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_cos(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_tan(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_sinh(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_cosh(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_tanh(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_asin(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_acos(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_atan(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_cbrt(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_ceil(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_exp2(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_floor(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_log2(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_log10(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_pow2(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_round(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_signum(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_recip(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_ln_1p(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_neg(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_to_degrees(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_to_radians(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_powi(CData $a, CData $a_meta, int $exp, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_powf(CData $a, CData $a_meta, float $exp, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_clamp(CData $a, CData $a_meta, float $min_val, float $max_val, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_sigmoid(CData $a, CData $a_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_softmax(CData $handle, CData $handle_meta, int $axis, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    // =========================================================================
    // Reductions
    // =========================================================================

    public function ndarray_sum(CData $handle, CData $handle_meta, CData $out_value, CData $out_dtype_ptr): int;

    public function ndarray_sum_axis(CData $handle, CData $handle_meta, int $axis, bool $keepdims, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_mean(CData $handle, CData $handle_meta, CData $out_value, CData $out_dtype_ptr): int;

    public function ndarray_mean_axis(CData $handle, CData $handle_meta, int $axis, bool $keepdims, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_min(CData $handle, CData $handle_meta, CData $out_value, CData $out_dtype_ptr): int;

    public function ndarray_min_axis(CData $handle, CData $handle_meta, int $axis, bool $keepdims, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_max(CData $handle, CData $handle_meta, CData $out_value, CData $out_dtype_ptr): int;

    public function ndarray_max_axis(CData $handle, CData $handle_meta, int $axis, bool $keepdims, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_argmin(CData $handle, CData $handle_meta, CData $out_value, CData $out_dtype_ptr): int;

    public function ndarray_argmin_axis(CData $handle, CData $handle_meta, int $axis, bool $keepdims, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_argmax(CData $handle, CData $handle_meta, CData $out_value, CData $out_dtype_ptr): int;

    public function ndarray_argmax_axis(CData $handle, CData $handle_meta, int $axis, bool $keepdims, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_product(CData $handle, CData $handle_meta, CData $out_value, CData $out_dtype_ptr): int;

    public function ndarray_product_axis(CData $handle, CData $handle_meta, int $axis, bool $keepdims, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_cumsum(CData $handle, CData $handle_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_cumsum_axis(CData $handle, CData $handle_meta, int $axis, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_cumprod(CData $handle, CData $handle_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_cumprod_axis(CData $handle, CData $handle_meta, int $axis, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_var(CData $handle, CData $handle_meta, float $ddof, CData $out_value, CData $out_dtype_ptr): int;

    public function ndarray_var_axis(CData $handle, CData $handle_meta, int $axis, bool $keepdims, float $ddof, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_std(CData $handle, CData $handle_meta, float $ddof, CData $out_value, CData $out_dtype_ptr): int;

    public function ndarray_std_axis(CData $handle, CData $handle_meta, int $axis, bool $keepdims, float $ddof, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_bincount(CData $handle, CData $handle_meta, int $minlength, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_sort_axis(CData $handle, CData $handle_meta, int $axis, int $kind, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_sort_flat(CData $handle, CData $handle_meta, int $kind, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_argsort_axis(CData $handle, CData $handle_meta, int $axis, int $kind, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_argsort_flat(CData $handle, CData $handle_meta, int $kind, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_topk_axis(CData $handle, CData $handle_meta, int $axis, int $k, bool $largest, bool $sorted, int $kind, CData $out_values, CData $out_indices, CData $out_shape, int $max_ndim): int;

    public function ndarray_topk_flat(CData $handle, CData $handle_meta, int $k, bool $largest, bool $sorted, int $kind, CData $out_values, CData $out_indices, CData $out_shape): int;

    // =========================================================================
    // Properties
    // =========================================================================

    public function ndarray_ndim(CData $handle, CData $out_ndim): int;

    public function ndarray_len(CData $handle, CData $out_len): int;

    public function ndarray_dtype(CData $handle, CData $out_dtype): int;

    public function ndarray_shape(CData $handle, CData $out_shape, int $max_ndim, CData $out_ndim): int;

    public function ndarray_scalar(CData $handle, CData $handle_meta, CData $out_value): int;

    // =========================================================================
    // Type Casting
    // =========================================================================

    public function ndarray_astype(CData $handle, CData $handle_meta, int $target_dtype, CData $out_handle): int;

    // =========================================================================
    // Shape Operations
    // =========================================================================

    public function ndarray_reshape(CData $handle, CData $handle_meta, CData $new_shape, int $new_ndim, int $order, CData $out_handle): int;

    public function ndarray_transpose(CData $handle, CData $handle_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_swap_axes(CData $handle, CData $handle_meta, int $axis1, int $axis2, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_merge_axes(CData $handle, CData $handle_meta, int $take, int $into, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_invert_axis(CData $handle, CData $handle_meta, int $axis, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_insert_axis(CData $handle, CData $handle_meta, int $axis, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_permute_axes(CData $handle, CData $handle_meta, CData $axes, int $num_axes, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_flatten(CData $handle, CData $handle_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_ravel(CData $handle, CData $handle_meta, int $order, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_squeeze(CData $handle, CData $handle_meta, CData $axes, int $num_axes, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_pad(CData $handle, CData $handle_meta, CData $pad_width, int $mode, CData $constant_values, int $constant_values_len, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    // =========================================================================
    // Linear Algebra Operations
    // =========================================================================

    public function ndarray_dot(CData $a, CData $a_meta, CData $b, CData $b_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_matmul(CData $a, CData $a_meta, CData $b, CData $b_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_diagonal(CData $handle, CData $handle_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_trace(CData $handle, CData $handle_meta, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_norm(CData $handle, CData $handle_meta, int $ord, CData $out_value, CData $out_dtype_ptr): int;

    public function ndarray_norm_axis(CData $handle, CData $handle_meta, int $axis, bool $keepdims, int $ord, CData $out_handle): int;

    // =========================================================================
    // Stacking (Joining and Splitting)
    // =========================================================================

    public function ndarray_concatenate(CData $handles, CData $handles_meta, int $num_arrays, int $axis, CData $out_handle, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_stack(CData $handles, CData $handles_meta, int $num_arrays, int $axis, CData $out_handle, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_split(CData $handle, CData $handle_meta, int $axis, CData $indices, int $num_indices, CData $out_offsets, CData $out_shapes, CData $out_strides): int;

    // =========================================================================
    // Tiling and Repeating
    // =========================================================================

    public function ndarray_tile(CData $handle, CData $handle_meta, CData $reps, int $reps_len, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;

    public function ndarray_repeat(CData $handle, CData $handle_meta, CData $repeats, int $repeats_len, int $axis, CData $out_handle, CData $out_dtype_ptr, CData $out_ndim, CData $out_shape, int $max_ndim): int;
}
