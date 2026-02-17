<?php

namespace NDArray\FFI;

use FFI\CData;

/**
 * Raw C-API interface for IDE autocompletion.
 *
 * @internal This interface is a type-hinting helper for FFI and should not be implemented.
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
    public function ndarray_copy(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, CData $out_handle): int;
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

    // =========================================================================
    // Element Access & Mutation
    // =========================================================================

    public function ndarray_get_element(CData $handle, int $flat_index, CData $out_value): int;
    public function ndarray_set_element(CData $handle, int $flat_index, CData $value): int;
    public function ndarray_get_data(CData $handle, CData $out_data, int $max_len, CData $out_len): int;

    // =========================================================================
    // Slice Operations
    // =========================================================================

    public function ndarray_fill(CData $handle, CData $value, int $offset, CData $shape, CData $strides, int $ndim): int;
    public function ndarray_assign(CData $dst, int $doff, CData $dshp, CData $dstr, CData $src, int $soff, CData $sshp, CData $sstr, int $ndim): int;

    // =========================================================================
    // Arithmetic Operations
    // =========================================================================

    public function ndarray_add(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $a_ndim, CData $b, int $b_offset, CData $b_shape, CData $b_strides, int $b_ndim, CData $out_handle, CData $out_shape, CData $out_ndim): int;
    public function ndarray_add_scalar(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, float $scalar, CData $out_handle): int;
    public function ndarray_sub(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $a_ndim, CData $b, int $b_offset, CData $b_shape, CData $b_strides, int $b_ndim, CData $out_handle, CData $out_shape, CData $out_ndim): int;
    public function ndarray_sub_scalar(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, float $scalar, CData $out_handle): int;
    public function ndarray_mul(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $a_ndim, CData $b, int $b_offset, CData $b_shape, CData $b_strides, int $b_ndim, CData $out_handle, CData $out_shape, CData $out_ndim): int;
    public function ndarray_mul_scalar(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, float $scalar, CData $out_handle): int;
    public function ndarray_div(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $a_ndim, CData $b, int $b_offset, CData $b_shape, CData $b_strides, int $b_ndim, CData $out_handle, CData $out_shape, CData $out_ndim): int;
    public function ndarray_div_scalar(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, float $scalar, CData $out_handle): int;
    public function ndarray_rem(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $a_ndim, CData $b, int $b_offset, CData $b_shape, CData $b_strides, int $b_ndim, CData $out_handle, CData $out_shape, CData $out_ndim): int;
    public function ndarray_rem_scalar(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, float $scalar, CData $out_handle): int;
    public function ndarray_hypot(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, float $b, CData $out_handle): int;

    // =========================================================================
    // Comparison Operations
    // =========================================================================

    public function ndarray_eq(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $a_ndim, CData $b, int $b_offset, CData $b_shape, CData $b_strides, int $b_ndim, CData $out_handle, CData $out_shape, CData $out_ndim): int;
    public function ndarray_eq_scalar(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, float $scalar, CData $out_handle): int;
    public function ndarray_ne(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $a_ndim, CData $b, int $b_offset, CData $b_shape, CData $b_strides, int $b_ndim, CData $out_handle, CData $out_shape, CData $out_ndim): int;
    public function ndarray_ne_scalar(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, float $scalar, CData $out_handle): int;
    public function ndarray_gt(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $a_ndim, CData $b, int $b_offset, CData $b_shape, CData $b_strides, int $b_ndim, CData $out_handle, CData $out_shape, CData $out_ndim): int;
    public function ndarray_gt_scalar(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, float $scalar, CData $out_handle): int;
    public function ndarray_gte(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $a_ndim, CData $b, int $b_offset, CData $b_shape, CData $b_strides, int $b_ndim, CData $out_handle, CData $out_shape, CData $out_ndim): int;
    public function ndarray_gte_scalar(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, float $scalar, CData $out_handle): int;
    public function ndarray_lt(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $a_ndim, CData $b, int $b_offset, CData $b_shape, CData $b_strides, int $b_ndim, CData $out_handle, CData $out_shape, CData $out_ndim): int;
    public function ndarray_lt_scalar(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, float $scalar, CData $out_handle): int;
    public function ndarray_lte(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $a_ndim, CData $b, int $b_offset, CData $b_shape, CData $b_strides, int $b_ndim, CData $out_handle, CData $out_shape, CData $out_ndim): int;
    public function ndarray_lte_scalar(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, float $scalar, CData $out_handle): int;

    // =========================================================================
    // Bitwise Operations
    // =========================================================================

    public function ndarray_bitand(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $a_ndim, CData $b, int $b_offset, CData $b_shape, CData $b_strides, int $b_ndim, CData $out_handle, CData $out_shape, CData $out_ndim): int;
    public function ndarray_bitand_scalar(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, int $scalar, CData $out_handle): int;
    public function ndarray_bitor(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $a_ndim, CData $b, int $b_offset, CData $b_shape, CData $b_strides, int $b_ndim, CData $out_handle, CData $out_shape, CData $out_ndim): int;
    public function ndarray_bitor_scalar(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, int $scalar, CData $out_handle): int;
    public function ndarray_bitxor(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $a_ndim, CData $b, int $b_offset, CData $b_shape, CData $b_strides, int $b_ndim, CData $out_handle, CData $out_shape, CData $out_ndim): int;
    public function ndarray_bitxor_scalar(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, int $scalar, CData $out_handle): int;
    public function ndarray_left_shift(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $a_ndim, CData $b, int $b_offset, CData $b_shape, CData $b_strides, int $b_ndim, CData $out_handle, CData $out_shape, CData $out_ndim): int;
    public function ndarray_left_shift_scalar(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, int $scalar, CData $out_handle): int;
    public function ndarray_right_shift(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $a_ndim, CData $b, int $b_offset, CData $b_shape, CData $b_strides, int $b_ndim, CData $out_handle, CData $out_shape, CData $out_ndim): int;
    public function ndarray_right_shift_scalar(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, int $scalar, CData $out_handle): int;

    // =========================================================================
    // Math Operations
    // =========================================================================

    // Unary operations
    public function ndarray_abs(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, CData $out_handle): int;
    public function ndarray_sqrt(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, CData $out_handle): int;
    public function ndarray_exp(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, CData $out_handle): int;
    public function ndarray_log(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, CData $out_handle): int;
    public function ndarray_ln(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, CData $out_handle): int;
    public function ndarray_sin(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, CData $out_handle): int;
    public function ndarray_cos(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, CData $out_handle): int;
    public function ndarray_tan(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, CData $out_handle): int;
    public function ndarray_sinh(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, CData $out_handle): int;
    public function ndarray_cosh(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, CData $out_handle): int;
    public function ndarray_tanh(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, CData $out_handle): int;
    public function ndarray_asin(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, CData $out_handle): int;
    public function ndarray_acos(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, CData $out_handle): int;
    public function ndarray_atan(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, CData $out_handle): int;
    public function ndarray_cbrt(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, CData $out_handle): int;
    public function ndarray_ceil(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, CData $out_handle): int;
    public function ndarray_exp2(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, CData $out_handle): int;
    public function ndarray_floor(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, CData $out_handle): int;
    public function ndarray_log2(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, CData $out_handle): int;
    public function ndarray_log10(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, CData $out_handle): int;
    public function ndarray_pow2(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, CData $out_handle): int;
    public function ndarray_round(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, CData $out_handle): int;
    public function ndarray_signum(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, CData $out_handle): int;
    public function ndarray_recip(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, CData $out_handle): int;
    public function ndarray_ln_1p(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, CData $out_handle): int;
    public function ndarray_neg(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, CData $out_handle): int;
    public function ndarray_to_degrees(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, CData $out_handle): int;
    public function ndarray_to_radians(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, CData $out_handle): int;
    public function ndarray_powi(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, int $exp, CData $out_handle): int;
    public function ndarray_powf(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, float $exp, CData $out_handle): int;
    public function ndarray_clamp(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, float $min_val, float $max_val, CData $out_handle): int;
    public function ndarray_sigmoid(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, CData $out_handle): int;
    public function ndarray_softmax(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, int $axis, CData $out_handle): int;

    // =========================================================================
    // Reductions
    // =========================================================================

    public function ndarray_sum(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, CData $out_value, CData $out_dtype): int;
    public function ndarray_sum_axis(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, int $axis, bool $keepdims, CData $out_handle): int;
    public function ndarray_mean(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, CData $out_value, CData $out_dtype): int;
    public function ndarray_mean_axis(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, int $axis, bool $keepdims, CData $out_handle): int;
    public function ndarray_min(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, CData $out_value, CData $out_dtype): int;
    public function ndarray_min_axis(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, int $axis, bool $keepdims, CData $out_handle): int;
    public function ndarray_max(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, CData $out_value, CData $out_dtype): int;
    public function ndarray_max_axis(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, int $axis, bool $keepdims, CData $out_handle): int;
    public function ndarray_argmin(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, CData $out_value, CData $out_dtype): int;
    public function ndarray_argmin_axis(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, int $axis, bool $keepdims, CData $out_handle): int;
    public function ndarray_argmax(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, CData $out_value, CData $out_dtype): int;
    public function ndarray_argmax_axis(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, int $axis, bool $keepdims, CData $out_handle): int;
    public function ndarray_product(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, CData $out_value, CData $out_dtype): int;
    public function ndarray_product_axis(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, int $axis, bool $keepdims, CData $out_handle): int;
    public function ndarray_cumsum(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, CData $out_handle): int;
    public function ndarray_cumsum_axis(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, int $axis, CData $out_handle): int;
    public function ndarray_cumprod(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, CData $out_handle): int;
    public function ndarray_cumprod_axis(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, int $axis, CData $out_handle): int;
    public function ndarray_var(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, float $ddof, CData $out_value, CData $out_dtype): int;
    public function ndarray_var_axis(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, int $axis, bool $keepdims, float $ddof, CData $out_handle): int;
    public function ndarray_std(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, float $ddof, CData $out_value, CData $out_dtype): int;
    public function ndarray_std_axis(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, int $axis, bool $keepdims, float $ddof, CData $out_handle): int;
    public function ndarray_sort_axis(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, int $axis, int $kind, CData $out_handle): int;
    public function ndarray_sort_flat(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, int $kind, CData $out_handle): int;
    public function ndarray_argsort_axis(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, int $axis, int $kind, CData $out_handle): int;
    public function ndarray_argsort_flat(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, int $kind, CData $out_handle): int;

    // =========================================================================
    // Properties
    // =========================================================================

    public function ndarray_ndim(CData $handle, CData $out_ndim): int;
    public function ndarray_len(CData $handle, CData $out_len): int;
    public function ndarray_dtype(CData $handle, CData $out_dtype): int;
    public function ndarray_shape(CData $handle, CData $out_shape, int $max_ndim, CData $out_ndim): int;

    // =========================================================================
    // Serialization
    // =========================================================================

    public function ndarray_to_json(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, CData $out_ptr, CData $out_len): int;
    public function ndarray_free_string(CData $ptr): void;
    public function ndarray_scalar(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, CData $out_value): int;

    // =========================================================================
    // Type Casting
    // =========================================================================

    public function ndarray_astype(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, int $target_dtype, CData $out_handle): int;

    // =========================================================================
    // Shape Operations
    // =========================================================================

    public function ndarray_reshape(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, CData $new_shape, int $new_ndim, int $order, CData $out_handle): int;
    public function ndarray_transpose(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, CData $out_handle): int;
    public function ndarray_swap_axes(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, int $axis1, int $axis2, CData $out_handle): int;
    public function ndarray_merge_axes(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, int $take, int $into, CData $out_handle): int;
    public function ndarray_invert_axis(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, int $axis, CData $out_handle): int;
    public function ndarray_insert_axis(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, int $axis, CData $out_handle): int;
    public function ndarray_permute_axes(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, CData $axes, int $num_axes, CData $out_handle): int;
    public function ndarray_flatten(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, CData $out_handle): int;
    public function ndarray_ravel(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, int $order, CData $out_handle): int;
    public function ndarray_squeeze(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, CData $axes, int $num_axes, CData $out_handle): int;

    // =========================================================================
    // Linear Algebra Operations
    // =========================================================================

    public function ndarray_dot(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $a_ndim, CData $b, int $b_offset, CData $b_shape, CData $b_strides, int $b_ndim, CData $out_handle): int;
    public function ndarray_matmul(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $a_ndim, CData $b, int $b_offset, CData $b_shape, CData $b_strides, int $b_ndim, CData $out_handle): int;
    public function ndarray_diagonal(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, CData $out_handle): int;
    public function ndarray_trace(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, CData $out_handle): int;

    // =========================================================================
    // Stacking (Joining and Splitting)
    // =========================================================================

    public function ndarray_concatenate(CData $handles, CData $offsets, CData $shapes, CData $strides, int $num_arrays, int $ndim, int $axis, CData $out_handle): int;
    public function ndarray_stack(CData $handles, CData $offsets, CData $shapes, CData $strides, int $num_arrays, int $ndim, int $axis, CData $out_handle): int;
    public function ndarray_split(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, int $axis, CData $indices, int $num_indices, CData $out_offsets, CData $out_shapes, CData $out_strides): int;
}
