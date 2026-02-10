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
    /**
     * Get the item size for a dtype.
     */
    public function dtype_item_size(int $dtype): int;

    /**
     * Check if a dtype value is valid.
     */
    public function dtype_is_valid(int $dtype): bool;

    /**
     * Check if a dtype is a signed integer.
     */
    public function dtype_is_signed(int $dtype): bool;

    /**
     * Check if a dtype is an unsigned integer.
     */
    public function dtype_is_unsigned(int $dtype): bool;

    /**
     * Check if a dtype is an integer type.
     */
    public function dtype_is_integer(int $dtype): bool;

    /**
     * Check if a dtype is a floating-point type.
     */
    public function dtype_is_float(int $dtype): bool;

    /**
     * Check if a dtype is a boolean type.
     */
    public function dtype_is_bool(int $dtype): bool;

    /**
     * Get the promoted dtype when combining two dtypes.
     */
    public function dtype_promote(int $a, int $b): int;

    /**
     * Get the last error message.
     */
    public function ndarray_get_last_error(CData $buf, int $len): int;

    // =========================================================================
    // Array Creation
    // =========================================================================

    public function ndarray_create_int8(CData $data, int $len, CData $shape, int $ndim, CData $out_handle): int;
    public function ndarray_create_int16(CData $data, int $len, CData $shape, int $ndim, CData $out_handle): int;
    public function ndarray_create_int32(CData $data, int $len, CData $shape, int $ndim, CData $out_handle): int;
    public function ndarray_create_int64(CData $data, int $len, CData $shape, int $ndim, CData $out_handle): int;
    public function ndarray_create_uint8(CData $data, int $len, CData $shape, int $ndim, CData $out_handle): int;
    public function ndarray_create_uint16(CData $data, int $len, CData $shape, int $ndim, CData $out_handle): int;
    public function ndarray_create_uint32(CData $data, int $len, CData $shape, int $ndim, CData $out_handle): int;
    public function ndarray_create_uint64(CData $data, int $len, CData $shape, int $ndim, CData $out_handle): int;
    public function ndarray_create_float32(CData $data, int $len, CData $shape, int $ndim, CData $out_handle): int;
    public function ndarray_create_float64(CData $data, int $len, CData $shape, int $ndim, CData $out_handle): int;
    public function ndarray_create_bool(CData $data, int $len, CData $shape, int $ndim, CData $out_handle): int;

    // =========================================================================
    // Bulk Data Access
    // =========================================================================

    public function ndarray_get_data_int8(CData $handle, CData $out_data, int $max_len, CData $out_len): int;
    public function ndarray_get_data_int16(CData $handle, CData $out_data, int $max_len, CData $out_len): int;
    public function ndarray_get_data_int32(CData $handle, CData $out_data, int $max_len, CData $out_len): int;
    public function ndarray_get_data_int64(CData $handle, CData $out_data, int $max_len, CData $out_len): int;
    public function ndarray_get_data_uint8(CData $handle, CData $out_data, int $max_len, CData $out_len): int;
    public function ndarray_get_data_uint16(CData $handle, CData $out_data, int $max_len, CData $out_len): int;
    public function ndarray_get_data_uint32(CData $handle, CData $out_data, int $max_len, CData $out_len): int;
    public function ndarray_get_data_uint64(CData $handle, CData $out_data, int $max_len, CData $out_len): int;
    public function ndarray_get_data_float32(CData $handle, CData $out_data, int $max_len, CData $out_len): int;
    public function ndarray_get_data_float64(CData $handle, CData $out_data, int $max_len, CData $out_len): int;
    public function ndarray_get_data_bool(CData $handle, CData $out_data, int $max_len, CData $out_len): int;

    // =========================================================================
    // Element Get (type-specific scalar read)
    // =========================================================================

    public function ndarray_get_element_int8(CData $handle, int $flat_index, CData $out_value): int;
    public function ndarray_get_element_int16(CData $handle, int $flat_index, CData $out_value): int;
    public function ndarray_get_element_int32(CData $handle, int $flat_index, CData $out_value): int;
    public function ndarray_get_element_int64(CData $handle, int $flat_index, CData $out_value): int;
    public function ndarray_get_element_uint8(CData $handle, int $flat_index, CData $out_value): int;
    public function ndarray_get_element_uint16(CData $handle, int $flat_index, CData $out_value): int;
    public function ndarray_get_element_uint32(CData $handle, int $flat_index, CData $out_value): int;
    public function ndarray_get_element_uint64(CData $handle, int $flat_index, CData $out_value): int;
    public function ndarray_get_element_float32(CData $handle, int $flat_index, CData $out_value): int;
    public function ndarray_get_element_float64(CData $handle, int $flat_index, CData $out_value): int;
    public function ndarray_get_element_bool(CData $handle, int $flat_index, CData $out_value): int;

    // =========================================================================
    // Element Set (type-specific scalar write)
    // =========================================================================

    public function ndarray_set_element_int8(CData $handle, int $flat_index, int $value): int;
    public function ndarray_set_element_int16(CData $handle, int $flat_index, int $value): int;
    public function ndarray_set_element_int32(CData $handle, int $flat_index, int $value): int;
    public function ndarray_set_element_int64(CData $handle, int $flat_index, int $value): int;
    public function ndarray_set_element_uint8(CData $handle, int $flat_index, int $value): int;
    public function ndarray_set_element_uint16(CData $handle, int $flat_index, int $value): int;
    public function ndarray_set_element_uint32(CData $handle, int $flat_index, int $value): int;
    public function ndarray_set_element_uint64(CData $handle, int $flat_index, int $value): int;
    public function ndarray_set_element_float32(CData $handle, int $flat_index, float $value): int;
    public function ndarray_set_element_float64(CData $handle, int $flat_index, float $value): int;
    public function ndarray_set_element_bool(CData $handle, int $flat_index, int $value): int;

    // =========================================================================
    // Slice Filling (view filling)
    // =========================================================================

    public function ndarray_fill_int8(CData $handle, int $value, int $offset, CData $shape, CData $strides, int $ndim): int;
    public function ndarray_fill_int16(CData $handle, int $value, int $offset, CData $shape, CData $strides, int $ndim): int;
    public function ndarray_fill_int32(CData $handle, int $value, int $offset, CData $shape, CData $strides, int $ndim): int;
    public function ndarray_fill_int64(CData $handle, int $value, int $offset, CData $shape, CData $strides, int $ndim): int;
    public function ndarray_fill_uint8(CData $handle, int $value, int $offset, CData $shape, CData $strides, int $ndim): int;
    public function ndarray_fill_uint16(CData $handle, int $value, int $offset, CData $shape, CData $strides, int $ndim): int;
    public function ndarray_fill_uint32(CData $handle, int $value, int $offset, CData $shape, CData $strides, int $ndim): int;
    public function ndarray_fill_uint64(CData $handle, int $value, int $offset, CData $shape, CData $strides, int $ndim): int;
    public function ndarray_fill_float32(CData $handle, float $value, int $offset, CData $shape, CData $strides, int $ndim): int;
    public function ndarray_fill_float64(CData $handle, float $value, int $offset, CData $shape, CData $strides, int $ndim): int;
    public function ndarray_fill_bool(CData $handle, int $value, int $offset, CData $shape, CData $strides, int $ndim): int;

    // =========================================================================
    // Slice Assignment (copy from another view)
    // =========================================================================

    public function ndarray_assign_int8(CData $dst, int $doff, CData $dshp, CData $dstr, CData $src, int $soff, CData $sshp, CData $sstr, int $ndim): int;
    public function ndarray_assign_int16(CData $dst, int $doff, CData $dshp, CData $dstr, CData $src, int $soff, CData $sshp, CData $sstr, int $ndim): int;
    public function ndarray_assign_int32(CData $dst, int $doff, CData $dshp, CData $dstr, CData $src, int $soff, CData $sshp, CData $sstr, int $ndim): int;
    public function ndarray_assign_int64(CData $dst, int $doff, CData $dshp, CData $dstr, CData $src, int $soff, CData $sshp, CData $sstr, int $ndim): int;
    public function ndarray_assign_uint8(CData $dst, int $doff, CData $dshp, CData $dstr, CData $src, int $soff, CData $sshp, CData $sstr, int $ndim): int;
    public function ndarray_assign_uint16(CData $dst, int $doff, CData $dshp, CData $dstr, CData $src, int $soff, CData $sshp, CData $sstr, int $ndim): int;
    public function ndarray_assign_uint32(CData $dst, int $doff, CData $dshp, CData $dstr, CData $src, int $soff, CData $sshp, CData $sstr, int $ndim): int;
    public function ndarray_assign_uint64(CData $dst, int $doff, CData $dshp, CData $dstr, CData $src, int $soff, CData $sshp, CData $sstr, int $ndim): int;
    public function ndarray_assign_float32(CData $dst, int $doff, CData $dshp, CData $dstr, CData $src, int $soff, CData $sshp, CData $sstr, int $ndim): int;
    public function ndarray_assign_float64(CData $dst, int $doff, CData $dshp, CData $dstr, CData $src, int $soff, CData $sshp, CData $sstr, int $ndim): int;
    public function ndarray_assign_bool(CData $dst, int $doff, CData $dshp, CData $dstr, CData $src, int $soff, CData $sshp, CData $sstr, int $ndim): int;

    // =========================================================================
    // Generators
    // =========================================================================

    public function ndarray_copy(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, CData $out_handle): int;

    public function ndarray_zeros(CData $shape, int $ndim, int $dtype, CData $out_handle): int;
    public function ndarray_ones(CData $shape, int $ndim, int $dtype, CData $out_handle): int;
    public function ndarray_full(CData $shape, int $ndim, CData $value, int $dtype, CData $out_handle): int;
    public function ndarray_eye(int $n, int $m, int $k, int $dtype, CData $out_handle): int;
    public function ndarray_arange(float $start, float $stop, float $step, int $dtype, CData $out_handle): int;
    public function ndarray_linspace(float $start, float $stop, int $num, bool $endpoint, int $dtype, CData $out_handle): int;
    public function ndarray_logspace(float $start, float $stop, int $num, float $base, int $dtype, CData $out_handle): int;
    public function ndarray_geomspace(float $start, float $stop, int $num, int $dtype, CData $out_handle): int;

    // =========================================================================
    // Arithmetic Operations - Array-Array
    // =========================================================================

    public function ndarray_add(CData $a, int $a_offset, CData $a_shape, CData $a_strides, CData $b, int $b_offset, CData $b_shape, CData $b_strides, int $ndim, CData $out_handle): int;
    public function ndarray_sub(CData $a, int $a_offset, CData $a_shape, CData $a_strides, CData $b, int $b_offset, CData $b_shape, CData $b_strides, int $ndim, CData $out_handle): int;
    public function ndarray_mul(CData $a, int $a_offset, CData $a_shape, CData $a_strides, CData $b, int $b_offset, CData $b_shape, CData $b_strides, int $ndim, CData $out_handle): int;
    public function ndarray_div(CData $a, int $a_offset, CData $a_shape, CData $a_strides, CData $b, int $b_offset, CData $b_shape, CData $b_strides, int $ndim, CData $out_handle): int;

    // =========================================================================
    // Arithmetic Operations - Array-Scalar
    // =========================================================================

    public function ndarray_add_scalar(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, float $scalar, CData $out_handle): int;
    public function ndarray_sub_scalar(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, float $scalar, CData $out_handle): int;
    public function ndarray_mul_scalar(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, float $scalar, CData $out_handle): int;
    public function ndarray_div_scalar(CData $a, int $a_offset, CData $a_shape, CData $a_strides, int $ndim, float $scalar, CData $out_handle): int;

    // =========================================================================
    // Properties
    // =========================================================================

    public function ndarray_free(CData $handle): int;
    public function ndarray_ndim(CData $handle, CData $out_ndim): int;
    public function ndarray_len(CData $handle, CData $out_len): int;
    public function ndarray_dtype(CData $handle, CData $out_dtype): int;
    public function ndarray_shape(CData $handle, CData $out_shape, int $max_ndim, CData $out_ndim): int;

    // =========================================================================
    // Serialization
    // =========================================================================

    public function ndarray_to_json(CData $handle, CData $out_ptr, CData $out_len, int $precision): int;
    public function ndarray_view_to_json(CData $handle, int $offset, CData $shape, CData $strides, int $ndim, int $precision, CData $out_ptr, CData $out_len): int;
    public function ndarray_free_string(CData $ptr): void;
}
