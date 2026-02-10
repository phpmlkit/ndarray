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

    /**
     * Create an NDArray from i8 data.
     */
    public function ndarray_create_int8(CData $data, int $len, CData $shape, int $ndim, CData $out_handle): int;

    /**
     * Create an NDArray from i16 data.
     */
    public function ndarray_create_int16(CData $data, int $len, CData $shape, int $ndim, CData $out_handle): int;

    /**
     * Create an NDArray from i32 data.
     */
    public function ndarray_create_int32(CData $data, int $len, CData $shape, int $ndim, CData $out_handle): int;

    /**
     * Create an NDArray from i64 data.
     */
    public function ndarray_create_int64(CData $data, int $len, CData $shape, int $ndim, CData $out_handle): int;

    /**
     * Create an NDArray from u8 data.
     */
    public function ndarray_create_uint8(CData $data, int $len, CData $shape, int $ndim, CData $out_handle): int;

    /**
     * Create an NDArray from u16 data.
     */
    public function ndarray_create_uint16(CData $data, int $len, CData $shape, int $ndim, CData $out_handle): int;

    /**
     * Create an NDArray from u32 data.
     */
    public function ndarray_create_uint32(CData $data, int $len, CData $shape, int $ndim, CData $out_handle): int;

    /**
     * Create an NDArray from u64 data.
     */
    public function ndarray_create_uint64(CData $data, int $len, CData $shape, int $ndim, CData $out_handle): int;

    /**
     * Create an NDArray from f32 data.
     */
    public function ndarray_create_float32(CData $data, int $len, CData $shape, int $ndim, CData $out_handle): int;

    /**
     * Create an NDArray from f64 data.
     */
    public function ndarray_create_float64(CData $data, int $len, CData $shape, int $ndim, CData $out_handle): int;

    /**
     * Create an NDArray from bool data.
     */
    public function ndarray_create_bool(CData $data, int $len, CData $shape, int $ndim, CData $out_handle): int;

    /**
     * Get int8 data.
     */
    public function ndarray_get_data_int8(CData $handle, CData $out_data, int $max_len, CData $out_len): int;

    /**
     * Get int16 data.
     */
    public function ndarray_get_data_int16(CData $handle, CData $out_data, int $max_len, CData $out_len): int;

    /**
     * Get int32 data.
     */
    public function ndarray_get_data_int32(CData $handle, CData $out_data, int $max_len, CData $out_len): int;

    /**
     * Get int64 data.
     */
    public function ndarray_get_data_int64(CData $handle, CData $out_data, int $max_len, CData $out_len): int;

    /**
     * Get uint8 data.
     */
    public function ndarray_get_data_uint8(CData $handle, CData $out_data, int $max_len, CData $out_len): int;

    /**
     * Get uint16 data.
     */
    public function ndarray_get_data_uint16(CData $handle, CData $out_data, int $max_len, CData $out_len): int;

    /**
     * Get uint32 data.
     */
    public function ndarray_get_data_uint32(CData $handle, CData $out_data, int $max_len, CData $out_len): int;

    /**
     * Get uint64 data.
     */
    public function ndarray_get_data_uint64(CData $handle, CData $out_data, int $max_len, CData $out_len): int;

    /**
     * Get float32 data.
     */
    public function ndarray_get_data_float32(CData $handle, CData $out_data, int $max_len, CData $out_len): int;

    /**
     * Get float64 data.
     */
    public function ndarray_get_data_float64(CData $handle, CData $out_data, int $max_len, CData $out_len): int;

    /**
     * Get bool data.
     */
    public function ndarray_get_data_bool(CData $handle, CData $out_data, int $max_len, CData $out_len): int;

    /**
     * Create an array filled with zeros.
     */
    public function ndarray_zeros(CData $shape, int $ndim, int $dtype, CData $out_handle): int;

    /**
     * Create an array filled with ones.
     */
    public function ndarray_ones(CData $shape, int $ndim, int $dtype, CData $out_handle): int;

    /**
     * Create an array filled with a specific value.
     */
    public function ndarray_full(CData $shape, int $ndim, CData $value, int $dtype, CData $out_handle): int;

    /**
     * Create an identity matrix (2D).
     */
    public function ndarray_eye(int $n, int $m, int $k, int $dtype, CData $out_handle): int;

    /**
     * Destroy an NDArray and free its memory.
     */
    public function ndarray_free(CData $handle): int;

    /**
     * Get the number of dimensions of an array.
     */
    public function ndarray_ndim(CData $handle, CData $out_ndim): int;

    /**
     * Get the total number of elements in an array.
     */
    public function ndarray_len(CData $handle, CData $out_len): int;

    /**
     * Get the dtype of an array.
     */
    public function ndarray_dtype(CData $handle, CData $out_dtype): int;

    /**
     * Get the shape of an array.
     */
    public function ndarray_shape(CData $handle, CData $out_shape, int $max_ndim, CData $out_ndim): int;

    /**
     * Serialize an NDArray to a nested JSON string.
     */
    public function ndarray_to_json(CData $handle, CData $out_ptr, CData $out_len, int $precision): int;

    /**
     * Free a string allocated by `ndarray_to_json`.
     */
    public function ndarray_free_string(CData $ptr): void;
}
