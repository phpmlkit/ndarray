<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Traits;

use PhpMlKit\NDArray\ArrayMetadata;
use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\FFI\Lib;
use PhpMlKit\NDArray\NDArray;

/**
 * Static window function generators (signal processing).
 *
 * These mirror common NumPy windows and return Float64 arrays of shape `[m]`.
 */
trait HasWindowFunctions
{
    public static function bartlett(int $m, bool $periodic = false): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');
        $status = $ffi->ndarray_bartlett($m, $periodic, Lib::addr($outHandle));
        Lib::checkStatus($status);

        return new NDArray($outHandle, new ArrayMetadata([$m]), DType::Float64);
    }

    public static function blackman(int $m, bool $periodic = false): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');
        $status = $ffi->ndarray_blackman($m, $periodic, Lib::addr($outHandle));
        Lib::checkStatus($status);

        return new NDArray($outHandle, new ArrayMetadata([$m]), DType::Float64);
    }

    public static function bohman(int $m, bool $periodic = false): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');
        $status = $ffi->ndarray_bohman($m, $periodic, Lib::addr($outHandle));
        Lib::checkStatus($status);

        return new NDArray($outHandle, new ArrayMetadata([$m]), DType::Float64);
    }

    public static function boxcar(int $m, bool $periodic = false): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');
        $status = $ffi->ndarray_boxcar($m, $periodic, Lib::addr($outHandle));
        Lib::checkStatus($status);

        return new NDArray($outHandle, new ArrayMetadata([$m]), DType::Float64);
    }

    public static function hamming(int $m, bool $periodic = false): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');
        $status = $ffi->ndarray_hamming($m, $periodic, Lib::addr($outHandle));
        Lib::checkStatus($status);

        return new NDArray($outHandle, new ArrayMetadata([$m]), DType::Float64);
    }

    public static function hanning(int $m, bool $periodic = false): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');
        $status = $ffi->ndarray_hanning($m, $periodic, Lib::addr($outHandle));
        Lib::checkStatus($status);

        return new NDArray($outHandle, new ArrayMetadata([$m]), DType::Float64);
    }

    public static function hann(int $m, bool $periodic = false): NDArray
    {
        return self::hanning($m, $periodic);
    }

    public static function kaiser(int $m, float $beta, bool $periodic = false): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');
        $status = $ffi->ndarray_kaiser($m, $beta, $periodic, Lib::addr($outHandle));
        Lib::checkStatus($status);

        return new NDArray($outHandle, new ArrayMetadata([$m]), DType::Float64);
    }

    public static function lanczos(int $m, bool $periodic = false): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');
        $status = $ffi->ndarray_lanczos($m, $periodic, Lib::addr($outHandle));
        Lib::checkStatus($status);

        return new NDArray($outHandle, new ArrayMetadata([$m]), DType::Float64);
    }

    public static function triang(int $m, bool $periodic = false): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');
        $status = $ffi->ndarray_triang($m, $periodic, Lib::addr($outHandle));
        Lib::checkStatus($status);

        return new NDArray($outHandle, new ArrayMetadata([$m]), DType::Float64);
    }
}
