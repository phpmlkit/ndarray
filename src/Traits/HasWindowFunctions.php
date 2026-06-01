<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Traits;

use PhpMlKit\NDArray\ArrayMetadata;
use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\FFI\Lib;
use PhpMlKit\NDArray\NDArray;

/**
 * Static window generators for tapering segments before spectral analysis or for filter design.
 *
 * Each method returns a new Float64 array of shape `[m]`. The `$periodic` flag selects the
 * variant: `false` (default) yields a symmetric window (common in filter design); `true` yields
 * a periodic window (common when weighting data for FFT-length analysis).
 */
trait HasWindowFunctions
{
    /**
     * Bartlett (triangular) window: piecewise linear, peaks at 1.0 at the center, 0.0 at both
     * ends for `m > 1`. Tapers a segment with mild boundary rolloff.
     *
     * @param int  $m        Number of samples. `m = 0` → empty array; `m = 1` → `[1.0]`.
     * @param bool $periodic if `true`, use the periodic variant; if `false`, symmetric (default)
     *
     * @return NDArray float64, shape `[m]`
     */
    public static function bartlett(int $m, bool $periodic = false): NDArray
    {
        $lib = Lib::get();
        $outHandle = $lib->new('struct NdArrayHandle*');
        $status = $lib->ndarray_bartlett($m, $periodic, Lib::addr($outHandle));
        $lib->checkStatus($status);

        return new NDArray($outHandle, new ArrayMetadata([$m]), DType::Float64);
    }

    /**
     * Blackman window: first three terms of a cosine sum. Stronger sidelobe suppression than
     * Hann or Hamming, with a wider main lobe.
     *
     * For `m > 1` and `0 ≤ n ≤ m-1`: {@code w(n) = 0.42 - 0.5*cos(2πn/(m-1)) + 0.08*cos(4πn/(m-1))}.
     *
     * @param int  $m        Number of samples. `m = 0` → empty array; `m = 1` → `[1.0]`.
     * @param bool $periodic if `true`, use the periodic variant; if `false`, symmetric (default)
     *
     * @return NDArray float64, shape `[m]`
     */
    public static function blackman(int $m, bool $periodic = false): NDArray
    {
        $lib = Lib::get();
        $outHandle = $lib->new('struct NdArrayHandle*');
        $status = $lib->ndarray_blackman($m, $periodic, Lib::addr($outHandle));
        $lib->checkStatus($status);

        return new NDArray($outHandle, new ArrayMetadata([$m]), DType::Float64);
    }

    /**
     * Bohman window: smooth taper derived from convolving two cosine-tapered segments; favorable
     * sidelobe decay for its smoothness.
     *
     * Let {@code x = |2n/(m-1) - 1|}. For `m > 1`: {@code w(n) = (1-x)*cos(πx) + sin(πx)/π}.
     *
     * @param int  $m        Number of samples. `m = 0` → empty array; `m = 1` → `[1.0]`.
     * @param bool $periodic if `true`, use the periodic variant; if `false`, symmetric (default)
     *
     * @return NDArray float64, shape `[m]`
     */
    public static function bohman(int $m, bool $periodic = false): NDArray
    {
        $lib = Lib::get();
        $outHandle = $lib->new('struct NdArrayHandle*');
        $status = $lib->ndarray_bohman($m, $periodic, Lib::addr($outHandle));
        $lib->checkStatus($status);

        return new NDArray($outHandle, new ArrayMetadata([$m]), DType::Float64);
    }

    /**
     * Boxcar (rectangular) window: all ones — equivalent to no tapering.
     *
     * @param int  $m        Number of samples. `m = 0` → empty array; `m = 1` → `[1.0]`.
     * @param bool $periodic if `true`, use the periodic variant; if `false`, symmetric (default)
     *
     * @return NDArray float64, shape `[m]`
     */
    public static function boxcar(int $m, bool $periodic = false): NDArray
    {
        $lib = Lib::get();
        $outHandle = $lib->new('struct NdArrayHandle*');
        $status = $lib->ndarray_boxcar($m, $periodic, Lib::addr($outHandle));
        $lib->checkStatus($status);

        return new NDArray($outHandle, new ArrayMetadata([$m]), DType::Float64);
    }

    /**
     * Hamming window: raised cosine with non-zero endpoints; often lowers nearest sidelobes
     * compared to Hann.
     *
     * For `m > 1` and `0 ≤ n ≤ m-1`: {@code w(n) = 0.54 - 0.46*cos(2πn/(m-1))}.
     *
     * @param int  $m        Number of samples. `m = 0` → empty array; `m = 1` → `[1.0]`.
     * @param bool $periodic if `true`, use the periodic variant; if `false`, symmetric (default)
     *
     * @return NDArray float64, shape `[m]`
     */
    public static function hamming(int $m, bool $periodic = false): NDArray
    {
        $lib = Lib::get();
        $outHandle = $lib->new('struct NdArrayHandle*');
        $status = $lib->ndarray_hamming($m, $periodic, Lib::addr($outHandle));
        $lib->checkStatus($status);

        return new NDArray($outHandle, new ArrayMetadata([$m]), DType::Float64);
    }

    /**
     * Hanning (Hann) window: raised cosine with zeros at both ends for `m > 1`; smooths
     * discontinuities at segment boundaries.
     *
     * For `m > 1` and `0 ≤ n ≤ m-1`: {@code w(n) = 0.5 - 0.5*cos(2πn/(m-1))}.
     *
     * @param int  $m        Number of samples. `m = 0` → empty array; `m = 1` → `[1.0]`.
     * @param bool $periodic if `true`, use the periodic variant; if `false`, symmetric (default)
     *
     * @return NDArray float64, shape `[m]`
     *
     * @see self::hann()
     */
    public static function hanning(int $m, bool $periodic = false): NDArray
    {
        $lib = Lib::get();
        $outHandle = $lib->new('struct NdArrayHandle*');
        $status = $lib->ndarray_hanning($m, $periodic, Lib::addr($outHandle));
        $lib->checkStatus($status);

        return new NDArray($outHandle, new ArrayMetadata([$m]), DType::Float64);
    }

    /**
     * Alias of {@see self::hanning()}: identical Hann window.
     *
     * @param int  $m        Number of samples. `m = 0` → empty array; `m = 1` → `[1.0]`.
     * @param bool $periodic if `true`, use the periodic variant; if `false`, symmetric (default)
     *
     * @return NDArray float64, shape `[m]`
     */
    public static function hann(int $m, bool $periodic = false): NDArray
    {
        return self::hanning($m, $periodic);
    }

    /**
     * Kaiser window: parameterized by `beta`. Larger `beta` improves sidelobe attenuation
     * (less spectral leakage) at the cost of a wider main lobe.
     *
     * @param int   $m        Number of samples. `m = 0` → empty array; `m = 1` → `[1.0]`.
     * @param float $beta     shape parameter controlling main-lobe width vs sidelobe level
     * @param bool  $periodic if `true`, use the periodic variant; if `false`, symmetric (default)
     *
     * @return NDArray float64, shape `[m]`
     */
    public static function kaiser(int $m, float $beta, bool $periodic = false): NDArray
    {
        $lib = Lib::get();
        $outHandle = $lib->new('struct NdArrayHandle*');
        $status = $lib->ndarray_kaiser($m, $beta, $periodic, Lib::addr($outHandle));
        $lib->checkStatus($status);

        return new NDArray($outHandle, new ArrayMetadata([$m]), DType::Float64);
    }

    /**
     * Lanczos window: built from a normalized sinc; common in interpolation and resampling.
     *
     * Let {@code x = 2n/(m-1) - 1}. For `m > 1`: {@code w(n) = sinc(x)} with {@code sinc(0)=1},
     * {@code sinc(x) = sin(πx)/(πx)} for {@code x ≠ 0}.
     *
     * @param int  $m        Number of samples. `m = 0` → empty array; `m = 1` → `[1.0]`.
     * @param bool $periodic if `true`, use the periodic variant; if `false`, symmetric (default)
     *
     * @return NDArray float64, shape `[m]`
     */
    public static function lanczos(int $m, bool $periodic = false): NDArray
    {
        $lib = Lib::get();
        $outHandle = $lib->new('struct NdArrayHandle*');
        $status = $lib->ndarray_lanczos($m, $periodic, Lib::addr($outHandle));
        $lib->checkStatus($status);

        return new NDArray($outHandle, new ArrayMetadata([$m]), DType::Float64);
    }

    /**
     * Triangular window: similar in spirit to Bartlett, but endpoint values are not forced to
     * zero in the same way (see Bartlett for the triangle with explicit zero endpoints).
     *
     * @param int  $m        Number of samples. `m = 0` → empty array; `m = 1` → `[1.0]`.
     * @param bool $periodic if `true`, use the periodic variant; if `false`, symmetric (default)
     *
     * @return NDArray float64, shape `[m]`
     */
    public static function triang(int $m, bool $periodic = false): NDArray
    {
        $lib = Lib::get();
        $outHandle = $lib->new('struct NdArrayHandle*');
        $status = $lib->ndarray_triang($m, $periodic, Lib::addr($outHandle));
        $lib->checkStatus($status);

        return new NDArray($outHandle, new ArrayMetadata([$m]), DType::Float64);
    }
}
