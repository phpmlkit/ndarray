<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Traits;

use PhpMlKit\NDArray\ArrayMetadata;
use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\Exceptions\NDArrayException;
use PhpMlKit\NDArray\FFI\Lib;
use PhpMlKit\NDArray\NDArray;
use PhpMlKit\NDArray\Normalization;

/**
 * Fourier and cosine transforms (FFT, real FFT, DCT), backed by `ndrustfft` with SciPy-compatible norms.
 */
trait HasFourier
{
    /**
     * Complex discrete Fourier transform along one axis. Real inputs are promoted to complex.
     *
     * @param null|int $n Transform length along `axis` (null = current size)
     */
    public function fft(?int $n = null, int $axis = -1, Normalization $norm = Normalization::Backward): NDArray
    {
        return $this->fourierOp('ndarray_fft', [$axis, $n ?? 0, $norm->toFfi()]);
    }

    /**
     * Inverse complex FFT along one axis (expects complex dtype).
     */
    public function ifft(?int $n = null, int $axis = -1, Normalization $norm = Normalization::Backward): NDArray
    {
        return $this->fourierOp('ndarray_ifft', [$axis, $n ?? 0, $norm->toFfi()]);
    }

    /**
     * N-dimensional complex FFT. `axes` null or empty transforms all axes in order.
     *
     * @param null|array<int> $axes Axis indices (negative indices allowed)
     */
    public function fftn(?array $axes = null, Normalization $norm = Normalization::Backward): NDArray
    {
        if (null === $axes || [] === $axes) {
            return $this->fourierOp('ndarray_fftn', [null, 0, $norm->toFfi()]);
        }

        $axesBuf = Lib::createCArray('int32_t', $axes);

        return $this->fourierOp('ndarray_fftn', [$axesBuf, \count($axes), $norm->toFfi()]);
    }

    /**
     * N-dimensional inverse complex FFT.
     *
     * @param null|array<int> $axes Axis indices (negative indices allowed)
     */
    public function ifftn(?array $axes = null, Normalization $norm = Normalization::Backward): NDArray
    {
        if (null === $axes || [] === $axes) {
            return $this->fourierOp('ndarray_ifftn', [null, 0, $norm->toFfi()]);
        }

        $axesBuf = Lib::createCArray('int32_t', $axes);

        return $this->fourierOp('ndarray_ifftn', [$axesBuf, \count($axes), $norm->toFfi()]);
    }

    /**
     * Real-input FFT along `axis`; result is complex with length `n//2+1` on that axis.
     */
    public function rfft(?int $n = null, int $axis = -1, Normalization $norm = Normalization::Backward): NDArray
    {
        return $this->fourierOp('ndarray_rfft', [$axis, $n ?? 0, $norm->toFfi()]);
    }

    /**
     * Inverse real FFT: Hermitian spectrum → real. `n` is the real length along `axis` (null = inferred).
     */
    public function irfft(?int $n = null, int $axis = -1, Normalization $norm = Normalization::Backward): NDArray
    {
        return $this->fourierOp('ndarray_irfft', [$axis, $n ?? 0, $norm->toFfi()]);
    }

    /**
     * 2-D FFT on the last two axes (requires `ndim >= 2`).
     */
    public function fft2(Normalization $norm = Normalization::Backward): NDArray
    {
        if ($this->ndim() < 2) {
            throw new \InvalidArgumentException('fft2 requires at least two dimensions');
        }

        return $this->fftn([-2, -1], $norm);
    }

    /**
     * 2-D inverse FFT on the last two axes (requires `ndim >= 2`).
     */
    public function ifft2(Normalization $norm = Normalization::Backward): NDArray
    {
        if ($this->ndim() < 2) {
            throw new \InvalidArgumentException('ifft2 requires at least two dimensions');
        }

        return $this->ifftn([-2, -1], $norm);
    }

    /**
     * Real discrete cosine transform along one axis.
     *
     * @param int      $type DCT-I … DCT-IV (`1` … `4`). Default `2` matches SciPy `dct(..., type=2)`.
     * @param null|int $n    Length along `axis` (null = current size)
     */
    public function dct(int $type = 2, ?int $n = null, int $axis = -1, Normalization $norm = Normalization::Backward): NDArray
    {
        $this->assertDctType($type);

        return $this->fourierOp('ndarray_dct', [$axis, $n ?? 0, $type, $norm->toFfi()]);
    }

    /**
     * Inverse DCT along one axis (pairs with {@see self::dct()} for the same `$type`).
     */
    public function idct(int $type = 2, ?int $n = null, int $axis = -1, Normalization $norm = Normalization::Backward): NDArray
    {
        $this->assertDctType($type);

        return $this->fourierOp('ndarray_idct', [$axis, $n ?? 0, $type, $norm->toFfi()]);
    }

    /**
     * N-dimensional DCT. `axes` null or empty applies along every axis in order.
     *
     * @param null|array<int> $axes Axis indices (negative indices allowed)
     */
    public function dctn(?array $axes = null, int $type = 2, Normalization $norm = Normalization::Backward): NDArray
    {
        $this->assertDctType($type);

        if (null === $axes || [] === $axes) {
            return $this->fourierOp('ndarray_dctn', [null, 0, $type, $norm->toFfi()]);
        }

        $axesBuf = Lib::createCArray('int32_t', $axes);

        return $this->fourierOp('ndarray_dctn', [$axesBuf, \count($axes), $type, $norm->toFfi()]);
    }

    /**
     * N-dimensional inverse DCT.
     *
     * @param null|array<int> $axes Axis indices (negative indices allowed)
     */
    public function idctn(?array $axes = null, int $type = 2, Normalization $norm = Normalization::Backward): NDArray
    {
        $this->assertDctType($type);

        if (null === $axes || [] === $axes) {
            return $this->fourierOp('ndarray_idctn', [null, 0, $type, $norm->toFfi()]);
        }

        $axesBuf = Lib::createCArray('int32_t', $axes);

        return $this->fourierOp('ndarray_idctn', [$axesBuf, \count($axes), $type, $norm->toFfi()]);
    }

    /**
     * 2-D DCT on the last two axes (requires `ndim >= 2`).
     */
    public function dct2(int $type = 2, Normalization $norm = Normalization::Backward): NDArray
    {
        if ($this->ndim() < 2) {
            throw new \InvalidArgumentException('dct2 requires at least two dimensions');
        }

        return $this->dctn([-2, -1], $type, $norm);
    }

    /**
     * 2-D inverse DCT on the last two axes (requires `ndim >= 2`).
     */
    public function idct2(int $type = 2, Normalization $norm = Normalization::Backward): NDArray
    {
        if ($this->ndim() < 2) {
            throw new \InvalidArgumentException('idct2 requires at least two dimensions');
        }

        return $this->idctn([-2, -1], $type, $norm);
    }

    private function assertDctType(int $type): void
    {
        if ($type < 1 || $type > 4) {
            throw new \InvalidArgumentException('DCT type must be 1, 2, 3, or 4');
        }
    }

    /**
     * @param array<int, mixed> $midArgs Arguments after `(handle, meta,` and before output pointers
     */
    private function fourierOp(string $func, array $midArgs): NDArray
    {
        $ffi = Lib::get();
        $outHandle = $ffi->new('struct NdArrayHandle*');
        [$outDtypeBuf, $outNdimBuf, $outShapeBuf] = Lib::createOutputMetadataBuffers();
        $meta = $this->meta()->toCData();
        $args = array_merge(
            [$this->handle, Lib::addr($meta)],
            $midArgs,
            [
                Lib::addr($outHandle),
                Lib::addr($outDtypeBuf),
                Lib::addr($outNdimBuf),
                $outShapeBuf,
                Lib::MAX_NDIM,
            ],
        );
        $status = $ffi->{$func}(...$args);

        Lib::checkStatus($status);

        $dtype = DType::tryFrom((int) $outDtypeBuf->cdata);
        if (null === $dtype) {
            throw new NDArrayException('Invalid dtype returned from Rust');
        }

        $ndim = (int) $outNdimBuf->cdata;
        $shape = Lib::extractShapeFromPointer($outShapeBuf, $ndim);

        return new NDArray($outHandle, new ArrayMetadata($shape), $dtype);
    }
}
