# Signal Processing

Discrete transforms for frequency-domain analysis (FFT, real FFT, and DCT).

All methods on this page are instance methods on `NDArray`.

::: tip Normalization
Transforms accept a `Normalization` enum (`backward`, `ortho`, `forward`, `none`). Default: `Normalization::Backward`.
:::

::: warning Unnormalized transforms (`Normalization::None`)
`Normalization::None` returns the raw backend scaling. This is useful for low-level work, but it means round-trips like
`$x->fft(norm: Normalization::None)->ifft(norm: Normalization::None)` will generally return a scaled version of `x` (not `x` itself).
:::

---

## fft()

```php
public function fft(?int $n = null, int $axis = -1, Normalization $norm = Normalization::Backward): NDArray
```

Complex discrete Fourier transform along one axis. Real inputs are promoted to a complex dtype.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$n` | `int\|null` | Transform length along `axis`. If null, uses the current axis length. |
| `$axis` | `int` | Axis index (negative indices allowed). Default: `-1` (last axis). |
| `$norm` | `Normalization` | Normalization mode. Default: `Normalization::Backward`. |

### Returns

- `NDArray` - Complex array (`Complex64` or `Complex128`).

### Examples

```php
use PhpMlKit\NDArray\NDArray;
use PhpMlKit\NDArray\Normalization;

$x = NDArray::array([0.0, 1.0, 2.0, 3.0]);
$X = $x->fft(norm: Normalization::Backward);
```

---

## ifft()

```php
public function ifft(?int $n = null, int $axis = -1, Normalization $norm = Normalization::Backward): NDArray
```

Inverse complex FFT along one axis.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$n` | `int\|null` | Transform length along `axis`. If null, uses the current axis length. |
| `$axis` | `int` | Axis index (negative indices allowed). Default: `-1`. |
| `$norm` | `Normalization` | Normalization mode. Default: `Normalization::Backward`. |

### Returns

- `NDArray` - Complex array.

### Examples

```php
use PhpMlKit\NDArray\Normalization;

$x = NDArray::array([0.0, 1.0, 2.0, 3.0]);
$X = $x->fft(norm: Normalization::Backward);
$x2 = $X->ifft(norm: Normalization::Backward);
```

---

## fftn()

```php
public function fftn(?array $axes = null, Normalization $norm = Normalization::Backward): NDArray
```

N-dimensional complex FFT. If `axes` is null or empty, transforms all axes in order.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$axes` | `array<int>\|null` | Axes to transform (negative indices allowed). Null/empty means all axes. |
| `$norm` | `Normalization` | Normalization mode. Default: `Normalization::Backward`. |

### Returns

- `NDArray` - Complex array.

### Examples

```php
use PhpMlKit\NDArray\Normalization;

$a = NDArray::array([[1.0, 0.0], [0.0, 0.0]]);
$F = $a->fftn(norm: Normalization::Backward);
```

---

## ifftn()

```php
public function ifftn(?array $axes = null, Normalization $norm = Normalization::Backward): NDArray
```

N-dimensional inverse complex FFT.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$axes` | `array<int>\|null` | Axes to transform (negative indices allowed). Null/empty means all axes. |
| `$norm` | `Normalization` | Normalization mode. Default: `Normalization::Backward`. |

### Returns

- `NDArray` - Complex array.

---

## fft2()

```php
public function fft2(Normalization $norm = Normalization::Backward): NDArray
```

2-D FFT on the last two axes. Requires `ndim >= 2`.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$norm` | `Normalization` | Normalization mode. Default: `Normalization::Backward`. |

### Returns

- `NDArray` - Complex array.

---

## ifft2()

```php
public function ifft2(Normalization $norm = Normalization::Backward): NDArray
```

2-D inverse FFT on the last two axes. Requires `ndim >= 2`.

---

## rfft()

```php
public function rfft(?int $n = null, int $axis = -1, Normalization $norm = Normalization::Backward): NDArray
```

Real-input FFT along one axis. The output is complex and has length `n//2 + 1` along the transformed axis.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$n` | `int\|null` | Real transform length along `axis`. If null, uses current axis length. |
| `$axis` | `int` | Axis index (negative indices allowed). Default: `-1`. |
| `$norm` | `Normalization` | Normalization mode. Default: `Normalization::Backward`. |

### Returns

- `NDArray` - Complex array.

---

## irfft()

```php
public function irfft(?int $n = null, int $axis = -1, Normalization $norm = Normalization::Backward): NDArray
```

Inverse real FFT: complex Hermitian spectrum → real signal.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$n` | `int\|null` | Real output length along `axis`. If null, it is inferred from the spectrum length. |
| `$axis` | `int` | Axis index (negative indices allowed). Default: `-1`. |
| `$norm` | `Normalization` | Normalization mode. Default: `Normalization::Backward`. |

### Returns

- `NDArray` - Real array (`Float32` or `Float64`).

::: tip Length inference
If you want an exact output length, pass `$n`. When omitted, `irfft()` infers a length from the frequency axis size.
:::

---

## dct()

```php
public function dct(int $type = 2, ?int $n = null, int $axis = -1, Normalization $norm = Normalization::Backward): NDArray
```

Discrete cosine transform (types I–IV) along one axis. This method only supports real inputs.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$type` | `int` | DCT type: `1`, `2`, `3`, or `4`. Default: `2`. |
| `$n` | `int\|null` | Transform length along `axis`. If null, uses current axis length. |
| `$axis` | `int` | Axis index (negative indices allowed). Default: `-1`. |
| `$norm` | `Normalization` | Normalization mode. Default: `Normalization::Backward`. |

### Returns

- `NDArray` - Real array with the same float dtype as the input (`Float32` or `Float64`).

### Raises

- `InvalidArgumentException` - If `$type` is not in `1..4`.
- `DTypeException` - If the input is complex.

### Examples

```php
use PhpMlKit\NDArray\Normalization;

$x = NDArray::array([1.0, 2.0, 3.0, 4.0]);
$y = $x->dct(2, norm: Normalization::Backward);
```

---

## idct()

```php
public function idct(int $type = 2, ?int $n = null, int $axis = -1, Normalization $norm = Normalization::Backward): NDArray
```

Inverse DCT along one axis. For reconstructing norms (`backward`, `ortho`, `forward`), pairing with the same `type` and `norm` reconstructs the original signal:

```php
$x2 = $x->dct(2, norm: Normalization::Ortho)->idct(2, norm: Normalization::Ortho);
```

---

## dctn()

```php
public function dctn(?array $axes = null, int $type = 2, Normalization $norm = Normalization::Backward): NDArray
```

N-dimensional DCT. If `axes` is null or empty, transforms all axes in order.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$axes` | `array<int>\|null` | Axes to transform (negative indices allowed). Null/empty means all axes. |
| `$type` | `int` | DCT type (`1..4`). Default: `2`. |
| `$norm` | `Normalization` | Normalization mode. Default: `Normalization::Backward`. |

---

## idctn()

```php
public function idctn(?array $axes = null, int $type = 2, Normalization $norm = Normalization::Backward): NDArray
```

N-dimensional inverse DCT.

---

## dct2() / idct2()

```php
public function dct2(int $type = 2, Normalization $norm = Normalization::Backward): NDArray
public function idct2(int $type = 2, Normalization $norm = Normalization::Backward): NDArray
```

2-D DCT on the last two axes (and inverse). Requires `ndim >= 2`.

