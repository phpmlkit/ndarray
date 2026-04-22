# Window Functions

Reference for common window functions used in signal processing (tapering) before spectral analysis.

All functions on this page are **static methods** on `NDArray` and return a `Float64` array of shape `[m]`.

::: tip Symmetric vs periodic
All window functions accept a `$periodic` flag:

- `periodic = false` (default): returns a **symmetric** window (useful for filter design)
- `periodic = true`: returns a **periodic** window (often used for FFT / spectral analysis)
:::

---

## bartlett()

```php
public static function bartlett(int $m, bool $periodic = false): NDArray
```

Bartlett (triangular) window with endpoints at zero for `m > 1`. Often used to taper a signal without introducing strong discontinuities at the boundaries.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$m` | `int` | Number of points in the output window. If `0`, an empty array is returned. If `1`, returns `[1.0]`. |
| `$periodic` | `bool` | If true, returns the periodic variant. Default: `false`. |

### Returns

- `NDArray` - Float64 window of shape `[m]`.

### Notes

For `m > 1`, the Bartlett window is a piecewise-linear triangle that reaches `1.0` in the center and tapers to `0.0` at both ends.

---

## triang()

```php
public static function triang(int $m, bool $periodic = false): NDArray
```

Triangular window. Similar to Bartlett, but does not necessarily force the endpoints to zero.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$m` | `int` | Number of points in the output window. If `0`, an empty array is returned. If `1`, returns `[1.0]`. |
| `$periodic` | `bool` | If true, returns the periodic variant. Default: `false`. |

### Returns

- `NDArray` - Float64 window of shape `[m]`.

---

## blackman()

```php
public static function blackman(int $m, bool $periodic = false): NDArray
```

Blackman window, formed using the first three terms of a cosine series. Designed for low spectral leakage (better sidelobe suppression than Hann/Hamming at the expense of a wider main lobe).

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$m` | `int` | Number of points in the output window. If `0`, an empty array is returned. If `1`, returns `[1.0]`. |
| `$periodic` | `bool` | If true, returns the periodic variant. Default: `false`. |

### Returns

- `NDArray` - Float64 window of shape `[m]`.

### Formula

For `m > 1` and `0 ≤ n ≤ m-1`:

\\[
w(n) = 0.42 - 0.5\\cos\\left(\\frac{2\\pi n}{m-1}\\right) + 0.08\\cos\\left(\\frac{4\\pi n}{m-1}\\right)
\\]

---

## hamming()

```php
public static function hamming(int $m, bool $periodic = false): NDArray
```

Hamming window, a raised cosine with non-zero endpoints. Commonly used to reduce the nearest sidelobe levels compared to the Hann window.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$m` | `int` | Number of points in the output window. If `0`, an empty array is returned. If `1`, returns `[1.0]`. |
| `$periodic` | `bool` | If true, returns the periodic variant. Default: `false`. |

### Returns

- `NDArray` - Float64 window of shape `[m]`.

### Formula

For `m > 1` and `0 ≤ n ≤ m-1`:

\\[
w(n) = 0.54 - 0.46\\cos\\left(\\frac{2\\pi n}{m-1}\\right)
\\]

---

## hanning() / hann()

```php
public static function hanning(int $m, bool $periodic = false): NDArray
public static function hann(int $m, bool $periodic = false): NDArray
```

Hanning (Hann) window, a raised cosine with endpoints at zero for `m > 1`. Useful for smoothing discontinuities at the beginning and end of a signal segment.

`hann()` is an alias of `hanning()`.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$m` | `int` | Number of points in the output window. If `0`, an empty array is returned. If `1`, returns `[1.0]`. |
| `$periodic` | `bool` | If true, returns the periodic variant. Default: `false`. |

### Returns

- `NDArray` - Float64 window of shape `[m]`.

### Formula

For `m > 1` and `0 ≤ n ≤ m-1`:

\\[
w(n) = 0.5 - 0.5\\cos\\left(\\frac{2\\pi n}{m-1}\\right)
\\]

---

## bohman()

```php
public static function bohman(int $m, bool $periodic = false): NDArray
```

Bohman window. A smooth window with good sidelobe behavior derived from convolving two cosine-tapered windows.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$m` | `int` | Number of points in the output window. If `0`, an empty array is returned. If `1`, returns `[1.0]`. |
| `$periodic` | `bool` | If true, returns the periodic variant. Default: `false`. |

### Returns

- `NDArray` - Float64 window of shape `[m]`.

### Formula

Let \(x = \\left|\\frac{2n}{m-1} - 1\\right|\). For `m > 1`:

\\[
w(n) = (1-x)\\cos(\\pi x) + \\frac{\\sin(\\pi x)}{\\pi}
\\]

---

## boxcar()

```php
public static function boxcar(int $m, bool $periodic = false): NDArray
```

Boxcar (rectangular) window. This is the “no windowing” case (all ones).

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$m` | `int` | Number of points in the output window. If `0`, an empty array is returned. If `1`, returns `[1.0]`. |
| `$periodic` | `bool` | If true, returns the periodic variant. Default: `false`. |

### Returns

- `NDArray` - Float64 window of shape `[m]`.

---

## lanczos()

```php
public static function lanczos(int $m, bool $periodic = false): NDArray
```

Lanczos window. Defined using a normalized sinc, it is commonly used in interpolation and resampling contexts.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$m` | `int` | Number of points in the output window. If `0`, an empty array is returned. If `1`, returns `[1.0]`. |
| `$periodic` | `bool` | If true, returns the periodic variant. Default: `false`. |

### Returns

- `NDArray` - Float64 window of shape `[m]`.

### Formula

Let \(x = \\frac{2n}{m-1} - 1\\). For `m > 1`:

\\[
w(n) = \\mathrm{sinc}(x), \\quad \\mathrm{sinc}(0)=1,\\; \\mathrm{sinc}(x)=\\frac{\\sin(\\pi x)}{\\pi x}
\\]

---

## kaiser()

```php
public static function kaiser(int $m, float $beta, bool $periodic = false): NDArray
```

Kaiser window, parameterized by `beta`. Larger `beta` increases sidelobe attenuation (less leakage) at the cost of a wider main lobe.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `$m` | `int` | Number of points in the output window. If `0`, an empty array is returned. If `1`, returns `[1.0]`. |
| `$beta` | `float` | Shape parameter controlling the trade-off between main-lobe width and sidelobe level. |
| `$periodic` | `bool` | If true, returns the periodic variant. Default: `false`. |

### Returns

- `NDArray` - Float64 window of shape `[m]`.

