# Global functions

Complete reference for namespaced functions that proxy to `NDArray` instance methods and static factories. They mirror the behavior documented on the rest of this API reference.

For full parameter lists, edge cases, and examples, follow the links to the corresponding class methods.

---

## Overview

- Each function delegates to `NDArray`; there is no second implementation in PHP.
- **Binary-style operations** take the receiver array as the **first** argument: `add($a, $b)` is the same as `$a->add($b)`.
- **Canonical list and signatures** always match the current `src/Functions.php` in the repository.

The catalogue is split into [base](#base), [linear algebra](#linalg), [FFT / DCT](#fft), and [windows](#windows) sections.

---

## Importing

Import only what you need with `use function`:

```php
<?php

use function PhpMlKit\NDArray\nd_array;
use function PhpMlKit\NDArray\add;
use function PhpMlKit\NDArray\Linalg\matmul;

$a = nd_array([[1, 2], [3, 4]]);
$b = add($a, $a);
$c = matmul($a, $a);
```

You can combine imports from the base namespace and sub-namespaces in one `use function` block if you prefer.

---

## Base namespace {#base}

### Array creation and factories

| Function | Maps to | See |
|----------|---------|-----|
| `nd_array` | `NDArray::fromArray()` | [Array Creation — fromArray](/api/array-creation#ndarray-fromarray) |
| `zeros` | `NDArray::zeros()` | [Array Creation — zeros](/api/array-creation#ndarray-zeros) |
| `ones` | `NDArray::ones()` | [Array Creation — ones](/api/array-creation#ndarray-ones) |
| `full` | `NDArray::full()` | [Array Creation — full](/api/array-creation#ndarray-full) |
| `from_buffer` | `NDArray::fromBuffer()` | [Array Creation — fromBuffer](/api/array-creation#ndarray-frombuffer) |
| `from_bytes` | `NDArray::fromBytes()` | [Array Creation — fromBytes](/api/array-creation#ndarray-frombytes) |
| `zeros_like` | `NDArray::zerosLike()` | [Array Creation](/api/array-creation) |
| `ones_like` | `NDArray::onesLike()` | [Array Creation](/api/array-creation) |
| `full_like` | `NDArray::fullLike()` | [Array Creation](/api/array-creation) |
| `eye` | `NDArray::eye()` | [Array Creation — eye](/api/array-creation#ndarray-eye) |
| `arange` | `NDArray::arange()` | [Array Creation — arange](/api/array-creation#ndarray-arange) |
| `linspace` | `NDArray::linspace()` | [Array Creation — linspace](/api/array-creation#ndarray-linspace) |
| `logspace` | `NDArray::logspace()` | [Array Creation - logspace](/api/array-creation#ndarray-logspace) |
| `geomspace` | `NDArray::geomspace()` | [Array Creation - geomspace](/api/array-creation#ndarray-geomspace) |
| `meshgrid` | `NDArray::meshgrid()` | [Array Creation - meshgrid](/api/array-creation#ndarray-meshgrid) |
| `random` | `NDArray::random()` | [Array Creation - random](/api/array-creation#ndarray-random) |
| `random_int` | `NDArray::randomInt()` | [Array Creation](/api/array-creation) |
| `randn` | `NDArray::randn()` | [Array Creation](/api/array-creation) |
| `normal` | `NDArray::normal()` | [Array Creation - normal](/api/array-creation#ndarray-normal) |
| `uniform` | `NDArray::uniform()` | [Array Creation - uniform](/api/array-creation#ndarray-uniform) |
| `tile` | `NDArray::tile()` | [Array Manipulation - tile](/api/array-manipulation#tile) |
| `repeat` | `NDArray::repeat()` | [Array Manipulation - repeat](/api/array-manipulation#repeat) |
| `copy` | `$a->copy()` | [Array Manipulation](/api/array-manipulation) |
| `astype` | `$a->astype()` | [Array Manipulation](/api/array-manipulation) |

### Element-wise math and arithmetic

| Function    | Maps to           | See |
|-------------|------------------|-----|
| `add`       | `$a->add()`       | [Mathematical Functions – add](/api/mathematical-functions#add) |
| `subtract`  | `$a->subtract()`  | [Mathematical Functions – subtract](/api/mathematical-functions#subtract) |
| `multiply`  | `$a->multiply()`  | [Mathematical Functions – multiply](/api/mathematical-functions#multiply) |
| `divide`    | `$a->divide()`    | [Mathematical Functions – divide](/api/mathematical-functions#divide) |
| `rem`       | `$a->rem()`       | [Mathematical Functions – rem](/api/mathematical-functions#rem) |
| `mod`       | `$a->mod()`       | [Mathematical Functions – mod](/api/mathematical-functions#mod) |
| `abs`       | `$a->abs()`       | [Mathematical Functions – abs](/api/mathematical-functions#abs) |
| `negative`  | `$a->negative()`  | [Mathematical Functions – negative](/api/mathematical-functions#negative) |
| `real`      | `$a->real()`      | [Mathematical Functions – real](/api/mathematical-functions#real) |
| `imag`      | `$a->imag()`      | [Mathematical Functions – imag](/api/mathematical-functions#imag) |
| `conjugate` | `$a->conjugate()` | [Mathematical Functions – conjugate](/api/mathematical-functions#conjugate) |
| `conj`      | `$a->conj()`      | [Mathematical Functions – conj](/api/mathematical-functions#conj) |
| `iscomplex` | `$a->iscomplex()` | [Mathematical Functions – iscomplex](/api/mathematical-functions#iscomplex) |
| `isreal`    | `$a->isreal()`    | [Mathematical Functions – isreal](/api/mathematical-functions#isreal) |
| `angle`     | `$a->angle()`     | [Mathematical Functions – angle](/api/mathematical-functions#angle) |
| `sqrt`      | `$a->sqrt()`      | [Mathematical Functions – sqrt](/api/mathematical-functions#sqrt) |
| `exp`       | `$a->exp()`       | [Mathematical Functions – exp](/api/mathematical-functions#exp) |
| `log`       | `$a->log()`       | [Mathematical Functions – log](/api/mathematical-functions#log) |
| `ln`        | `$a->ln()`        | [Mathematical Functions – ln](/api/mathematical-functions#ln) |
| `sin`       | `$a->sin()`       | [Mathematical Functions – sin](/api/mathematical-functions#sin) |
| `cos`       | `$a->cos()`       | [Mathematical Functions – cos](/api/mathematical-functions#cos) |
| `tan`       | `$a->tan()`       | [Mathematical Functions – tan](/api/mathematical-functions#tan) |
| `sinh`      | `$a->sinh()`      | [Mathematical Functions – sinh](/api/mathematical-functions#sinh) |
| `cosh`      | `$a->cosh()`      | [Mathematical Functions – cosh](/api/mathematical-functions#cosh) |
| `tanh`      | `$a->tanh()`      | [Mathematical Functions – tanh](/api/mathematical-functions#tanh) |
| `asin`      | `$a->asin()`      | [Mathematical Functions – asin](/api/mathematical-functions#asin) |
| `acos`      | `$a->acos()`      | [Mathematical Functions – acos](/api/mathematical-functions#acos) |
| `atan`      | `$a->atan()`      | [Mathematical Functions – atan](/api/mathematical-functions#atan) |
| `cbrt`      | `$a->cbrt()`      | [Mathematical Functions – cbrt](/api/mathematical-functions#cbrt) |
| `ceil`      | `$a->ceil()`      | [Mathematical Functions – ceil](/api/mathematical-functions#ceil) |
| `exp2`      | `$a->exp2()`      | [Mathematical Functions – exp2](/api/mathematical-functions#exp2) |
| `floor`     | `$a->floor()`     | [Mathematical Functions – floor](/api/mathematical-functions#floor) |
| `log2`      | `$a->log2()`      | [Mathematical Functions – log2](/api/mathematical-functions#log2) |
| `log10`     | `$a->log10()`     | [Mathematical Functions – log10](/api/mathematical-functions#log10) |
| `pow2`      | `$a->pow2()`      | [Mathematical Functions – pow2](/api/mathematical-functions#pow2) |
| `round`     | `$a->round()`     | [Mathematical Functions – round](/api/mathematical-functions#round) |
| `signum`    | `$a->signum()`    | [Mathematical Functions – signum](/api/mathematical-functions#signum) |
| `recip`     | `$a->recip()`     | [Mathematical Functions – recip](/api/mathematical-functions#recip) |
| `ln1p`      | `$a->ln1p()`      | [Mathematical Functions – ln1p](/api/mathematical-functions#ln1p) |
| `to_degrees`| `$a->toDegrees()` | [Mathematical Functions – toDegrees](/api/mathematical-functions#todegrees) |
| `to_radians`| `$a->toRadians()` | [Mathematical Functions – toRadians](/api/mathematical-functions#toradians) |
| `powi`      | `$a->powi()`      | [Mathematical Functions – powi](/api/mathematical-functions#powi) |
| `powf`      | `$a->powf()`      | [Mathematical Functions – powf](/api/mathematical-functions#powf) |
| `hypot`     | `$a->hypot()`     | [Mathematical Functions – hypot](/api/mathematical-functions#hypot) |

### Bitwise and shifts

| Function      | Maps to             | See                                                              |
|---------------|--------------------|------------------------------------------------------------------|
| `bitand`      | `$a->bitand()`     | [Bitwise Operations – bitand](/api/bitwise-operations#bitand)    |
| `bitor`       | `$a->bitor()`      | [Bitwise Operations – bitor](/api/bitwise-operations#bitor)      |
| `bitxor`      | `$a->bitxor()`     | [Bitwise Operations – bitxor](/api/bitwise-operations#bitxor)    |
| `left_shift`  | `$a->leftShift()`  | [Bitwise Operations – left_shift](/api/bitwise-operations#left_shift) |
| `right_shift` | `$a->rightShift()` | [Bitwise Operations – right_shift](/api/bitwise-operations#right_shift) |

### Clipping and extrema

| Function   | Maps to         | See                                                            |
|------------|----------------|----------------------------------------------------------------|
| `clamp`    | `$a->clamp()`  | [Mathematical Functions – clamp](/api/mathematical-functions#clamp)   |
| `clip`     | `$a->clip()`   | [Mathematical Functions – clip](/api/mathematical-functions#clip)     |
| `minimum`  | `$a->minimum()`| [Mathematical Functions – minimum](/api/mathematical-functions#minimum) |
| `maximum`  | `$a->maximum()`| [Mathematical Functions – maximum](/api/mathematical-functions#maximum) |
| `sigmoid`  | `$a->sigmoid()`| [Mathematical Functions – sigmoid](/api/mathematical-functions#sigmoid) |
| `softmax`  | `$a->softmax()`| [Mathematical Functions – softmax](/api/mathematical-functions#softmax) |

### Comparisons

| Function | Maps to       | See                                                        |
|----------|--------------|------------------------------------------------------------|
| `eq`     | `$a->eq()`   | [Logic Functions – eq](/api/logic-functions#eq)            |
| `ne`     | `$a->ne()`   | [Logic Functions – ne](/api/logic-functions#ne)            |
| `gt`     | `$a->gt()`   | [Logic Functions – gt](/api/logic-functions#gt)            |
| `gte`    | `$a->gte()`  | [Logic Functions – gte](/api/logic-functions#gte)          |
| `lt`     | `$a->lt()`   | [Logic Functions – lt](/api/logic-functions#lt)            |
| `lte`    | `$a->lte()`  | [Logic Functions – lte](/api/logic-functions#lte)          |

### Logical (boolean element-wise)

| Function | Maps to       | See                                                    |
|----------|--------------|--------------------------------------------------------|
| `and`    | `$a->and()`  | [Logic Functions – and](/api/logic-functions#and)      |
| `or`     | `$a->or()`   | [Logic Functions – or](/api/logic-functions#or)        |
| `not`    | `$a->not()`  | [Logic Functions – not](/api/logic-functions#not)      |
| `xor`    | `$a->xor()`  | [Logic Functions – xor](/api/logic-functions#xor)      |

### Statistics and reductions

| Function    | Maps to           | See                                                      |
|-------------|------------------|----------------------------------------------------------|
| `sum`       | `$a->sum()`      | [Statistics – sum](/api/statistics#sum)                  |
| `mean`      | `$a->mean()`     | [Statistics – mean](/api/statistics#mean)                |
| `amin`      | `$a->min()`      | [Statistics – min](/api/statistics#min)                  |
| `amax`      | `$a->max()`      | [Statistics – max](/api/statistics#max)                  |
| `argmin`    | `$a->argmin()`   | [Sorting & Searching – argmin](/api/sorting-searching#argmin) |
| `argmax`    | `$a->argmax()`   | [Sorting & Searching – argmax](/api/sorting-searching#argmax) |
| `sort`      | `$a->sort()`     | [Sorting & Searching – sort](/api/sorting-searching#sort)     |
| `argsort`   | `$a->argsort()`  | [Sorting & Searching – argsort](/api/sorting-searching#argsort) |
| `topk`      | `$a->topk()`     | [Sorting & Searching – topk](/api/sorting-searching#topk)     |
| `product`   | `$a->product()`  | [Statistics – product](/api/statistics#product)          |
| `cumsum`    | `$a->cumsum()`   | [Statistics – cumsum](/api/statistics#cumsum)            |
| `cumprod`   | `$a->cumprod()`  | [Statistics – cumprod](/api/statistics#cumprod)          |
| `var`       | `$a->var()`      | [Statistics – var](/api/statistics#var)                  |
| `std`       | `$a->std()`      | [Statistics – std](/api/statistics#std)                  |
| `bincount`  | `$a->bincount()` | [Statistics – bincount](/api/statistics#bincount)        |

### Shape, padding, tiling

| Function       | Maps to            | See                                                                          |
|----------------|-------------------|------------------------------------------------------------------------------|
| `pad`          | `$a->pad()`        | [Array Manipulation – pad](/api/array-manipulation#pad)                      |
| `reshape`      | `$a->reshape()`    | [Array Manipulation – reshape](/api/array-manipulation#reshape)              |
| `transpose`    | `$a->transpose()`  | [Array Manipulation – transpose](/api/array-manipulation#transpose)          |
| `swapaxes`     | `$a->swapaxes()`   | [Array Manipulation – swapaxes](/api/array-manipulation#swapaxes)            |
| `permute`      | `$a->permute()`    | [Array Manipulation – permute](/api/array-manipulation#permute)              |
| `mergeaxes`    | `$a->mergeaxes()`  | [Array Manipulation – mergeaxes](/api/array-manipulation#mergeaxes)          |
| `flip`         | `$a->flip()`       | [Array Manipulation – flip](/api/array-manipulation#flip)                    |
| `insertaxis`   | `$a->insertaxis()` | [Array Manipulation – insertaxis](/api/array-manipulation#insertaxis)        |
| `flatten`      | `$a->flatten()`    | [Array Manipulation – flatten](/api/array-manipulation#flatten)              |
| `ravel`        | `$a->ravel()`      | [Array Manipulation – ravel](/api/array-manipulation#ravel)                  |
| `squeeze`      | `$a->squeeze()`    | [Array Manipulation – squeeze](/api/array-manipulation#squeeze)              |
| `expand_dims`  | `$a->expandDims()` | [Array Manipulation – expandDims](/api/array-manipulation#expanddims)        |
| `tile`         | `$a->tile()`       | [Array Manipulation – tile](/api/array-manipulation#tile)                    |
| `repeat`       | `$a->repeat()`     | [Array Manipulation – repeat](/api/array-manipulation#repeat)                |

### Stacking and splitting

| Function        | Maps to                   | See                                                              |
|-----------------|--------------------------|------------------------------------------------------------------|
| `concatenate`   | `NDArray::concatenate()`  | [Array Manipulation – concatenate](/api/array-manipulation#concatenate) |
| `stack`         | `NDArray::stack()`        | [Array Manipulation – stack](/api/array-manipulation#stack)             |
| `vstack`        | `NDArray::vstack()`       | [Array Manipulation – vstack](/api/array-manipulation#vstack)           |
| `hstack`        | `NDArray::hstack()`       | [Array Manipulation – hstack](/api/array-manipulation#hstack)           |
| `split`         | `$a->split()`             | [Array Manipulation – split](/api/array-manipulation#split)              |
| `vsplit`        | `$a->vsplit()`            | [Array Manipulation – vsplit](/api/array-manipulation#vsplit)            |
| `hsplit`        | `$a->hsplit()`            | [Array Manipulation – hsplit](/api/array-manipulation#hsplit)            |

### Indexing, take/put, selection

| Function           | Maps to                | See                                                         |
|--------------------|-----------------------|-------------------------------------------------------------|
| `get`              | `$a->get()`           | [Indexing Routines – get](/api/indexing-routines#get)                       |
| `set`              | `$a->set()`           | [Indexing Routines – set](/api/indexing-routines#set)                       |
| `set_at`           | `$a->setAt()`         | [Indexing Routines – setAt](/api/indexing-routines#setat)                   |
| `get_at`           | `$a->getAt()`         | [Indexing Routines – getAt](/api/indexing-routines#getat)                   |
| `take`             | `$a->take()`          | [Indexing Routines – take](/api/indexing-routines#take)                     |
| `take_along_axis`  | `$a->takeAlongAxis()` | [Indexing Routines – takeAlongAxis](/api/indexing-routines#takealongaxis)   |
| `put`              | `$a->put()`           | [Indexing Routines – put](/api/indexing-routines#put)                       |
| `put_along_axis`   | `$a->putAlongAxis()`  | [Indexing Routines – putAlongAxis](/api/indexing-routines#putalongaxis)     |
| `scatter_add`      | `$a->scatterAdd()`    | [Indexing Routines – scatterAdd](/api/indexing-routines#scatteradd)         |
| `where`            | `NDArray::where()`    | [Indexing Routines – where](/api/indexing-routines#where)                   |

### Slicing and assignment

| Function   | Maps to         | See                                                   |
|------------|----------------|-------------------------------------------------------|
| `slice`    | `$a->slice()`  | [Indexing Routines – slice](/api/indexing-routines#slice)       |
| `assign`   | `$a->assign()` | [Indexing Routines – assign](/api/indexing-routines#assign)     |

---

## Linear algebra namespace {#linalg}

Import with:

```php
use function PhpMlKit\NDArray\Linalg\norm;
use function PhpMlKit\NDArray\Linalg\matmul;
```

| Function         | Maps to                | See                                                        |
|------------------|-----------------------|------------------------------------------------------------|
| `norm`           | `$a->norm()`           | [Linear Algebra – norm](/api/linear-algebra#norm)          |
| `dot`            | `$a->dot()`            | [Linear Algebra – dot](/api/linear-algebra#dot)            |
| `matmul`         | `$a->matmul()`         | [Linear Algebra – matmul](/api/linear-algebra#matmul)      |
| `diagonal`       | `$a->diagonal()`       | [Linear Algebra – diagonal](/api/linear-algebra#diagonal)  |
| `diag`           | `$a->diag()`           | [Linear Algebra – diag](/api/linear-algebra#diag)          |
| `trace`          | `$a->trace()`          | [Linear Algebra – trace](/api/linear-algebra#trace)        |
| `solve`          | `$a->solve()`          | [Linear Algebra – solve](/api/linear-algebra#solve)        |
| `inv`            | `$a->inv()`            | [Linear Algebra – inv](/api/linear-algebra#inv)            |
| `det`            | `$a->det()`            | [Linear Algebra – det](/api/linear-algebra#det)            |
| `svd`            | `$a->svd()`            | [Linear Algebra – svd](/api/linear-algebra#svd)            |
| `qr`             | `$a->qr()`             | [Linear Algebra – qr](/api/linear-algebra#qr)              |
| `eig`            | `$a->eig()`            | [Linear Algebra – eig](/api/linear-algebra#eig)            |
| `eigvals`        | `$a->eigvals()`        | [Linear Algebra – eigvals](/api/linear-algebra#eigvals)    |
| `eigh`           | `$a->eigh()`           | [Linear Algebra – eigh](/api/linear-algebra#eigh)          |
| `eigvalsh`       | `$a->eigvalsh()`       | [Linear Algebra – eigvalsh](/api/linear-algebra#eigvalsh)  |
| `cholesky`       | `$a->cholesky()`       | [Linear Algebra – cholesky](/api/linear-algebra#cholesky)  |
| `lstsq`          | `$a->lstsq()`          | [Linear Algebra – lstsq](/api/linear-algebra#lstsq)        |
| `least_squares`  | `$a->leastSquares()`   | [Linear Algebra – leastSquares](/api/linear-algebra#leastsquares) |
| `pinv`           | `$a->pinv()`           | [Linear Algebra – pinv](/api/linear-algebra#pinv)          |
| `cond`           | `$a->cond()`           | [Linear Algebra – cond](/api/linear-algebra#cond)          |
| `rank`           | `$a->rank()`           | [Linear Algebra – rank](/api/linear-algebra#rank)          |

---

## FFT and DCT namespace {#fft}

Import with:

```php
use function PhpMlKit\NDArray\Fft\fft;
use function PhpMlKit\NDArray\Fft\ifft;
```

| Function | Maps to        | See                                         |
|----------|---------------|----------------------------------------------|
| `fft`    | `$a->fft()`    | [Signal Processing – fft](/api/signal-processing#fft)       |
| `ifft`   | `$a->ifft()`   | [Signal Processing – ifft](/api/signal-processing#ifft)     |
| `fftn`   | `$a->fftn()`   | [Signal Processing – fftn](/api/signal-processing#fftn)     |
| `ifftn`  | `$a->ifftn()`  | [Signal Processing – ifftn](/api/signal-processing#ifftn)   |
| `rfft`   | `$a->rfft()`   | [Signal Processing – rfft](/api/signal-processing#rfft)     |
| `irfft`  | `$a->irfft()`  | [Signal Processing – irfft](/api/signal-processing#irfft)   |
| `fft2`   | `$a->fft2()`   | [Signal Processing – fft2](/api/signal-processing#fft2)     |
| `ifft2`  | `$a->ifft2()`  | [Signal Processing – ifft2](/api/signal-processing#ifft2)   |
| `dct`    | `$a->dct()`    | [Signal Processing – dct](/api/signal-processing#dct)       |
| `idct`   | `$a->idct()`   | [Signal Processing – idct](/api/signal-processing#idct)     |
| `dctn`   | `$a->dctn()`   | [Signal Processing – dctn](/api/signal-processing#dctn)     |
| `idctn`  | `$a->idctn()`  | [Signal Processing – idctn](/api/signal-processing#idctn)   |
| `dct2`   | `$a->dct2()`   | [Signal Processing – dct2](/api/signal-processing#dct2)     |
| `idct2`  | `$a->idct2()`  | [Signal Processing – idct2](/api/signal-processing#idct2)   |

---

## Window functions namespace {#windows}

Import with:

```php
use function PhpMlKit\NDArray\Windows\hanning;
```

Each function wraps the matching static method on `NDArray` and returns a `Float64` vector of length `$m`.

| Function   | Maps to                | See                                                 |
|------------|-----------------------|-----------------------------------------------------|
| `bartlett` | `NDArray::bartlett()` | [Window Functions – bartlett](/api/window-functions#bartlett) |
| `blackman` | `NDArray::blackman()` | [Window Functions – blackman](/api/window-functions#blackman) |
| `bohman`   | `NDArray::bohman()`   | [Window Functions – bohman](/api/window-functions#bohman)     |
| `boxcar`   | `NDArray::boxcar()`   | [Window Functions – boxcar](/api/window-functions#boxcar)     |
| `hamming`  | `NDArray::hamming()`  | [Window Functions – hamming](/api/window-functions#hamming)   |
| `hanning`  | `NDArray::hanning()`  | [Window Functions – hanning](/api/window-functions#hanning)   |
| `hann`     | `NDArray::hann()`     | [Window Functions – hann](/api/window-functions#hann)         |
| `kaiser`   | `NDArray::kaiser()`   | [Window Functions – kaiser](/api/window-functions#kaiser)     |
| `lanczos`  | `NDArray::lanczos()`  | [Window Functions – lanczos](/api/window-functions#lanczos)   |
| `triang`   | `NDArray::triang()`   | [Window Functions – triang](/api/window-functions#triang)     |

---

## See also

- [API Reference overview](/api/)
- [NDArray class](/api/ndarray-class)
- Source: `src/Functions.php` in the package repository
