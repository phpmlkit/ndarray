# FFI Interoperability

Working with external libraries through PHP's FFI interface.

## Overview

NDArray provides two low-level methods for exchanging data with external C libraries without intermediate PHP arrays:

- **`fromBuffer()`** — Import data from an external C buffer into NDArray
- **`intoBuffer()`** — Export NDArray data to an external C buffer

These methods are useful when integrating with specialized libraries for audio processing, image I/O, hardware interfaces, or any other scenario where data originates from or needs to go to C code.

## When to Use FFI Interop

Use `fromBuffer()` and `intoBuffer()` when:

- Loading data from audio libraries
- Reading images via specialized libraries
- Interfacing with hardware or scientific equipment
- Passing data to/from other Rust/C libraries
- Performance is critical and PHP array overhead is unacceptable

## fromBuffer: Importing External Data

Create an NDArray by copying data from a C buffer allocated by an external library.

```php
// Example: Loading audio with libsndfile
$sndfile = FFI::cdef("
    typedef struct SNDFILE SNDFILE;
    float* sf_readf_float(SNDFILE *file, float *ptr, long frames);
", "libsndfile.dylib");

// Read audio data into C buffer
$buffer = $sndfile->sf_readf_float($file, null, 44100);

// Wrap as NDArray (copies data)
$audio = NDArray::fromBuffer($buffer, [44100, 2], DType::Float32);

// Original buffer can be freed immediately
$sndfile->free($buffer);

// Now use $audio normally
$mean = $audio->mean();
```

### Important Considerations

**Type Safety**: The `dtype` parameter must match the buffer's actual C type exactly. No conversion is performed so raw memory is copied byte-for-byte.

| C Type | DType |
|--------|-------|
| `float*` | `DType::Float32` |
| `double*` | `DType::Float64` |
| `int32_t*` | `DType::Int32` |
| `int64_t*` | `DType::Int64` |

**Size Matching**: The buffer must contain exactly the number of elements implied by the shape. Undefined behavior occurs if the sizes don't match.

```php
// Good: 6 elements in buffer, shape [2, 3] = 6 elements
$arr = NDArray::fromBuffer($buffer, [2, 3], DType::Float32);

// Bad: Buffer has 3 elements, shape [2, 3] expects 6
// Reads garbage memory for remaining 3 elements
```

**Buffer Lifetime**: The buffer must remain valid during the `fromBuffer()` call. After the call completes, you're free to manage the buffer as needed — NDArray owns a copy.

## intoBuffer: Exporting to External Libraries

Copy NDArray data into a C buffer allocated by or for an external library.

```php
// Example: Writing audio to libsndfile
$audio = NDArray::random([44100, 2], DType::Float32);

// Allocate C buffer
$ffi = FFI::cdef("");
$buffer = $ffi->new('float[88200]');

// Copy NDArray data into buffer
$elementsCopied = $audio->intoBuffer($buffer);
// $elementsCopied === 88200

// Pass buffer to external library
$sndfile->sf_writef_float($file, $buffer, 44100);
```

### Partial Exports

Export only a portion of the array using the `$start` and `$len` parameters:

```php
$large = NDArray::random([1000000], DType::Float32);

// Export first 1000 elements (start at 0, copy 1000)
$buffer = $ffi->new('float[1000]');
$elementsCopied = $large->intoBuffer($buffer, 0, 1000);
// $elementsCopied === 1000

// Export from middle of array (start at 5000, copy 1000)
$buffer = $ffi->new('float[1000]');
$elementsCopied = $large->intoBuffer($buffer, 5000, 1000);
// $elementsCopied === 1000

// Export from offset to end (start at 5000, copy remaining)
$buffer = $ffi->new('float[995000]');
$elementsCopied = $large->intoBuffer($buffer, 5000);
// $elementsCopied === 995000 (all elements from index 5000 to end)
```

**Parameter defaults:**
- `$start` defaults to `0` (beginning of array)
- `$len` defaults to `null` (copy all remaining elements from start to end)

## Complete Workflow Example

Audio processing pipeline using libsndfile:

```php
class AudioProcessor {
    private FFI $sndfile;
    
    public function __construct() {
        $this->sndfile = FFI::cdef("
            // libsndfile declarations
        ", "libsndfile.dylib");
    }
    
    public function load(string $path): NDArray {
        $file = $this->sndfile->sf_open($path, ...);
        $info = $this->sndfile->sf_get_info($file);
        
        // Allocate C buffer
        $totalSamples = $info->frames * $info->channels;
        $buffer = $this->sndfile->new("float[$totalSamples]");
        
        // Read audio data
        $this->sndfile->sf_readf_float($file, $buffer, $info->frames);
        $this->sndfile->sf_close($file);
        
        // Convert to NDArray
        $audio = NDArray::fromBuffer(
            $buffer,
            [$info->frames, $info->channels],
            DType::Float32
        );
        
        // Free C buffer
        $this->sndfile->free($buffer);
        
        return $audio;
    }
    
    public function save(NDArray $audio, string $path): void {
        [$frames, $channels] = $audio->shape();
        
        // Allocate C buffer
        $buffer = $this->sndfile->new("float[$audio->size()]");
        
        // Copy data to buffer
        $audio->intoBuffer($buffer);
        
        // Write to file
        $file = $this->sndfile->sf_open($path, ...);
        $this->sndfile->sf_writef_float($file, $buffer, $frames);
        $this->sndfile->sf_close($file);
        
        // Free buffer
        $this->sndfile->free($buffer);
    }
}

// Usage
$processor = new AudioProcessor();
$audio = $processor->load('input.wav');

// Process with NDArray
$normalized = $audio->subtract($audio->mean())->divide($audio->std());

$processor->save($normalized, 'output.wav');
```

## Common Pitfalls

### Type Mismatches

```php
// WRONG: Int buffer with Float dtype
$intBuffer = $ffi->new('int32_t[4]');
$arr = NDArray::fromBuffer($intBuffer, [2, 2], DType::Float32);
// Result: Garbage values (bitwise reinterpretation)

// RIGHT: Type must match
$arr = NDArray::fromBuffer($intBuffer, [2, 2], DType::Int32);
```

### Buffer Size Mismatches

```php
// WRONG: 3 elements in buffer, shape expects 6
$buffer = $ffi->new('float[3]');
$arr = NDArray::fromBuffer($buffer, [2, 3], DType::Float32);
// Reads 3 garbage values from adjacent memory

// RIGHT: Match sizes exactly
$buffer = $ffi->new('float[6]');
$arr = NDArray::fromBuffer($buffer, [2, 3], DType::Float32);
```

### Dangling Pointers

```php
// WRONG: Buffer freed before fromBuffer completes
$buffer = $this->sndfile->read(...);
$this->sndfile->free($buffer);  // Too early!
$arr = NDArray::fromBuffer($buffer, ...);  // Segfault

// RIGHT: Keep buffer alive during copy
$buffer = $this->sndfile->read(...);
$arr = NDArray::fromBuffer($buffer, ...);  // Copy happens here
$this->sndfile->free($buffer);  // Safe to free now
```

## See Also

- **[FFI Internals](/guide/advanced/ffi-internals)** — How NDArray uses FFI
- **[Array Creation API](/api/array-creation)** — `fromBuffer()` reference
- **[Array Import/Export API](/api/array-import-export)** — `intoBuffer()` reference
- **[Performance Guide](/guide/advanced/performance)** — Optimization strategies
