# FFI Interoperability

Working with external libraries through PHP's FFI interface.

## Overview

NDArray provides methods for exchanging data with external C libraries and binary formats:

- **`fromBuffer()`** — Import data from an external C buffer into NDArray
- **`toBuffer()`** — Export NDArray data to a C buffer (allocates if needed)
- **`fromBytes()`** — Import data from a binary string (little-endian)
- **`toBytes()`** — Export NDArray data to a binary string (little-endian)

These methods are useful when integrating with specialized libraries for audio processing, image I/O, hardware interfaces, file serialization, or any scenario where data needs to cross the PHP/C boundary efficiently.

## When to Use FFI Interop

Use these methods when:

- Loading data from audio/image libraries that return C pointers
- Reading binary files containing raw array data
- Interfacing with hardware or scientific equipment
- Passing data to/from other Rust/C libraries
- Serializing arrays to binary format for storage or network transfer
- Performance is critical and PHP array overhead is unacceptable

## fromBuffer: Importing from C Buffers

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

// Now use $audio normally
$mean = $audio->mean();
```

### Important Considerations

**Type Safety**: The `dtype` parameter must match the buffer's actual C type exactly. No conversion is performed—raw memory is copied byte-for-byte.

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

**Buffer Lifetime**: The buffer must remain valid during the `fromBuffer()` call. After the call completes, you're free to manage the buffer as needed—NDArray owns a copy.

## toBuffer: Exporting to C Buffers

Export NDArray data to a C buffer for use with external libraries.

```php
// Example: Writing audio to libsndfile
$audio = NDArray::random([44100, 2], DType::Float32);

// Option 1: Let NDArray allocate the buffer
$buffer = $audio->toBuffer();
// Pass buffer to external library
$sndfile->sf_writef_float($file, $buffer, 44100);

// Option 2: Use a pre-allocated buffer
$ffi = FFI::cdef("");
$existingBuffer = $ffi->new('float[88200]');
$audio->toBuffer($existingBuffer);
```

### Partial Exports

Export only a portion of the array using the `$start` and `$len` parameters:

```php
$large = NDArray::random([1000000], DType::Float32);

// Export first 1000 elements (allocate new buffer)
$buffer = $large->toBuffer(start: 0, len: 1000);

// Export from middle of array (start at 5000, copy 1000)
$buffer = $large->toBuffer(start: 5000, len: 1000);

// Export from offset to end (start at 5000, copy remaining)
$buffer = $large->toBuffer(start: 5000);
```

**Parameter defaults:**
- `$buffer` defaults to `null` (allocate new buffer)
- `$start` defaults to `0` (beginning of array)
- `$len` defaults to `null` (copy all remaining elements from start to end)

### Buffer Ownership

When `toBuffer()` allocates a buffer (no `$buffer` argument), the returned `CData` is managed by PHP's FFI. It will be garbage collected when no longer referenced. Copy the data immediately if you need it to persist beyond the current scope.

## fromBytes: Importing Binary Strings

Create an NDArray from a binary string. The data is interpreted in little-endian format.

```php
// Read binary file
$binaryData = file_get_contents('audio.raw');

// Create array from bytes
$audio = NDArray::fromBytes($binaryData, [44100, 2], DType::Float32);
```

### Byte Order

`fromBytes()` assumes **little-endian** byte order. This is the native format for most modern systems (x86, x86_64, ARM). If you're working with big-endian data, you'll need to convert it first.

### Size Validation

`fromBytes()` validates that the string length matches the expected size:

```php
// Good: 24 bytes = 3 Float64 elements
$bytes = str_repeat('\x00', 24);
$arr = NDArray::fromBytes($bytes, [3], DType::Float64);

// Bad: Throws ShapeException
$bytes = str_repeat('\x00', 20);  // Only 20 bytes
$arr = NDArray::fromBytes($bytes, [3], DType::Float64);  // Expects 24 bytes
```

## toBytes: Exporting to Binary Strings

Export NDArray data as a binary string in little-endian format.

```php
$arr = NDArray::array([1.0, 2.0, 3.0], DType::Float64);
$bytes = $arr->toBytes();

// Save to file
file_put_contents('data.bin', $bytes);

// Send over network
socket_write($socket, $bytes);
```

### Use Cases

- **File I/O**: Save/load arrays in binary format
- **Network protocols**: Send array data over sockets
- **Inter-process communication**: Share data with other processes
- **Caching**: Store serialized arrays for later use

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

### Byte Order Issues

```php
// Reading big-endian data on little-endian system
$bigEndianBytes = file_get_contents('big_endian_data.bin');

// WRONG: Direct import assumes little-endian
$arr = NDArray::fromBytes($bigEndianBytes, ...);

// RIGHT: Convert byte order first
$littleEndianBytes = '';  // Swap bytes appropriately
// ... conversion logic ...
$arr = NDArray::fromBytes($littleEndianBytes, ...);
```

## Summary

| Method | Input | Output | Use Case |
|--------|-------|--------|----------|
| `fromBuffer()` | C pointer (`CData`) | NDArray | Import from C libraries |
| `toBuffer()` | NDArray | C pointer (`CData`) | Export to C libraries |
| `fromBytes()` | Binary string | NDArray | Load from files/sockets |
| `toBytes()` | NDArray | Binary string | Save to files/sockets |

## See Also

- **[FFI Internals](/guide/advanced/ffi-internals)** — How NDArray uses FFI
- **[Array Creation API](/api/array-creation)** — `fromBuffer()` and `fromBytes()` reference
- **[Array Import/Export API](/api/array-import-export)** — `toBuffer()` and `toBytes()` reference
- **[Performance Guide](/guide/advanced/performance)** — Optimization strategies
