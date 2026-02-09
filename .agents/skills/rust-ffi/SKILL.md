---
name: rust-ffi
description: FFI cross-language interop expert covering C/C++ bindings, bindgen, cbindgen, PyO3, JNI, memory layout, data conversion, and safe FFI patterns.
metadata:
  triggers:
    - FFI
    - C interop
    - C++ interop
    - bindgen
    - cbindgen
    - PyO3
    - JNI
    - extern
    - libc
    - CString
    - CStr
    - cross-language
---


## Binding Generation

### C/C++ → Rust (bindgen)

```bash
# Auto-generate bindings
bindgen input.h \
    --output src/bindings.rs \
    --allowlist-type 'my_*' \
    --allowlist-function 'my_*'
```

### Rust → C (cbindgen)

```bash
# Generate C header
cbindgen --crate mylib --output include/mylib.h
```


## Solution Patterns

### Pattern 1: Calling C Functions

```rust
use std::ffi::{CStr, CString};
use libc::c_int;

#[link(name = "curl")]
extern "C" {
    fn curl_version() -> *const libc::c_char;
    fn curl_easy_perform(curl: *mut c_int) -> c_int;
}

// ✅ Safe wrapper
fn get_version() -> String {
    unsafe {
        let ptr = curl_version();
        // SAFETY: curl_version returns valid null-terminated string
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}
```

### Pattern 2: String Passing

```rust
// ✅ Safe way to pass strings
fn process_c_string(s: &CStr) {
    // SAFETY: s is a valid CStr, ptr is valid for call duration
    unsafe {
        some_c_function(s.as_ptr());
    }
}

// Creating CString from Rust
fn get_c_string() -> Result<CString, std::ffi::NulError> {
    CString::new("hello")
}

// ❌ Dangerous: temporary CString
// let ptr = CString::new("hello").unwrap().as_ptr();  // Dangling!

// ✅ Correct: keep CString alive
let c_str = CString::new("hello")?;
let ptr = c_str.as_ptr();
// use ptr...
// c_str dropped here
```

### Pattern 3: Callback Functions

```rust
extern "C" fn callback(data: *mut libc::c_void) {
    // SAFETY: data must be a valid pointer to UserData
    // Caller guarantees this invariant
    unsafe {
        let user_data: &mut UserData = &mut *(data as *mut UserData);
        user_data.count += 1;
    }
}

fn register_callback(callback: extern "C" fn(*mut c_void), data: *mut c_void) {
    unsafe {
        some_c_lib_register(callback, data);
    }
}
```

### Pattern 4: C++ Interop with cxx

```rust
// Using cxx for safe C++ FFI
use cxx::CxxString;

#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("my_library.h");

        type MyClass;

        fn do_something(&self, input: i32) -> i32;
        fn get_data(&self) -> &CxxString;
    }
}

struct RustWrapper {
    inner: cxx::UniquePtr<ffi::MyClass>,
}

impl RustWrapper {
    pub fn new() -> Self {
        Self {
            inner: ffi::create_my_class(),
        }
    }

    pub fn do_something(&self, input: i32) -> i32 {
        self.inner.do_something(input)
    }
}
```


## Data Type Mapping

| Rust | C | Notes |
|------|---|-------|
| `i32` | `int` | Usually matches |
| `i64` | `long long` | Platform-dependent |
| `usize` | `uintptr_t` | Pointer-sized |
| `*const T` | `const T*` | Read-only |
| `*mut T` | `T*` | Mutable |
| `&CStr` | `const char*` | UTF-8 guaranteed |
| `CString` | `char*` | Ownership transfer |
| `NonNull<T>` | `T*` | Non-null pointer |
| `Option<NonNull<T>>` | `T*` (nullable) | Nullable pointer |


## Error Handling

### C Error Codes

```rust
fn call_c_api() -> Result<(), Box<dyn std::error::Error>> {
    // SAFETY: c_function is properly initialized
    let result = unsafe { c_function_that_returns_int() };
    if result < 0 {
        return Err(format!("C API error: {}", result).into());
    }
    Ok(())
}
```

### Panic Across FFI

```rust
// Panics across FFI boundary = UB
// Must catch or prevent

#[no_mangle]
pub extern "C" fn safe_call() -> i32 {
    let result = std::panic::catch_unwind(|| {
        rust_code_that_might_panic()
    });

    match result {
        Ok(value) => value,
        Err(_) => -1,  // Error code
    }
}
```

### C++ Exceptions

```rust
// C++ exceptions → Rust panic (with cxx)
// Must catch at FFI boundary

#[no_mangle]
pub extern "C" fn safe_cpp_call(error_code: *mut i32) -> *const c_char {
    let result = std::panic::catch_unwind(|| {
        unsafe { cpp_function() }
    });

    match result {
        Ok(Ok(value)) => value.as_ptr(),
        Ok(Err(e)) => {
            if !error_code.is_null() {
                unsafe { *error_code = e.code(); }
            }
            std::ptr::null()
        }
        Err(_) => {
            if !error_code.is_null() {
                unsafe { *error_code = -999; }
            }
            std::ptr::null()
        }
    }
}
```


## Memory Management

| Scenario | Who Frees | How |
|----------|-----------|-----|
| C allocates, Rust uses | C | Don't free from Rust |
| Rust allocates, C uses | Rust | C notifies when done |
| Shared buffer | Agreed protocol | Document clearly |

```rust
// ✅ Rust allocates, C borrows
#[no_mangle]
pub extern "C" fn create_buffer(len: usize) -> *mut u8 {
    let mut buf = vec![0u8; len];
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);  // Don't drop
    ptr
}

#[no_mangle]
pub extern "C" fn free_buffer(ptr: *mut u8, len: usize) {
    unsafe {
        // SAFETY: ptr was allocated by create_buffer with this len
        let _ = Vec::from_raw_parts(ptr, len, len);
    }  // Vec dropped, memory freed
}
```


## Workflow

### Step 1: Choose FFI Strategy

```
Need to call C code?
  → Simple functions? Manual extern declarations
  → Complex API? Use bindgen
  → C++? Use cxx crate

Exporting to C?
  → Use cbindgen to generate headers
  → Mark functions #[no_mangle]
  → Use extern "C"
```

### Step 2: Define Safety Invariants

```
For every FFI call:
1. Document pointer validity requirements
2. Document lifetime expectations
3. Document thread safety assumptions
4. Document panic handling
```

### Step 3: Build Safe Wrapper

```
unsafe FFI calls
  ↓
Safe private functions (validate inputs)
  ↓
Safe public API (no unsafe visible)
```

### Step 4: Test Thoroughly

```bash
# Test with Miri
cargo +nightly miri test

# Memory safety check
valgrind ./target/release/program

# Cross-compile test
cargo build --target x86_64-unknown-linux-gnu
```


## Language-Specific Tools

| Language | Tool | Use Case |
|----------|------|----------|
| Python | **PyO3** | Python extensions |
| Java | **jni** | Android/JVM |
| Node.js | **napi-rs** | Node.js addons |
| C# | **csharp-bindgen** | .NET interop |
| Go | **cgo** | Go bridge |
| C++ | **cxx** | Safe C++ FFI |


## Common Pitfalls

| Pitfall | Consequence | Avoid By |
|---------|-------------|----------|
| String encoding error | Garbled text | Use CStr/CString |
| Lifetime mismatch | Use-after-free | Clear ownership |
| Cross-thread non-Send | Data race | Arc + Mutex |
| Fat pointer to C | Memory corruption | Flatten data |
| Missing #[no_mangle] | Symbol not found | Explicit export |
| Panic across FFI | UB | catch_unwind |


## Review Checklist

When reviewing FFI code:

- [ ] All extern functions have SAFETY comments
- [ ] String conversion uses CStr/CString properly
- [ ] Memory ownership is clearly documented
- [ ] No panics across FFI boundary (use catch_unwind)
- [ ] FFI types use #[repr(C)]
- [ ] Raw pointers validated before dereferencing
- [ ] Functions exported with #[no_mangle]
- [ ] Callbacks have correct ABI (extern "C")
- [ ] Tested with Miri for UB detection
- [ ] Documentation explains ownership protocol


## Verification Commands

```bash
# Check safety
cargo +nightly miri test

# Memory leaks
valgrind --leak-check=full ./target/release/program

# Generate bindings
bindgen wrapper.h --output src/ffi.rs

# Generate C header
cbindgen --lang c --output target/mylib.h

# Check exports
nm target/release/libmylib.so | grep my_function
```


## Safety Guidelines

1. **Minimize unsafe**: Only wrap necessary C calls
2. **Defensive programming**: Check null pointers, validate ranges
3. **Clear documentation**: Who owns memory, who frees it
4. **Test coverage**: FFI bugs are extremely hard to debug
5. **Use Miri**: Detect undefined behavior early


## Related Skills

- **rust-unsafe** - Unsafe code fundamentals
- **rust-ownership** - Memory and lifetime management
- **rust-coding** - Export conventions
- **rust-performance** - FFI overhead optimization
- **rust-web** - Using FFI in web services
