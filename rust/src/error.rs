use std::cell::RefCell;
use std::ffi::CString;
use std::fmt::Display;
use std::os::raw::c_char;
use std::ptr;

// Thread-local storage for the last error message
thread_local! {
    static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

// Error codes
pub const SUCCESS: i32 = 0;
pub const ERR_GENERIC: i32 = 1;
pub const ERR_SHAPE: i32 = 2;
pub const ERR_DTYPE: i32 = 3;
pub const ERR_ALLOC: i32 = 4;
pub const ERR_PANIC: i32 = 5;
pub const ERR_INDEX: i32 = 6;
pub const ERR_MATH: i32 = 7;

/// Set the last error message.
pub fn set_last_error<E: Display>(err: E) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = Some(err.to_string());
    });
}

/// Extract a string from a panic payload (Box<dyn Any + Send>).
pub(crate) fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<String>() {
        return s.clone();
    }
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        return (*s).to_string();
    }
    "Rust panic occurred".to_string()
}

/// Classify a panic message and return the appropriate error code and display message.
///
/// Maps ndarray and Rust error patterns to our error codes so PHP can throw
/// the correct exception type (ShapeException, IndexException, etc.).
pub(crate) fn classify_panic_message(msg: &str) -> (i32, String) {
    let msg_lower = msg.to_lowercase();

    // ndarray ShapeError variants (from ndarray/src/error.rs)
    // Display format: "ShapeError/IncompatibleShape: incompatible shapes"
    if msg_lower.contains("incompatibleshape")
        || msg_lower.contains("incompatible shape")
        || msg_lower.contains("incompatible shapes")
        || msg_lower.contains("incompatiblelayout")
        || msg_lower.contains("incompatible memory layout")
        || msg_lower.contains("rangelimited")
        || msg_lower.contains("shape does not fit")
        || msg_lower.contains("dimension mismatch")
        || msg_lower.contains("shape mismatch")
    {
        return (ERR_SHAPE, "incompatible shapes".to_string());
    }

    // ndarray OutOfBounds
    if msg_lower.contains("outofbounds")
        || msg_lower.contains("out of bounds")
        || msg_lower.contains("index out of bounds")
        || msg_lower.contains("slice begin")
        || msg_lower.contains("past end of axis")
        || msg_lower.contains("out of range")
    {
        return (ERR_INDEX, "index out of bounds".to_string());
    }

    // ndarray Overflow, division, NaN
    if msg_lower.contains("overflow")
        || msg_lower.contains("arithmetic overflow")
        || msg_lower.contains("division by zero")
        || msg_lower.contains("attempt to divide")
    {
        return (ERR_MATH, "math error".to_string());
    }

    // Cast/type errors
    if msg_lower.contains("failed to cast")
        || msg_lower.contains("cannot cast")
        || msg_lower.contains("type mismatch")
        || msg_lower.contains("dtype")
        || msg_lower.contains("not supported for")
    {
        return (ERR_DTYPE, "type error".to_string());
    }

    // Unsupported (ndarray ErrorKind::Unsupported)
    if msg_lower.contains("unsupported") {
        return (ERR_GENERIC, "unsupported operation".to_string());
    }

    // Fallback: generic panic
    (ERR_PANIC, "Rust panic occurred".to_string())
}

/// Get the last error message.
///
/// Writes the message to `buf` up to `len` bytes.
/// Returns the actual length of the message.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_last_error(buf: *mut c_char, len: usize) -> usize {
    if buf.is_null() || len == 0 {
        return 0;
    }

    LAST_ERROR.with(|e| {
        if let Some(msg) = &*e.borrow() {
            let c_str = match CString::new(msg.as_str()) {
                Ok(s) => s,
                Err(_) => return 0,
            };
            let bytes = c_str.as_bytes_with_nul();
            let copy_len = bytes.len().min(len);

            ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, buf, copy_len);

            // Ensure null termination if truncated
            if copy_len == len {
                *buf.add(len - 1) = 0;
            }

            copy_len
        } else {
            0
        }
    })
}

/// Macro to wrap FFI functions with error handling and panic catching.
///
/// Catches panics (e.g. from ndarray's unwrap on shape/index errors) and
/// classifies them into the appropriate error code so PHP can throw the
/// correct exception type. Suppresses the default panic output to stderr
/// since we convert panics to error codes.
#[macro_export]
macro_rules! ffi_guard {
    ($body:block) => {
        {
            let prev_hook = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $body));
            let _ = std::panic::take_hook();
            std::panic::set_hook(prev_hook);
            match result {
                Ok(res) => res,
                Err(payload) => {
                    let msg = $crate::error::panic_payload_to_string(payload);
                    let (code, display_msg) = $crate::error::classify_panic_message(&msg);
                    $crate::error::set_last_error(display_msg);
                    code
                }
            }
        }
    };
}
