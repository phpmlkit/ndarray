//! Macro to wrap FFI functions with error handling and panic catching.
//!
//! Catches panics (e.g. from ndarray's unwrap on shape/index errors) and
//! classifies them into the appropriate error code so PHP can throw
//! the correct exception type. Suppresses the default panic output to stderr
//! since we convert panics to error codes.

use crate::helpers::error::{ERR_DTYPE, ERR_GENERIC, ERR_INDEX, ERR_MATH, ERR_PANIC, ERR_SHAPE};

/// Extract a string from a panic payload (Box<dyn Any + Send>).
pub fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
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
pub fn classify_panic_message(msg: &str) -> (i32, String) {
    let msg_lower = msg.to_lowercase();

    // ndarray ShapeError variants
    if msg_lower.contains("incompatibleshape")
        || msg_lower.contains("incompatible shape")
        || msg_lower.contains("incompatible shapes")
        || msg_lower.contains("incompatiblelayout")
        || msg_lower.contains("incompatible memory layout")
        || msg_lower.contains("rangelimited")
        || msg_lower.contains("shape does not fit")
        || msg_lower.contains("dimension mismatch")
        || msg_lower.contains("shape mismatch")
        || msg_lower.contains("are not compatible for matrix multiplication")
        || msg_lower.contains("overflows isize")
        || msg_lower.contains("ndarray: inputs")
    {
        return (ERR_SHAPE, msg.to_string());
    }

    // ndarray OutOfBounds
    if msg_lower.contains("outofbounds")
        || msg_lower.contains("out of bounds")
        || msg_lower.contains("index out of bounds")
        || msg_lower.contains("slice begin")
        || msg_lower.contains("past end of axis")
        || msg_lower.contains("out of range")
    {
        return (ERR_INDEX, msg.to_string());
    }

    // ndarray Overflow, division, NaN
    if msg_lower.contains("overflow")
        || msg_lower.contains("arithmetic overflow")
        || msg_lower.contains("division by zero")
        || msg_lower.contains("attempt to divide")
    {
        return (ERR_MATH, msg.to_string());
    }

    // Cast/type errors
    if msg_lower.contains("failed to cast")
        || msg_lower.contains("cannot cast")
        || msg_lower.contains("type mismatch")
        || msg_lower.contains("dtype")
        || msg_lower.contains("not supported for")
    {
        return (ERR_DTYPE, msg.to_string());
    }

    // Unsupported (ndarray ErrorKind::Unsupported)
    if msg_lower.contains("unsupported") {
        return (ERR_GENERIC, "unsupported operation".to_string());
    }

    // Fallback: generic panic
    (ERR_PANIC, "Rust panic occurred".to_string())
}

#[macro_export]
macro_rules! ffi_guard {
    ($body:block) => {{
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $body));
        let _ = std::panic::take_hook();
        std::panic::set_hook(prev_hook);
        match result {
            Ok(res) => res,
            Err(payload) => {
                let msg = $crate::macros::ffi_guard::panic_payload_to_string(payload);
                let (code, display_msg) = $crate::macros::ffi_guard::classify_panic_message(&msg);
                $crate::helpers::error::set_last_error(display_msg);
                code
            }
        }
    }};
}
