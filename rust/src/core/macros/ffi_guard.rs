//! Macro to wrap FFI functions with error handling and panic catching.
//!
//! Catches panics (e.g. from ndarray's unwrap on shape/index errors) and
//! classifies them into the appropriate error code so PHP can throw the
//! correct exception type. Suppresses the default panic output to stderr
//! since we convert panics to error codes.

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
                let msg = $crate::core::error::panic_payload_to_string(payload);
                let (code, display_msg) = $crate::core::error::classify_panic_message(&msg);
                $crate::core::error::set_last_error(display_msg);
                code
            }
        }
    }};
}
