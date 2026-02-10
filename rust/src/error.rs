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
#[macro_export]
macro_rules! ffi_guard {
    ($body:block) => {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $body)) {
            Ok(result) => result,
            Err(_) => {
                $crate::error::set_last_error("Rust panic occurred");
                $crate::error::ERR_PANIC
            }
        }
    };
}
