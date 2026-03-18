use std::cell::RefCell;
use std::fmt::Display;

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

/// Get the last error message, if any.
pub fn get_last_error_message() -> Option<String> {
    LAST_ERROR.with(|e| e.borrow().clone())
}
