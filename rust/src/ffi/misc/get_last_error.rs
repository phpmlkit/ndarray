//! FFI function to retrieve the last error message.

use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr;

use crate::helpers::error::get_last_error_message;

/// Get the last error message.
///
/// Writes the message to `buf` up to `len` bytes.
/// Returns the actual length of the message.
#[no_mangle]
pub unsafe extern "C" fn ndarray_get_last_error(buf: *mut c_char, len: usize) -> usize {
    if buf.is_null() || len == 0 {
        return 0;
    }

    if let Some(msg) = get_last_error_message() {
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
}
