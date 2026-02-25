//! NDArray free function.

use crate::error::SUCCESS;
use crate::ffi::NdArrayHandle;

/// Destroy an NDArray and free its memory.
#[no_mangle]
pub unsafe extern "C" fn ndarray_free(handle: *mut NdArrayHandle) -> i32 {
    crate::ffi_guard!({
        if !handle.is_null() {
            let _ = NdArrayHandle::into_wrapper(handle);
        }
        SUCCESS
    })
}
