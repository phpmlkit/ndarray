//! Macros to generate `extract_view` and `extract_view_as` functions for each type.

/// Macro to generate `extract_view` functions for each type (immutable)
#[macro_export]
macro_rules! define_extract_view {
    ($name:ident, $variant:path, $type:ty) => {
        /// Extract a view of the specific type from the wrapper.
        ///
        /// # Safety
        /// The caller must ensure the ArrayMetadata is valid.
        pub unsafe fn $name<'a>(
            wrapper: &'a crate::types::NDArrayWrapper,
            meta: &'a crate::types::ArrayMetadata,
        ) -> Option<ndarray::ArrayViewD<'a, $type>> {
            let offset = meta.offset;
            let shape = meta.shape_slice();
            let strides = meta.strides_slice();
            match &wrapper.data {
                $variant(arr) => {
                    let guard = arr.read();
                    let ptr = guard.as_ptr();
                    let view_ptr = ptr.add(offset);
                    let strides_ix = ndarray::IxDyn(strides);
                    ndarray::ArrayViewD::<$type>::from_shape_ptr(
                        ndarray::IxDyn(shape).strides(strides_ix),
                        view_ptr,
                    )
                    .into()
                }
                _ => None,
            }
        }
    };
}

/// Macro to generate `extract_view_mut` functions for each type (mutable)
#[macro_export]
macro_rules! define_extract_view_mut {
    ($name:ident, $variant:path, $type:ty) => {
        /// Extract a mutable view of the specific type from the wrapper.
        ///
        /// # Safety
        /// The caller must ensure the ArrayMetadata is valid and that there are
        /// no other readers or writers to this array.
        pub unsafe fn $name<'a>(
            wrapper: &'a crate::types::NDArrayWrapper,
            meta: &'a crate::types::ArrayMetadata,
        ) -> Option<ndarray::ArrayViewMutD<'a, $type>> {
            let offset = meta.offset;
            let shape = meta.shape_slice();
            let strides = meta.strides_slice();
            match &wrapper.data {
                $variant(arr) => {
                    let mut guard = arr.write();
                    let ptr = guard.as_mut_ptr();
                    let view_ptr = ptr.add(offset);
                    let strides_ix = ndarray::IxDyn(strides);
                    ndarray::ArrayViewMutD::<$type>::from_shape_ptr(
                        ndarray::IxDyn(shape).strides(strides_ix),
                        view_ptr,
                    )
                    .into()
                }
                _ => None,
            }
        }
    };
}

/// Macro to generate `extract_view_as` functions.
#[macro_export]
macro_rules! define_extract_view_as {
    (
        $name:ident,
        $target_type:ty,
        $native_fn:ident,
        [$(($fallback_fn:ident, $conv:expr)),+ $(,)?]
    ) => {
        pub fn $name(
            wrapper: &crate::types::NDArrayWrapper,
            meta: &crate::types::ArrayMetadata,
        ) -> Option<ndarray::ArrayD<$target_type>> {
            unsafe {
                if let Some(view) = $native_fn(wrapper, meta) {
                    return Some(view.to_owned());
                }

                $(
                    if let Some(view) = $fallback_fn(wrapper, meta) {
                        return Some(view.mapv($conv));
                    }
                )+
            }
            None
        }
    };
}
