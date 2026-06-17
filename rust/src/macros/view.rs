//! Macros to generate array extraction functions for each type.

/// Array extraction for each type.
#[macro_export]
macro_rules! define_extract_array {
    ($name:ident, $variant:path, $type:ty) => {
        pub unsafe fn $name(
            wrapper: &crate::types::NDArrayWrapper,
            meta: &crate::types::ArrayMetadata,
        ) -> Option<ndarray::ArrayD<$type>> {
            let offset = meta.offset;
            let shape = meta.shape_slice();
            let strides = meta.strides_slice();
            let total: usize = shape.iter().product();

            match &wrapper.data {
                $variant(arr) => {
                    let guard = arr.read();
                    let base = guard.as_ptr();
                    let ptr = base.add(offset);
                    let shape_ix = ndarray::IxDyn(shape);

                    if total == 0 {
                        return ndarray::ArrayD::from_shape_vec(shape_ix, Vec::new()).ok();
                    }

                    if $crate::helpers::is_c_contiguous(shape, strides) {
                        let data = std::slice::from_raw_parts(ptr, total).to_vec();
                        return ndarray::ArrayD::from_shape_vec(shape_ix, data).ok();
                    }

                    // Custom strides — stride-based iteration via temporary view
                    let strides_ix = ndarray::IxDyn(strides);
                    let view = ndarray::ArrayViewD::<$type>::from_shape_ptr(
                        shape_ix.strides(strides_ix),
                        ptr,
                    );
                    let data: Vec<$type> = view.iter().copied().collect();
                    ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(shape), data).ok()
                }
                _ => None,
            }
        }
    };
}

/// Immutable view extraction for each type.
///
/// Returns a zero-copy `ArrayViewD` when the data variant matches.
/// **Does not** check contiguity — the caller must gate on `is_c_contiguous()` first.
#[macro_export]
macro_rules! define_extract_view {
    ($name:ident, $variant:path, $type:ty) => {
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

/// Mutable view extraction for each type.
#[macro_export]
macro_rules! define_extract_view_mut {
    ($name:ident, $variant:path, $type:ty) => {
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

/// Generate `extract_array_as` functions
#[macro_export]
macro_rules! define_extract_array_as {
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
                if let Some(data) = $native_fn(wrapper, meta) {
                    return Some(data);
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
