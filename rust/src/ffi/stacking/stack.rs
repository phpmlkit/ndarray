//! Stack N arrays along a new axis.
//!
//! Accepts handles, offsets, shapes, strides for each array. All arrays must have
//! same dtype and identical shapes.

use ndarray::{stack, Axis};
use parking_lot::RwLock;
use std::slice;
use std::sync::Arc;

use crate::core::view_helpers::{
    extract_view_bool, extract_view_f32, extract_view_f64, extract_view_i16, extract_view_i32,
    extract_view_i64, extract_view_i8, extract_view_u16, extract_view_u32, extract_view_u64,
    extract_view_u8,
};
use crate::core::{ArrayData, NDArrayWrapper};
use crate::dtype::DType;
use crate::error::{set_last_error, ERR_DTYPE, ERR_GENERIC, ERR_SHAPE, SUCCESS};
use crate::ffi::stacking::helpers::resolve_axis_for_stack;
use crate::ffi::{write_output_metadata, NdArrayHandle};

/// Stack N arrays along a new axis.
///
/// handles: pointer to array of num_arrays handles
/// offsets: pointer to array of num_arrays offsets
/// shapes: pointer to num_arrays * ndim values (row-major)
/// strides: pointer to num_arrays * ndim values (same layout)
#[no_mangle]
pub unsafe extern "C" fn ndarray_stack(
    handles: *const *const NdArrayHandle,
    offsets: *const usize,
    shapes: *const usize,
    strides: *const usize,
    num_arrays: usize,
    ndim: usize,
    axis: i32,
    out_handle: *mut *mut NdArrayHandle,
    out_ndim: *mut usize,
    out_shape: *mut usize,
    max_ndim: usize,
) -> i32 {
    if handles.is_null()
        || offsets.is_null()
        || shapes.is_null()
        || strides.is_null()
        || out_handle.is_null()
        || out_shape.is_null()
        || out_ndim.is_null()
        || num_arrays == 0
    {
        return ERR_GENERIC;
    }

    crate::ffi_guard!({
        let handles_slice = slice::from_raw_parts(handles, num_arrays);
        let offsets_slice = slice::from_raw_parts(offsets, num_arrays);

        let first_handle = match handles_slice.get(0) {
            Some(&h) => h,
            None => {
                set_last_error("stack requires at least one array".to_string());
                return ERR_GENERIC;
            }
        };
        let wrapper_0 = NdArrayHandle::as_wrapper(first_handle as *mut _);
        let dtype = wrapper_0.dtype;

        for &h in handles_slice.iter().skip(1) {
            let w = NdArrayHandle::as_wrapper(h as *mut _);
            if w.dtype != dtype {
                set_last_error("stack requires arrays with same dtype".to_string());
                return ERR_DTYPE;
            }
        }

        let shape_0 = slice::from_raw_parts(shapes, ndim);
        let axis_usize = match resolve_axis_for_stack(shape_0, axis) {
            Ok(a) => a,
            Err(e) => {
                set_last_error(e);
                return ERR_SHAPE;
            }
        };

        let result_wrapper = match dtype {
            DType::Float64 => {
                let mut views = Vec::with_capacity(num_arrays);
                for i in 0..num_arrays {
                    let shp = slice::from_raw_parts(shapes.add(i * ndim), ndim);
                    let strd = slice::from_raw_parts(strides.add(i * ndim), ndim);
                    let w = NdArrayHandle::as_wrapper(*handles_slice.get(i).unwrap() as *mut _);
                    let v = match extract_view_f64(w, offsets_slice[i], shp, strd) {
                        Some(v) => v,
                        None => {
                            set_last_error("Failed to extract f64 view".to_string());
                            return ERR_GENERIC;
                        }
                    };
                    views.push(v);
                }
                let arr = match stack(Axis(axis_usize), &views) {
                    Ok(a) => a.into_dyn(),
                    Err(e) => {
                        set_last_error(e.to_string());
                        return ERR_SHAPE;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Float64(Arc::new(RwLock::new(arr))),
                    dtype: DType::Float64,
                }
            }
            DType::Float32 => {
                let mut views = Vec::with_capacity(num_arrays);
                for i in 0..num_arrays {
                    let shp = slice::from_raw_parts(shapes.add(i * ndim), ndim);
                    let strd = slice::from_raw_parts(strides.add(i * ndim), ndim);
                    let w = NdArrayHandle::as_wrapper(*handles_slice.get(i).unwrap() as *mut _);
                    let v = match extract_view_f32(w, offsets_slice[i], shp, strd) {
                        Some(v) => v,
                        None => {
                            set_last_error("Failed to extract f32 view".to_string());
                            return ERR_GENERIC;
                        }
                    };
                    views.push(v);
                }
                let arr = match stack(Axis(axis_usize), &views) {
                    Ok(a) => a.into_dyn(),
                    Err(e) => {
                        set_last_error(e.to_string());
                        return ERR_SHAPE;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Float32(Arc::new(RwLock::new(arr))),
                    dtype: DType::Float32,
                }
            }
            DType::Int64 => {
                let mut views = Vec::with_capacity(num_arrays);
                for i in 0..num_arrays {
                    let shp = slice::from_raw_parts(shapes.add(i * ndim), ndim);
                    let strd = slice::from_raw_parts(strides.add(i * ndim), ndim);
                    let w = NdArrayHandle::as_wrapper(*handles_slice.get(i).unwrap() as *mut _);
                    let v = match extract_view_i64(w, offsets_slice[i], shp, strd) {
                        Some(v) => v,
                        None => {
                            set_last_error("Failed to extract i64 view".to_string());
                            return ERR_GENERIC;
                        }
                    };
                    views.push(v);
                }
                let arr = match stack(Axis(axis_usize), &views) {
                    Ok(a) => a.into_dyn(),
                    Err(e) => {
                        set_last_error(e.to_string());
                        return ERR_SHAPE;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Int64(Arc::new(RwLock::new(arr))),
                    dtype: DType::Int64,
                }
            }
            DType::Int32 => {
                let mut views = Vec::with_capacity(num_arrays);
                for i in 0..num_arrays {
                    let shp = slice::from_raw_parts(shapes.add(i * ndim), ndim);
                    let strd = slice::from_raw_parts(strides.add(i * ndim), ndim);
                    let w = NdArrayHandle::as_wrapper(*handles_slice.get(i).unwrap() as *mut _);
                    let v = match extract_view_i32(w, offsets_slice[i], shp, strd) {
                        Some(v) => v,
                        None => {
                            set_last_error("Failed to extract i32 view".to_string());
                            return ERR_GENERIC;
                        }
                    };
                    views.push(v);
                }
                let arr = match stack(Axis(axis_usize), &views) {
                    Ok(a) => a.into_dyn(),
                    Err(e) => {
                        set_last_error(e.to_string());
                        return ERR_SHAPE;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Int32(Arc::new(RwLock::new(arr))),
                    dtype: DType::Int32,
                }
            }
            DType::Int16 => {
                let mut views = Vec::with_capacity(num_arrays);
                for i in 0..num_arrays {
                    let shp = slice::from_raw_parts(shapes.add(i * ndim), ndim);
                    let strd = slice::from_raw_parts(strides.add(i * ndim), ndim);
                    let w = NdArrayHandle::as_wrapper(*handles_slice.get(i).unwrap() as *mut _);
                    let v = match extract_view_i16(w, offsets_slice[i], shp, strd) {
                        Some(v) => v,
                        None => {
                            set_last_error("Failed to extract i16 view".to_string());
                            return ERR_GENERIC;
                        }
                    };
                    views.push(v);
                }
                let arr = match stack(Axis(axis_usize), &views) {
                    Ok(a) => a.into_dyn(),
                    Err(e) => {
                        set_last_error(e.to_string());
                        return ERR_SHAPE;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Int16(Arc::new(RwLock::new(arr))),
                    dtype: DType::Int16,
                }
            }
            DType::Int8 => {
                let mut views = Vec::with_capacity(num_arrays);
                for i in 0..num_arrays {
                    let shp = slice::from_raw_parts(shapes.add(i * ndim), ndim);
                    let strd = slice::from_raw_parts(strides.add(i * ndim), ndim);
                    let w = NdArrayHandle::as_wrapper(*handles_slice.get(i).unwrap() as *mut _);
                    let v = match extract_view_i8(w, offsets_slice[i], shp, strd) {
                        Some(v) => v,
                        None => {
                            set_last_error("Failed to extract i8 view".to_string());
                            return ERR_GENERIC;
                        }
                    };
                    views.push(v);
                }
                let arr = match stack(Axis(axis_usize), &views) {
                    Ok(a) => a.into_dyn(),
                    Err(e) => {
                        set_last_error(e.to_string());
                        return ERR_SHAPE;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Int8(Arc::new(RwLock::new(arr))),
                    dtype: DType::Int8,
                }
            }
            DType::Uint64 => {
                let mut views = Vec::with_capacity(num_arrays);
                for i in 0..num_arrays {
                    let shp = slice::from_raw_parts(shapes.add(i * ndim), ndim);
                    let strd = slice::from_raw_parts(strides.add(i * ndim), ndim);
                    let w = NdArrayHandle::as_wrapper(*handles_slice.get(i).unwrap() as *mut _);
                    let v = match extract_view_u64(w, offsets_slice[i], shp, strd) {
                        Some(v) => v,
                        None => {
                            set_last_error("Failed to extract u64 view".to_string());
                            return ERR_GENERIC;
                        }
                    };
                    views.push(v);
                }
                let arr = match stack(Axis(axis_usize), &views) {
                    Ok(a) => a.into_dyn(),
                    Err(e) => {
                        set_last_error(e.to_string());
                        return ERR_SHAPE;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Uint64(Arc::new(RwLock::new(arr))),
                    dtype: DType::Uint64,
                }
            }
            DType::Uint32 => {
                let mut views = Vec::with_capacity(num_arrays);
                for i in 0..num_arrays {
                    let shp = slice::from_raw_parts(shapes.add(i * ndim), ndim);
                    let strd = slice::from_raw_parts(strides.add(i * ndim), ndim);
                    let w = NdArrayHandle::as_wrapper(*handles_slice.get(i).unwrap() as *mut _);
                    let v = match extract_view_u32(w, offsets_slice[i], shp, strd) {
                        Some(v) => v,
                        None => {
                            set_last_error("Failed to extract u32 view".to_string());
                            return ERR_GENERIC;
                        }
                    };
                    views.push(v);
                }
                let arr = match stack(Axis(axis_usize), &views) {
                    Ok(a) => a.into_dyn(),
                    Err(e) => {
                        set_last_error(e.to_string());
                        return ERR_SHAPE;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Uint32(Arc::new(RwLock::new(arr))),
                    dtype: DType::Uint32,
                }
            }
            DType::Uint16 => {
                let mut views = Vec::with_capacity(num_arrays);
                for i in 0..num_arrays {
                    let shp = slice::from_raw_parts(shapes.add(i * ndim), ndim);
                    let strd = slice::from_raw_parts(strides.add(i * ndim), ndim);
                    let w = NdArrayHandle::as_wrapper(*handles_slice.get(i).unwrap() as *mut _);
                    let v = match extract_view_u16(w, offsets_slice[i], shp, strd) {
                        Some(v) => v,
                        None => {
                            set_last_error("Failed to extract u16 view".to_string());
                            return ERR_GENERIC;
                        }
                    };
                    views.push(v);
                }
                let arr = match stack(Axis(axis_usize), &views) {
                    Ok(a) => a.into_dyn(),
                    Err(e) => {
                        set_last_error(e.to_string());
                        return ERR_SHAPE;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Uint16(Arc::new(RwLock::new(arr))),
                    dtype: DType::Uint16,
                }
            }
            DType::Uint8 => {
                let mut views = Vec::with_capacity(num_arrays);
                for i in 0..num_arrays {
                    let shp = slice::from_raw_parts(shapes.add(i * ndim), ndim);
                    let strd = slice::from_raw_parts(strides.add(i * ndim), ndim);
                    let w = NdArrayHandle::as_wrapper(*handles_slice.get(i).unwrap() as *mut _);
                    let v = match extract_view_u8(w, offsets_slice[i], shp, strd) {
                        Some(v) => v,
                        None => {
                            set_last_error("Failed to extract u8 view".to_string());
                            return ERR_GENERIC;
                        }
                    };
                    views.push(v);
                }
                let arr = match stack(Axis(axis_usize), &views) {
                    Ok(a) => a.into_dyn(),
                    Err(e) => {
                        set_last_error(e.to_string());
                        return ERR_SHAPE;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Uint8(Arc::new(RwLock::new(arr))),
                    dtype: DType::Uint8,
                }
            }
            DType::Bool => {
                let mut views = Vec::with_capacity(num_arrays);
                for i in 0..num_arrays {
                    let shp = slice::from_raw_parts(shapes.add(i * ndim), ndim);
                    let strd = slice::from_raw_parts(strides.add(i * ndim), ndim);
                    let w = NdArrayHandle::as_wrapper(*handles_slice.get(i).unwrap() as *mut _);
                    let v = match extract_view_bool(w, offsets_slice[i], shp, strd) {
                        Some(v) => v,
                        None => {
                            set_last_error("Failed to extract bool view".to_string());
                            return ERR_GENERIC;
                        }
                    };
                    views.push(v);
                }
                let arr = match stack(Axis(axis_usize), &views) {
                    Ok(a) => a.into_dyn(),
                    Err(e) => {
                        set_last_error(e.to_string());
                        return ERR_SHAPE;
                    }
                };
                NDArrayWrapper {
                    data: ArrayData::Bool(Arc::new(RwLock::new(arr))),
                    dtype: DType::Bool,
                }
            }
        };

        if let Err(e) = write_output_metadata(
            &result_wrapper,
            std::ptr::null_mut(),
            out_ndim,
            out_shape,
            max_ndim,
        ) {
            set_last_error(e);
            return ERR_GENERIC;
        }

        *out_handle = NdArrayHandle::from_wrapper(Box::new(result_wrapper));
        SUCCESS
    })
}
