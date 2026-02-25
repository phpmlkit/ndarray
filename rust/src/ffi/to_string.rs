//! String formatting for NDArray.
//!
//! Provides human-readable string representation of arrays with proper
//! formatting, truncation for large arrays, and support for all dimensions.

use crate::core::view_helpers::{
    extract_view_as_bool, extract_view_as_f32, extract_view_as_f64, extract_view_as_i16,
    extract_view_as_i32, extract_view_as_i64, extract_view_as_i8, extract_view_as_u16,
    extract_view_as_u32, extract_view_as_u64, extract_view_as_u8,
};
use crate::ffi::metadata::ViewMetadata;
use crate::ffi::types::NdArrayHandle;
use std::io::Write;

/// Format an array into a string buffer.
#[no_mangle]
pub unsafe extern "C" fn ndarray_to_string(
    handle: *const NdArrayHandle,
    meta: *const ViewMetadata,
    buffer: *mut std::os::raw::c_char,
    buffer_size: usize,
    threshold: usize,
    edgeitems: usize,
    precision: usize,
) -> usize {
    if handle.is_null() || meta.is_null() || buffer.is_null() {
        return 0;
    }

    let meta = &*meta;
    let wrapper = NdArrayHandle::as_wrapper(handle as *mut _);
    let shape = std::slice::from_raw_parts(meta.shape, meta.ndim);
    let ndim = shape.len();

    let mut write_buf: Vec<u8> = Vec::with_capacity(buffer_size);

    let result = match ndim {
        0 => format_0d(&wrapper, meta, &mut write_buf, precision),
        1 => format_1d(
            &wrapper,
            meta,
            shape,
            &mut write_buf,
            threshold,
            edgeitems,
            precision,
        ),
        2 => format_2d(
            &wrapper,
            meta,
            shape,
            &mut write_buf,
            threshold,
            edgeitems,
            precision,
        ),
        3 => format_3d(
            &wrapper,
            meta,
            shape,
            &mut write_buf,
            threshold,
            edgeitems,
            precision,
        ),
        _ => format_nd(
            &wrapper,
            meta,
            shape,
            &mut write_buf,
            threshold,
            edgeitems,
            precision,
        ),
    };

    if result.is_err() {
        return 0;
    }

    let written = write_buf.len();

    if written >= buffer_size {
        return written + 1;
    }

    let buf_slice = std::slice::from_raw_parts_mut(buffer as *mut u8, buffer_size);
    buf_slice[..written].copy_from_slice(&write_buf);
    buf_slice[written] = 0;

    written
}

fn write_scalar<T: std::fmt::Display>(
    buf: &mut Vec<u8>,
    val: &T,
    _precision: usize,
) -> std::io::Result<()> {
    write!(buf, "{}", val)
}

fn format_0d(
    wrapper: &crate::core::NDArrayWrapper,
    meta: &ViewMetadata,
    buf: &mut Vec<u8>,
    precision: usize,
) -> std::io::Result<()> {
    macro_rules! format_scalar {
        ($extract_fn:ident) => {{
            if let Some(arr) = $extract_fn(wrapper, meta) {
                if let Some(val) = arr.iter().next() {
                    write_scalar(buf, val, precision)?;
                }
            }
        }};
    }

    match wrapper.dtype {
        crate::dtype::DType::Float64 => format_scalar!(extract_view_as_f64),
        crate::dtype::DType::Float32 => format_scalar!(extract_view_as_f32),
        crate::dtype::DType::Int64 => format_scalar!(extract_view_as_i64),
        crate::dtype::DType::Int32 => format_scalar!(extract_view_as_i32),
        crate::dtype::DType::Int16 => format_scalar!(extract_view_as_i16),
        crate::dtype::DType::Int8 => format_scalar!(extract_view_as_i8),
        crate::dtype::DType::Uint64 => format_scalar!(extract_view_as_u64),
        crate::dtype::DType::Uint32 => format_scalar!(extract_view_as_u32),
        crate::dtype::DType::Uint16 => format_scalar!(extract_view_as_u16),
        crate::dtype::DType::Uint8 => format_scalar!(extract_view_as_u8),
        crate::dtype::DType::Bool => {
            if let Some(arr) = extract_view_as_bool(wrapper, meta) {
                if let Some(val) = arr.iter().next() {
                    write!(buf, "{}", if *val != 0 { "true" } else { "false" })?;
                }
            }
        }
    }
    Ok(())
}

fn format_1d(
    wrapper: &crate::core::NDArrayWrapper,
    meta: &ViewMetadata,
    shape: &[usize],
    buf: &mut Vec<u8>,
    threshold: usize,
    edgeitems: usize,
    precision: usize,
) -> std::io::Result<()> {
    let size = shape[0];
    write!(buf, "[")?;

    if size <= threshold || size <= 2 * edgeitems {
        format_1d_elements(wrapper, meta, buf, 0, size, precision)?;
    } else {
        format_1d_elements(wrapper, meta, buf, 0, edgeitems, precision)?;
        write!(buf, " ... ")?;
        format_1d_elements(wrapper, meta, buf, size - edgeitems, size, precision)?;
    }

    write!(buf, "]")?;
    Ok(())
}

fn format_1d_elements(
    wrapper: &crate::core::NDArrayWrapper,
    meta: &ViewMetadata,
    buf: &mut Vec<u8>,
    start: usize,
    end: usize,
    precision: usize,
) -> std::io::Result<()> {
    macro_rules! format_elements {
        ($extract_fn:ident) => {{
            if let Some(arr) = $extract_fn(wrapper, meta) {
                for (i, val) in arr.iter().enumerate().skip(start).take(end - start) {
                    if i > start {
                        write!(buf, " ")?;
                    }
                    write_scalar(buf, val, precision)?;
                }
            }
        }};
    }

    match wrapper.dtype {
        crate::dtype::DType::Float64 => format_elements!(extract_view_as_f64),
        crate::dtype::DType::Float32 => format_elements!(extract_view_as_f32),
        crate::dtype::DType::Int64 => format_elements!(extract_view_as_i64),
        crate::dtype::DType::Int32 => format_elements!(extract_view_as_i32),
        crate::dtype::DType::Int16 => format_elements!(extract_view_as_i16),
        crate::dtype::DType::Int8 => format_elements!(extract_view_as_i8),
        crate::dtype::DType::Uint64 => format_elements!(extract_view_as_u64),
        crate::dtype::DType::Uint32 => format_elements!(extract_view_as_u32),
        crate::dtype::DType::Uint16 => format_elements!(extract_view_as_u16),
        crate::dtype::DType::Uint8 => format_elements!(extract_view_as_u8),
        crate::dtype::DType::Bool => {
            if let Some(arr) = extract_view_as_bool(wrapper, meta) {
                for (i, val) in arr.iter().enumerate().skip(start).take(end - start) {
                    if i > start {
                        write!(buf, " ")?;
                    }
                    write!(buf, "{}", if *val != 0 { "true" } else { "false" })?;
                }
            }
        }
    }
    Ok(())
}

fn format_2d(
    wrapper: &crate::core::NDArrayWrapper,
    meta: &ViewMetadata,
    shape: &[usize],
    buf: &mut Vec<u8>,
    threshold: usize,
    edgeitems: usize,
    precision: usize,
) -> std::io::Result<()> {
    let rows = shape[0];
    let cols = shape[1];
    let size = rows * cols;

    writeln!(buf, "[")?;

    let show_all_rows = rows <= 2 * edgeitems || size <= threshold;

    if show_all_rows {
        for row in 0..rows {
            write!(buf, " [")?;
            format_2d_row(wrapper, meta, buf, row, cols, precision)?;
            writeln!(buf, "]")?;
        }
    } else {
        for row in 0..edgeitems {
            write!(buf, " [")?;
            format_2d_row(wrapper, meta, buf, row, cols, precision)?;
            writeln!(buf, "]")?;
        }
        writeln!(buf, " ...")?;
        for row in (rows - edgeitems)..rows {
            write!(buf, " [")?;
            format_2d_row(wrapper, meta, buf, row, cols, precision)?;
            writeln!(buf, "]")?;
        }
    }

    write!(buf, "]")?;
    Ok(())
}

fn format_2d_row(
    wrapper: &crate::core::NDArrayWrapper,
    meta: &ViewMetadata,
    buf: &mut Vec<u8>,
    row: usize,
    cols: usize,
    precision: usize,
) -> std::io::Result<()> {
    let start_idx = row * cols;

    macro_rules! format_cols {
        ($extract_fn:ident) => {{
            if let Some(arr) = $extract_fn(wrapper, meta) {
                for (i, val) in arr.iter().enumerate().skip(start_idx).take(cols) {
                    if i > start_idx {
                        write!(buf, " ")?;
                    }
                    write_scalar(buf, val, precision)?;
                }
            }
        }};
    }

    match wrapper.dtype {
        crate::dtype::DType::Float64 => format_cols!(extract_view_as_f64),
        crate::dtype::DType::Float32 => format_cols!(extract_view_as_f32),
        crate::dtype::DType::Int64 => format_cols!(extract_view_as_i64),
        crate::dtype::DType::Int32 => format_cols!(extract_view_as_i32),
        crate::dtype::DType::Int16 => format_cols!(extract_view_as_i16),
        crate::dtype::DType::Int8 => format_cols!(extract_view_as_i8),
        crate::dtype::DType::Uint64 => format_cols!(extract_view_as_u64),
        crate::dtype::DType::Uint32 => format_cols!(extract_view_as_u32),
        crate::dtype::DType::Uint16 => format_cols!(extract_view_as_u16),
        crate::dtype::DType::Uint8 => format_cols!(extract_view_as_u8),
        crate::dtype::DType::Bool => {
            if let Some(arr) = extract_view_as_bool(wrapper, meta) {
                for (i, val) in arr.iter().enumerate().skip(start_idx).take(cols) {
                    if i > start_idx {
                        write!(buf, " ")?;
                    }
                    write!(buf, "{}", if *val != 0 { "true" } else { "false" })?;
                }
            }
        }
    }
    Ok(())
}

fn format_3d(
    wrapper: &crate::core::NDArrayWrapper,
    meta: &ViewMetadata,
    shape: &[usize],
    buf: &mut Vec<u8>,
    threshold: usize,
    edgeitems: usize,
    precision: usize,
) -> std::io::Result<()> {
    let depth = shape[0];
    let rows = shape[1];
    let cols = shape[2];
    let size: usize = shape.iter().product();

    writeln!(buf, "[")?;

    let show_all = depth <= 2 * edgeitems || size <= threshold;

    if show_all {
        for d in 0..depth {
            format_3d_slice(wrapper, meta, buf, d, rows, cols, precision)?;
            if d < depth - 1 {
                writeln!(buf)?;
            }
        }
    } else {
        for d in 0..edgeitems {
            format_3d_slice(wrapper, meta, buf, d, rows, cols, precision)?;
            writeln!(buf)?;
        }
        writeln!(buf, "...")?;
        for d in (depth - edgeitems)..depth {
            format_3d_slice(wrapper, meta, buf, d, rows, cols, precision)?;
            if d < depth - 1 {
                writeln!(buf)?;
            }
        }
    }

    write!(buf, "]")?;
    Ok(())
}

fn format_3d_slice(
    wrapper: &crate::core::NDArrayWrapper,
    meta: &ViewMetadata,
    buf: &mut Vec<u8>,
    slice_idx: usize,
    rows: usize,
    cols: usize,
    precision: usize,
) -> std::io::Result<()> {
    let slice_start = slice_idx * rows * cols;

    for row in 0..rows {
        write!(buf, "  [")?;
        let row_start = slice_start + row * cols;
        format_3d_row(wrapper, meta, buf, row_start, cols, precision)?;
        writeln!(buf, "]")?;
    }
    Ok(())
}

fn format_3d_row(
    wrapper: &crate::core::NDArrayWrapper,
    meta: &ViewMetadata,
    buf: &mut Vec<u8>,
    start_idx: usize,
    cols: usize,
    precision: usize,
) -> std::io::Result<()> {
    macro_rules! format_cols {
        ($extract_fn:ident) => {{
            if let Some(arr) = $extract_fn(wrapper, meta) {
                for (i, val) in arr.iter().enumerate().skip(start_idx).take(cols) {
                    if i > start_idx {
                        write!(buf, " ")?;
                    }
                    write_scalar(buf, val, precision)?;
                }
            }
        }};
    }

    match wrapper.dtype {
        crate::dtype::DType::Float64 => format_cols!(extract_view_as_f64),
        crate::dtype::DType::Float32 => format_cols!(extract_view_as_f32),
        crate::dtype::DType::Int64 => format_cols!(extract_view_as_i64),
        crate::dtype::DType::Int32 => format_cols!(extract_view_as_i32),
        crate::dtype::DType::Int16 => format_cols!(extract_view_as_i16),
        crate::dtype::DType::Int8 => format_cols!(extract_view_as_i8),
        crate::dtype::DType::Uint64 => format_cols!(extract_view_as_u64),
        crate::dtype::DType::Uint32 => format_cols!(extract_view_as_u32),
        crate::dtype::DType::Uint16 => format_cols!(extract_view_as_u16),
        crate::dtype::DType::Uint8 => format_cols!(extract_view_as_u8),
        crate::dtype::DType::Bool => {
            if let Some(arr) = extract_view_as_bool(wrapper, meta) {
                for (i, val) in arr.iter().enumerate().skip(start_idx).take(cols) {
                    if i > start_idx {
                        write!(buf, " ")?;
                    }
                    write!(buf, "{}", if *val != 0 { "true" } else { "false" })?;
                }
            }
        }
    }
    Ok(())
}

fn format_nd(
    wrapper: &crate::core::NDArrayWrapper,
    meta: &ViewMetadata,
    shape: &[usize],
    buf: &mut Vec<u8>,
    threshold: usize,
    edgeitems: usize,
    precision: usize,
) -> std::io::Result<()> {
    let ndim = shape.len();
    let size: usize = shape.iter().product();

    let mut strides = vec![0usize; ndim];
    if ndim > 0 {
        strides[ndim - 1] = 1;
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }

    writeln!(buf, "[")?;

    let dim0_size = shape[0];
    let subshape = &shape[1..];
    let sub_strides = &strides[1..];

    let should_truncate = size > threshold && dim0_size > 2 * edgeitems;

    if !should_truncate {
        for i in 0..dim0_size {
            write_indent(buf, 1);
            format_nd_slice(
                wrapper,
                meta,
                subshape,
                sub_strides,
                i * strides[0],
                buf,
                threshold,
                edgeitems,
                precision,
                1,
            )?;
            if i < dim0_size - 1 {
                writeln!(buf)?;
            }
        }
    } else {
        for i in 0..edgeitems {
            write_indent(buf, 1);
            format_nd_slice(
                wrapper,
                meta,
                subshape,
                sub_strides,
                i * strides[0],
                buf,
                threshold,
                edgeitems,
                precision,
                1,
            )?;
            writeln!(buf)?;
        }
        writeln!(buf, "...")?;
        for i in (dim0_size - edgeitems)..dim0_size {
            write_indent(buf, 1);
            format_nd_slice(
                wrapper,
                meta,
                subshape,
                sub_strides,
                i * strides[0],
                buf,
                threshold,
                edgeitems,
                precision,
                1,
            )?;
            if i < dim0_size - 1 {
                writeln!(buf)?;
            }
        }
    }

    write!(buf, "]")?;
    Ok(())
}

fn write_indent(buf: &mut Vec<u8>, depth: usize) {
    for _ in 0..depth {
        let _ = write!(buf, "  ");
    }
}

fn format_nd_slice(
    wrapper: &crate::core::NDArrayWrapper,
    meta: &ViewMetadata,
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    buf: &mut Vec<u8>,
    threshold: usize,
    edgeitems: usize,
    precision: usize,
    depth: usize,
) -> std::io::Result<()> {
    let ndim = shape.len();

    if ndim == 0 {
        return format_element_at_offset(wrapper, meta, offset, buf, precision);
    }

    if ndim == 1 {
        let size = shape[0];
        write!(buf, "[")?;
        if size <= threshold || size <= 2 * edgeitems {
            format_elements_at_offset(wrapper, meta, strides[0], offset, 0, size, buf, precision)?;
        } else {
            format_elements_at_offset(
                wrapper, meta, strides[0], offset, 0, edgeitems, buf, precision,
            )?;
            write!(buf, " ... ")?;
            format_elements_at_offset(
                wrapper,
                meta,
                strides[0],
                offset,
                size - edgeitems,
                size,
                buf,
                precision,
            )?;
        }
        write!(buf, "]")?;
        return Ok(());
    }

    let dim_size = shape[0];
    let subshape = &shape[1..];
    let sub_strides = &strides[1..];

    writeln!(buf, "[")?;

    let sub_size: usize = subshape.iter().product();
    let should_truncate = sub_size > threshold && dim_size > 2 * edgeitems;

    if !should_truncate {
        for i in 0..dim_size {
            write_indent(buf, depth + 1);
            format_nd_slice(
                wrapper,
                meta,
                subshape,
                sub_strides,
                offset + i * strides[0],
                buf,
                threshold,
                edgeitems,
                precision,
                depth + 1,
            )?;
            if i < dim_size - 1 {
                writeln!(buf)?;
            }
        }
    } else {
        for i in 0..edgeitems {
            write_indent(buf, depth + 1);
            format_nd_slice(
                wrapper,
                meta,
                subshape,
                sub_strides,
                offset + i * strides[0],
                buf,
                threshold,
                edgeitems,
                precision,
                depth + 1,
            )?;
            writeln!(buf)?;
        }
        write_indent(buf, depth + 1);
        writeln!(buf, "...")?;
        for i in (dim_size - edgeitems)..dim_size {
            write_indent(buf, depth + 1);
            format_nd_slice(
                wrapper,
                meta,
                subshape,
                sub_strides,
                offset + i * strides[0],
                buf,
                threshold,
                edgeitems,
                precision,
                depth + 1,
            )?;
            if i < dim_size - 1 {
                writeln!(buf)?;
            }
        }
    }

    writeln!(buf, "]")?;
    Ok(())
}

fn format_element_at_offset(
    wrapper: &crate::core::NDArrayWrapper,
    meta: &ViewMetadata,
    offset: usize,
    buf: &mut Vec<u8>,
    precision: usize,
) -> std::io::Result<()> {
    macro_rules! format_elem {
        ($extract_fn:ident) => {{
            if let Some(arr) = $extract_fn(wrapper, meta) {
                for (i, val) in arr.iter().enumerate() {
                    if i == offset {
                        write_scalar(buf, val, precision)?;
                        break;
                    }
                }
            }
        }};
    }

    match wrapper.dtype {
        crate::dtype::DType::Float64 => format_elem!(extract_view_as_f64),
        crate::dtype::DType::Float32 => format_elem!(extract_view_as_f32),
        crate::dtype::DType::Int64 => format_elem!(extract_view_as_i64),
        crate::dtype::DType::Int32 => format_elem!(extract_view_as_i32),
        crate::dtype::DType::Int16 => format_elem!(extract_view_as_i16),
        crate::dtype::DType::Int8 => format_elem!(extract_view_as_i8),
        crate::dtype::DType::Uint64 => format_elem!(extract_view_as_u64),
        crate::dtype::DType::Uint32 => format_elem!(extract_view_as_u32),
        crate::dtype::DType::Uint16 => format_elem!(extract_view_as_u16),
        crate::dtype::DType::Uint8 => format_elem!(extract_view_as_u8),
        crate::dtype::DType::Bool => {
            if let Some(arr) = extract_view_as_bool(wrapper, meta) {
                for (i, val) in arr.iter().enumerate() {
                    if i == offset {
                        write!(buf, "{}", if *val != 0 { "true" } else { "false" })?;
                        break;
                    }
                }
            }
        }
    }
    Ok(())
}

fn format_elements_at_offset(
    wrapper: &crate::core::NDArrayWrapper,
    meta: &ViewMetadata,
    stride: usize,
    base_offset: usize,
    start: usize,
    end: usize,
    buf: &mut Vec<u8>,
    precision: usize,
) -> std::io::Result<()> {
    macro_rules! format_elems {
        ($extract_fn:ident) => {{
            if let Some(arr) = $extract_fn(wrapper, meta) {
                for (i, val) in arr.iter().enumerate() {
                    if i >= base_offset + start * stride && i < base_offset + end * stride {
                        let idx = (i - base_offset) / stride;
                        if idx as usize >= start && (idx as usize) < end {
                            if idx as usize > start {
                                write!(buf, " ")?;
                            }
                            write_scalar(buf, val, precision)?;
                        }
                    }
                }
            }
        }};
    }

    match wrapper.dtype {
        crate::dtype::DType::Float64 => format_elems!(extract_view_as_f64),
        crate::dtype::DType::Float32 => format_elems!(extract_view_as_f32),
        crate::dtype::DType::Int64 => format_elems!(extract_view_as_i64),
        crate::dtype::DType::Int32 => format_elems!(extract_view_as_i32),
        crate::dtype::DType::Int16 => format_elems!(extract_view_as_i16),
        crate::dtype::DType::Int8 => format_elems!(extract_view_as_i8),
        crate::dtype::DType::Uint64 => format_elems!(extract_view_as_u64),
        crate::dtype::DType::Uint32 => format_elems!(extract_view_as_u32),
        crate::dtype::DType::Uint16 => format_elems!(extract_view_as_u16),
        crate::dtype::DType::Uint8 => format_elems!(extract_view_as_u8),
        crate::dtype::DType::Bool => {
            if let Some(arr) = extract_view_as_bool(wrapper, meta) {
                for (i, val) in arr.iter().enumerate() {
                    if i >= base_offset + start * stride && i < base_offset + end * stride {
                        let idx = (i - base_offset) / stride;
                        if idx as usize >= start && (idx as usize) < end {
                            if idx as usize > start {
                                write!(buf, " ")?;
                            }
                            write!(buf, "{}", if *val != 0 { "true" } else { "false" })?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
