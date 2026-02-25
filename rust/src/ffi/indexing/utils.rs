#[inline]
pub fn normalize_index(idx: i64, len: usize) -> Result<usize, String> {
    let mut i = idx;
    if i < 0 {
        i += len as i64;
    }
    if i < 0 || i >= len as i64 {
        return Err(format!(
            "Index {} is out of bounds for axis with size {}",
            idx, len
        ));
    }
    Ok(i as usize)
}
