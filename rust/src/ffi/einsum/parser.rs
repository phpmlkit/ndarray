//! Einsum subscript parser.
//!
//! Parses NumPy-style subscript notation into structured label sets.
//! Supports both single-operand (`ii->`, `ij->ji`) and two-operand patterns.

/// A parsed and validated einsum subscript specification.
#[derive(Debug, Clone)]
pub struct EinsumSpec {
    /// Output shape computed from label→dimension mapping.
    pub out_shape: Vec<usize>,
    /// Axes to contract: (lhs_axis, rhs_axis).
    pub contracted_axes: Vec<(usize, usize)>,
    /// Mapping from output position to (source, axis) or None for broadcast.
    pub output_map: Vec<(bool, usize)>,
    /// Whether this is an element-wise operation.
    pub is_elementwise: bool,
    /// Whether this has a second operand (false = single-operand).
    pub has_rhs: bool,
    /// Shapes for operand validation.
    pub lhs_shape: Vec<usize>,
    pub rhs_shape: Vec<usize>,
    /// Whether to reduce: labels appear only in input, not output (single-operand contraction).
    pub reduce_axes: Vec<usize>,
    /// Whether to permute: single-operand with all labels in output but reordered.
    pub permute_order: Option<Vec<usize>>,
}

/// Parse "ij,jk->ik", "ii->", "ij->ji", etc.
/// `rhs_shape` is `None` for single-operand patterns like `ii->`.
pub fn parse(
    subscripts: &str,
    lhs_shape: &[usize],
    rhs_shape: Option<&[usize]>,
) -> Result<EinsumSpec, String> {
    let s = subscripts.trim().to_lowercase();

    let (input_part, output_part, has_explicit_output) = match s.split_once("->") {
        Some((inp, out)) => (inp.to_string(), out.to_string(), true),
        None => (s, String::new(), false),
    };

    let parts: Vec<&str> = input_part.split(',').map(|p| p.trim()).collect();

    if parts.len() < 1 || parts.len() > 2 {
        return Err(format!(
            "einsum requires 1 or 2 operands, got {}",
            parts.len()
        ));
    }

    let is_single = parts.len() == 1;

    // Validate second operand presence
    if !is_single && rhs_shape.is_none() {
        return Err("two operands in subscript but second array is null".to_string());
    }

    let lhs_labels: Vec<char> = parts[0].chars().collect();
    let rhs_labels: Vec<char> = if is_single {
        Vec::new()
    } else {
        parts[1].chars().collect()
    };

    if lhs_labels.len() != lhs_shape.len() {
        return Err(format!(
            "subscript '{}' has {} labels but operand 1 has {} dimensions",
            parts[0],
            lhs_labels.len(),
            lhs_shape.len()
        ));
    }
    if !is_single {
        let rs = rhs_shape.unwrap();
        if rhs_labels.len() != rs.len() {
            return Err(format!(
                "subscript '{}' has {} labels but operand 2 has {} dimensions",
                parts[1],
                rhs_labels.len(),
                rs.len()
            ));
        }
    }

    // Count label occurrences
    let mut label_counts: Vec<(char, usize)> = Vec::new();
    for &l in &lhs_labels {
        inc_label(&mut label_counts, l);
    }
    for &l in &rhs_labels {
        inc_label(&mut label_counts, l);
    }

    for (label, count) in &label_counts {
        if *count > 2 {
            return Err(format!(
                "label '{}' appears {} times in subscripts (max 2)",
                label, count
            ));
        }
    }

    // Determine output labels
    let out_labels: Vec<char> = if has_explicit_output {
        output_part.chars().collect()
    } else {
        label_counts
            .iter()
            .filter(|(_, c)| *c == 1)
            .map(|(l, _)| *l)
            .collect()
    };

    let rhs_shape_slice = rhs_shape.unwrap_or(&[]);
    let lhs_map: Vec<(char, usize)> = lhs_labels
        .iter()
        .enumerate()
        .map(|(i, &l)| (l, i))
        .collect();
    let rhs_map: Vec<(char, usize)> = rhs_labels
        .iter()
        .enumerate()
        .map(|(i, &l)| (l, i))
        .collect();

    // Compute output shape
    let mut out_shape = Vec::with_capacity(out_labels.len());
    let mut output_map = Vec::with_capacity(out_labels.len());

    for &ol in &out_labels {
        let lhs_axis = lhs_map.iter().find(|(l, _)| *l == ol).map(|(_, a)| *a);
        let rhs_axis = rhs_map.iter().find(|(l, _)| *l == ol).map(|(_, a)| *a);

        match (lhs_axis, rhs_axis) {
            (Some(la), Some(ra)) => {
                if lhs_shape[la] != rhs_shape_slice[ra] {
                    return Err(format!(
                        "label '{}' dimension mismatch: {} vs {}",
                        ol, lhs_shape[la], rhs_shape_slice[ra]
                    ));
                }
                out_shape.push(lhs_shape[la]);
                output_map.push((true, la));
            }
            (Some(a), None) => {
                out_shape.push(lhs_shape[a]);
                output_map.push((true, a));
            }
            (None, Some(a)) => {
                out_shape.push(rhs_shape_slice[a]);
                output_map.push((false, a));
            }
            (None, None) => return Err(format!("output label '{}' not found in inputs", ol)),
        }
    }

    // Find contracted axes
    let mut contracted = Vec::new();
    if !is_single {
        for (label, count) in &label_counts {
            if *count == 2 && !out_labels.contains(label) {
                let la = lhs_map.iter().find(|(l, _)| *l == *label).unwrap().1;
                let ra = rhs_map.iter().find(|(l, _)| *l == *label).unwrap().1;
                if lhs_shape[la] != rhs_shape_slice[ra] {
                    return Err(format!(
                        "contracted label '{}' dimension mismatch: {} vs {}",
                        label, lhs_shape[la], rhs_shape_slice[ra]
                    ));
                }
                contracted.push((la, ra));
            }
        }
    }

    // Find reduction axes (single-operand: labels not in output)
    let mut reduce_axes = Vec::new();
    if is_single {
        for (label, count) in &label_counts {
            if !out_labels.contains(label) {
                let la = lhs_map.iter().find(|(l, _)| *l == *label).unwrap().1;
                for _ in 0..*count {
                    reduce_axes.push(la);
                }
            }
        }
    }

    // Detect permute (single-op, all labels preserved in output, same count, reordered)
    let permute_order =
        if is_single && reduce_axes.is_empty() && out_labels.len() == lhs_labels.len() {
            let mut order = Vec::with_capacity(out_labels.len());
            for ol in &out_labels {
                order.push(lhs_map.iter().find(|(l, _)| l == ol).unwrap().1);
            }
            if order != (0..lhs_shape.len()).collect::<Vec<_>>() {
                Some(order)
            } else {
                None
            }
        } else {
            None
        };

    let is_elementwise = !is_single
        && contracted.is_empty()
        && out_labels.len() == lhs_labels.len()
        && out_labels == lhs_labels
        && out_labels == rhs_labels;

    Ok(EinsumSpec {
        out_shape,
        contracted_axes: contracted,
        output_map,
        is_elementwise,
        has_rhs: !is_single,
        lhs_shape: lhs_shape.to_vec(),
        rhs_shape: rhs_shape_slice.to_vec(),
        reduce_axes,
        permute_order,
    })
}

fn inc_label(counts: &mut Vec<(char, usize)>, label: char) {
    for (l, c) in counts.iter_mut() {
        if *l == label {
            *c += 1;
            return;
        }
    }
    counts.push((label, 1));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_mm() {
        let s = parse("ij,jk->ik", &[3, 4], Some(&[4, 5])).unwrap();
        assert_eq!(s.out_shape, vec![3, 5]);
        assert_eq!(s.contracted_axes.len(), 1);
    }

    #[test]
    fn test_parse_dot() {
        let s = parse("i,i->", &[4], Some(&[4])).unwrap();
        assert!(s.out_shape.is_empty());
    }

    #[test]
    fn test_parse_outer() {
        let s = parse("i,j->ij", &[3], Some(&[4])).unwrap();
        assert_eq!(s.out_shape, vec![3, 4]);
    }

    #[test]
    fn test_parse_trace() {
        let s = parse("ii->", &[4, 4], None).unwrap();
        assert!(s.out_shape.is_empty());
        assert_eq!(s.reduce_axes.len(), 2);
    }

    #[test]
    fn test_parse_transpose() {
        let s = parse("ij->ji", &[3, 4], None).unwrap();
        assert_eq!(s.out_shape, vec![4, 3]);
        assert!(s.permute_order.is_some());
    }

    #[test]
    fn test_parse_diagonal() {
        let s = parse("ii->i", &[4, 4], None).unwrap();
        assert_eq!(s.out_shape, vec![4]);
    }

    #[test]
    fn test_parse_sum_axis() {
        let s = parse("ij->i", &[3, 4], None).unwrap();
        assert_eq!(s.out_shape, vec![3]);
    }

    #[test]
    fn test_parse_sum_all() {
        let s = parse("i->", &[5], None).unwrap();
        assert!(s.out_shape.is_empty());
    }
}
