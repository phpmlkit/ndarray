use ndarray::ArrayViewD;
use serde::Serialize;
use serde_json::Value;

// Helper to serialize array recursively
pub fn to_json<T, F>(view: ArrayViewD<T>, mapper: F) -> Value
where
    T: Clone,
    F: Fn(&T) -> Value + Copy,
{
    if view.ndim() == 0 {
        if let Some(val) = view.first() {
            mapper(val)
        } else {
            Value::Null
        }
    } else {
        let vec: Vec<Value> = view.outer_iter().map(|sub| to_json(sub, mapper)).collect();
        Value::Array(vec)
    }
}

// Helper to map serializable values to JSON Value
pub fn to_val<T: Serialize>(v: &T) -> Value {
    serde_json::to_value(v).unwrap_or(Value::Null)
}

// Helper to map boolean values (stored as u8) to JSON Value
pub fn bool_mapper(v: &u8) -> Value {
    Value::Bool(*v != 0)
}
