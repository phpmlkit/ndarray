/// Broadcasts two ArrayBase instances and performs element-wise operation.
#[macro_export]
macro_rules! broadcast_binary {
    ($a:expr, $b:expr, $fn:path) => {{
        use crate::core::view_helpers::broadcast_shape;
        use crate::core::error::{set_last_error, ERR_SHAPE};
        use ndarray::Zip;

        let broadcast_shape = match broadcast_shape($a.shape(), $b.shape()) {
            Some(s) => s,
            None => {
                set_last_error(format!(
                    "Cannot broadcast shapes {:?} and {:?}",
                    $a.shape(),
                    $b.shape()
                ));
                return ERR_SHAPE;
            }
        };

        let a_bc = match $a.broadcast(broadcast_shape.as_slice()) {
            Some(v) => v,
            None => {
                set_last_error(format!("Failed to broadcast first operand"));
                return ERR_SHAPE;
            }
        };
        let b_bc = match $b.broadcast(broadcast_shape.as_slice()) {
            Some(v) => v,
            None => {
                set_last_error(format!("Failed to broadcast second operand"));
                return ERR_SHAPE;
            }
        };

        Zip::from(&a_bc).and(&b_bc).map_collect($fn)
    }};
}
