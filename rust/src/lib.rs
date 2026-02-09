#[no_mangle]
pub extern "C" fn hello_from_rust() -> i32 {
    println!("Hello from Rust!");
    42
}

#[no_mangle]
pub extern "C" fn add_numbers(a: i32, b: i32) -> i32 {
    a + b
}
