use std::env;
use std::path::{Path, PathBuf};

fn main() {
    generate_c_header();
    link_fortran_static();

    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}

fn generate_c_header() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let config_file = PathBuf::from(&crate_dir).join("cbindgen.toml");
    let output_file = PathBuf::from(&crate_dir)
        .join("..")
        .join("include")
        .join("ndarray_php.h");

    if let Some(parent) = output_file.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(cbindgen::Config::from_file(&config_file).unwrap())
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(&output_file);
}

fn link_fortran_static() {
    if env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() == "macos" {
        link_macos_fortran_static();
    }
}

// ------------------------------------------------------------------
// macOS
// ------------------------------------------------------------------

fn link_macos_fortran_static() {
    let Some(lib_dir) = homebrew_gcc_lib_dir() else {
        return;
    };

    println!("cargo:rustc-link-arg=-Wl,-dead_strip_dylibs");
    macos_force_load(lib_dir.join("libgfortran.a"));
    macos_force_load(lib_dir.join("libquadmath.a"));

    if let Some((gcc, gcc_eh)) = find_libgcc(&lib_dir) {
        macos_force_load(gcc);
        macos_force_load(gcc_eh);
    }
}

fn homebrew_gcc_lib_dir() -> Option<PathBuf> {
    let gcc_base = PathBuf::from("/opt/homebrew/Cellar/gcc");
    if !gcc_base.exists() {
        return None;
    }

    let version = std::fs::read_dir(&gcc_base)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .max_by(|a, b| parse_version(a).cmp(&parse_version(b)))?;

    Some(gcc_base.join(version).join("lib/gcc/current"))
}

fn find_libgcc(dir: &Path) -> Option<(PathBuf, PathBuf)> {
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let gcc = path.join("libgcc.a");
            let eh = path.join("libgcc_eh.a");
            if gcc.exists() && eh.exists() {
                return Some((gcc, eh));
            }
            if let Some(found) = find_libgcc(&path) {
                return Some(found);
            }
        }
    }
    None
}

fn macos_force_load(path: PathBuf) {
    if path.exists() {
        println!("cargo:rustc-link-arg=-Wl,-force_load,{}", path.display());
    }
}

fn parse_version(s: &str) -> Vec<u32> {
    s.split('.').filter_map(|p| p.parse().ok()).collect()
}
