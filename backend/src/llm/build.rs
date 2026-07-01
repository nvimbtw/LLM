use std::env;
use std::process::Command;

fn main() {
    let manifest = env::var("CARGO_MANIFEST_DIR").unwrap();
    let shader_dir = format!("{}/src/training/shaders", manifest);

    // Resolve the CUDA toolkit root: honor CUDA_HOME/CUDA_PATH if set, else fall
    // back to /opt/cuda (where it lives on this box). nvcc is often not on PATH,
    // so we invoke it by absolute path instead of relying on `nvcc` alone.
    let cuda_home = env::var("CUDA_HOME")
        .or_else(|_| env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/opt/cuda".to_string());
    let nvcc = format!("{}/bin/nvcc", cuda_home);

    Command::new(&nvcc)
        .args([
            "-c",
            &format!("{}/shaders.cu", shader_dir),
            "-o",
            &format!("{}/shaders.o", shader_dir),
            "-O3",
            "--use_fast_math",
            "-arch=sm_86",
            "--compiler-options",
            "-fPIC",
        ])
        .status()
        .unwrap_or_else(|e| panic!("failed to run nvcc at {}: {}", nvcc, e));

    Command::new("ar")
        .args([
            "rcs",
            &format!("{}/libshaders.a", shader_dir),
            &format!("{}/shaders.o", shader_dir),
        ])
        .status()
        .expect("ar failed");

    println!("cargo:rustc-link-search=native={}", shader_dir);
    println!("cargo:rustc-link-search=native={}/lib64", cuda_home);
    println!("cargo:rustc-link-lib=static=shaders");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cublasLt");
    println!("cargo:rerun-if-changed=src/training/shaders/shaders.cu");
    // Track the build script itself: without this, editing the link directives
    // below (e.g. adding cublas) does NOT trigger a rerun, so Cargo keeps
    // replaying a stale cached output and the new libraries never get linked.
    println!("cargo:rerun-if-changed=build.rs");
}
