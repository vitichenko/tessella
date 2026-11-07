use std::fs;
use std::path::Path;
use std::process::Command;

fn detect_local_compute_capability() -> Option<usize> {
    let out_dir = std::env::var("OUT_DIR").ok()?;
    let temp_src = Path::new(&out_dir).join("detect_cuda.c");
    let temp_bin = Path::new(&out_dir).join("detect_cuda");

    let src = r#"
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        return 1;
    }
    struct cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        printf("%d%d\n", prop.major, prop.minor);
        return 0;
    }
    return 1;
}
"#;

    if fs::write(&temp_src, src).is_err() {
        return None;
    }

    let mut nvcc_cmd = Command::new("nvcc");
    let compile_res = nvcc_cmd.arg("-o").arg(&temp_bin).arg(&temp_src).status();

    let compile_ok = match compile_res {
        Ok(status) => status.success(),
        Err(_) => {
            // Try fallback path
            let mut nvcc_fallback = Command::new("/usr/local/cuda/bin/nvcc");
            nvcc_fallback
                .arg("-o")
                .arg(&temp_bin)
                .arg(&temp_src)
                .status()
                .map(|status| status.success())
                .unwrap_or(false)
        }
    };

    let mut detected = None;
    if compile_ok {
        if let Ok(output) = Command::new(&temp_bin).output() {
            if output.status.success() {
                let cap_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if let Ok(cap) = cap_str.parse::<usize>() {
                    detected = Some(cap);
                }
            }
        }
    }

    let _ = fs::remove_file(&temp_src);
    let _ = fs::remove_file(&temp_bin);

    detected
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/compatibility.cuh");
    println!("cargo:rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo:rerun-if-changed=src/binary_op_macros.cuh");

    // Multi-arch capability selection:
    // If multiple architectures are requested in TORCH_CUDA_ARCH_LIST or CUDA_COMPUTE_CAP,
    // we select the lowest one (e.g. 8.6) to build a forward-compatible PTX binary.
    let mut selected_cap = None;

    // 1. Check TORCH_CUDA_ARCH_LIST
    if let Ok(arch_list_str) = std::env::var("TORCH_CUDA_ARCH_LIST") {
        let delimiters = [';', ' ', ','];
        let mut current_str = arch_list_str.clone();
        for d in delimiters {
            current_str = current_str.replace(d, ",");
        }
        let mut caps = Vec::new();
        for part in current_str.split(',') {
            let part = part.trim();
            if !part.is_empty() {
                let normalized = part.replace('.', "");
                if let Ok(cap) = normalized.parse::<usize>() {
                    caps.push(cap);
                }
            }
        }
        if !caps.is_empty() {
            caps.sort();
            selected_cap = Some(caps[0]);
        }
    }

    // 2. Check CUDA_COMPUTE_CAP
    if selected_cap.is_none() {
        if let Ok(compute_cap_str) = std::env::var("CUDA_COMPUTE_CAP") {
            let delimiters = [';', ' ', ','];
            let mut current_str = compute_cap_str.clone();
            for d in delimiters {
                current_str = current_str.replace(d, ",");
            }
            let mut caps = Vec::new();
            for part in current_str.split(',') {
                let part = part.trim();
                if !part.is_empty() {
                    let normalized = part.replace('.', "");
                    if let Ok(cap) = normalized.parse::<usize>() {
                        caps.push(cap);
                    }
                }
            }
            if !caps.is_empty() {
                caps.sort();
                selected_cap = Some(caps[0]);
            }
        }
    }

    // 3. Fallback to runtime capability detection
    if selected_cap.is_none() {
        selected_cap = detect_local_compute_capability();
    }

    // 4. Default to fallback capability if everything fails (Pascal GTX 1060 = sm_61)
    let fallback_cap = 61;
    let cap = selected_cap.unwrap_or_else(|| {
        println!("cargo:warning=Compute capability detection failed or in a non-graphics environment. Fallback to compute capability: {}", fallback_cap);
        fallback_cap
    });

    println!("cargo:info=Selecting compute capability: {}", cap);
    std::env::set_var("CUDA_COMPUTE_CAP", cap.to_string());

    let builder = bindgen_cuda::Builder::default();
    println!("cargo:info={builder:?}");
    let _bindings = builder.build_ptx().unwrap();
}
