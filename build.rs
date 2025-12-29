#![allow(unused_imports)]

use std::env;
use std::path::{Path, PathBuf};

fn assembly(file_vec: &mut Vec<PathBuf>, base_dir: &Path, _arch: &str, _is_msvc: bool) {
    #[cfg(target_env = "msvc")]
    if _is_msvc {
        let sfx = match _arch {
            "x86_64" => "x86_64",
            "aarch64" => "armv8",
            _ => "unknown",
        };
        let files = glob::glob(&format!("{}/win64/*-{}.asm", base_dir.display(), sfx))
            .expect("unable to collect assembly files");
        for file in files {
            file_vec.push(file.unwrap());
        }
        return;
    }

    file_vec.push(base_dir.join("assembly.S"));
}

fn main() {
    // account for cross-compilation [by examining environment variables]
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    let blst_base_dir = manifest_dir.join("blst");
    println!("Using blst source directory {}", blst_base_dir.display());

    let mut cc = cc::Build::new();
    let c_src_dir = blst_base_dir.join("src");
    println!("cargo:rerun-if-changed={}", c_src_dir.display());
    let mut file_vec = vec![c_src_dir.join("server.c")];

    if target_arch.eq("x86_64") || target_arch.eq("aarch64") {
        let asm_dir = blst_base_dir.join("build");
        println!("cargo:rerun-if-changed={}", asm_dir.display());
        assembly(
            &mut file_vec,
            &asm_dir,
            &target_arch,
            cc.get_compiler().is_like_msvc(),
        );
    } else {
        cc.define("__BLST_NO_ASM__", None);
    }

    // Enable ADX instructions if available
    if target_arch.eq("x86_64") {
        if target_env.eq("sgx") {
            println!("Enabling ADX for Intel SGX target");
            cc.define("__ADX__", None);
        } else if env::var("CARGO_ENCODED_RUSTFLAGS")
            .unwrap_or_default()
            .contains("target-cpu=")
        {
            // If target-cpu is specified on the rustc command line,
            // then obey the resulting target-features.
            let feat_list = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
            let features: Vec<_> = feat_list.split(',').collect();
            if !features.contains(&"ssse3") {
                println!("Compiling in portable mode without ISA extensions");
                cc.define("__BLST_PORTABLE__", None);
            } else if features.contains(&"adx") {
                println!("Enabling ADX because it was set as target-feature");
                cc.define("__ADX__", None);
            }
        }
    }

    cc.flag_if_supported("-mno-avx") // avoid costly transitions
        .flag_if_supported("-fno-builtin")
        .flag_if_supported("-Wno-unused-function")
        .flag_if_supported("-Wno-unused-command-line-argument");
    if target_arch.eq("wasm32") || target_os.eq("none") || target_os.eq("unknown") {
        cc.flag_if_supported("-ffreestanding");
    }
    if target_env.eq("sgx") {
        cc.flag_if_supported("-mlvi-hardening");
        cc.define("__SGX_LVI_HARDENING__", None);
        cc.define("__BLST_NO_CPUID__", None);
        cc.define("__ELF__", None);
    }
    if !cfg!(debug_assertions) {
        cc.opt_level(2);
    }
    cc.files(&file_vec).compile("blst");

    // Generate bindings.
    let bindings = bindgen::Builder::default()
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .use_core()
        .opaque_type("blst_pairing")
        .opaque_type("blst_uniq")
        .derive_default(true)
        .derive_eq(true)
        .rustified_enum("BLST_ERROR")
        .clang_arg("-D__BLST_RUST_BINDGEN__")
        .header("blst/blst.h")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
