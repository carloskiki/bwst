#![allow(unused_imports)]

use std::env;
use std::path::{Path, PathBuf};

// Functions used by the library
const ALLOWLIST_FUNCTIONS: &[&str] = &[
    // G1 curve functions
    "blst_p1_add_or_double",
    "blst_p1_cneg",
    "blst_p1_mult",
    "blst_p1_is_equal",
    "blst_p1_generator",
    "blst_p1_is_inf",
    "blst_p1_double",
    "blst_p1_in_g1",
    "blst_p1_uncompress",
    "blst_p1_from_affine",
    "blst_p1_compress",
    "blst_p1_to_affine",
    "blst_p1s_to_affine",
    "blst_p1s_mult_pippenger_scratch_sizeof",
    "blst_p1s_mult_pippenger",
    "blst_p1_on_curve",
    // G2 curve functions
    "blst_p2_add_or_double",
    "blst_p2_cneg",
    "blst_p2_mult",
    "blst_p2_is_equal",
    "blst_p2_generator",
    "blst_p2_is_inf",
    "blst_p2_double",
    "blst_p2_in_g2",
    "blst_p2_uncompress",
    "blst_p2_from_affine",
    "blst_p2_compress",
    "blst_p2_to_affine",
    "blst_p2s_to_affine",
    "blst_p2s_mult_pippenger_scratch_sizeof",
    "blst_p2s_mult_pippenger",
    "blst_p2_on_curve",
    // Scalar functions
    "blst_fr_add",
    "blst_fr_cneg",
    "blst_fr_mul",
    "blst_fr_sqr",
    "blst_fr_eucl_inverse",
    // Hash-to-curve functions
    "blst_hash_to_g1",
    "blst_hash_to_g2",
    // Encoding functions
    "blst_encode_to_g1",
    "blst_encode_to_g2",
    // Scalar conversion functions
    "blst_scalar_fr_check",
    "blst_fr_from_scalar",
    "blst_scalar_from_fr",
    // FP12 and pairing functions
    "blst_fp12_mul",
    "blst_fp12_one",
    "blst_miller_loop",
    "blst_fp12_finalverify",
];

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
    let mut bindings = bindgen::Builder::default()
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .use_core()
        .opaque_type("blst_pairing")
        .opaque_type("blst_uniq")
        .derive_default(true)
        .derive_eq(true)
        .rustified_enum("BLST_ERROR")
        .clang_arg("-D__BLST_RUST_BINDGEN__")
        // Remove PartialEq - single regex for all types
        .no_partialeq("blst_(p[12]|fp12|pairing|uniq)")
        // Remove Copy
        .no_copy("blst_(pairing|uniq|scalar)")
        // Remove Default
        .no_default("blst_fp12")
        // Whitelist types used in the library
        .allowlist_type("blst_p1")
        .allowlist_type("blst_p2")
        .allowlist_type("blst_fr")
        .allowlist_type("blst_scalar")
        .allowlist_type("blst_fp")
        .allowlist_type("blst_fp2")
        .allowlist_type("blst_fp12")
        .allowlist_type("blst_p1_affine")
        .allowlist_type("blst_p2_affine")
        .allowlist_type("limb_t")
        .allowlist_type("BLST_ERROR")
        .header("blst/blst.h");

    // Allowlist all functions used in the library
    for func in ALLOWLIST_FUNCTIONS {
        bindings = bindings.allowlist_function(func);
    }

    let bindings = bindings.generate().expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
