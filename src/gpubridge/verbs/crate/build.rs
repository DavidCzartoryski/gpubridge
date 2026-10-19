//! Compiles the C shim against rdma-core's headers, with the `ibverbs` feature only.
//!
//! No build dependencies (the multitcp crate's rule: std only, builds offline): this runs
//! the C compiler (`CC`, default `cc`) and the archiver (`AR`, default `ar`) itself.
//! `GPUBRIDGE_VERBS_INCLUDE_DIR` adds a directory holding `infiniband/verbs.h`;
//! `GPUBRIDGE_VERBS_LIB_DIR` one holding `libibverbs.so.1`.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=shim/gbverbs.c");
    println!("cargo:rerun-if-changed=shim/gbverbs.h");
    for var in [
        "GPUBRIDGE_VERBS_INCLUDE_DIR",
        "GPUBRIDGE_VERBS_LIB_DIR",
        "CC",
        "AR",
        "CFLAGS",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    if env::var_os("CARGO_FEATURE_IBVERBS").is_none() {
        return;
    }
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        panic!(
            "the ibverbs feature needs Linux (rdma-core); build without it for the mock provider"
        );
    }
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let cc = env::var("CC").unwrap_or_else(|_| "cc".into());
    let ar = env::var("AR").unwrap_or_else(|_| "ar".into());
    let object = out.join("gbverbs.o");
    let mut compile = Command::new(&cc);
    compile.args([
        "-std=gnu11",
        "-O2",
        "-fPIC",
        "-Wall",
        "-Wextra",
        "-Werror=implicit-function-declaration",
        "-Werror=incompatible-pointer-types",
        "-c",
        "shim/gbverbs.c",
        "-o",
    ]);
    compile.arg(&object);
    if let Ok(dir) = env::var("GPUBRIDGE_VERBS_INCLUDE_DIR") {
        compile.arg(format!("-I{dir}"));
    }
    if let Ok(flags) = env::var("CFLAGS") {
        compile.args(flags.split_whitespace());
    }
    let status = compile
        .status()
        .unwrap_or_else(|e| panic!("could not run the C compiler {cc:?}: {e}"));
    if !status.success() {
        panic!(
            "compiling shim/gbverbs.c failed. It needs rdma-core's headers (infiniband/verbs.h): \
             install rdma-core's development package, or set GPUBRIDGE_VERBS_INCLUDE_DIR to a \
             directory holding them (docs/VERBS_TRANSPORT.md)"
        );
    }
    let archive = out.join("libgbverbs_shim.a");
    let _ = std::fs::remove_file(&archive);
    let status = Command::new(&ar)
        .arg("crs")
        .arg(&archive)
        .arg(&object)
        .status()
        .unwrap_or_else(|e| panic!("could not run the archiver {ar:?}: {e}"));
    assert!(status.success(), "{ar} failed to archive the shim");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=gbverbs_shim");
    if let Ok(dir) = env::var("GPUBRIDGE_VERBS_LIB_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
    }
    // By SONAME, so the -devel package's unversioned libibverbs.so isn't needed.
    println!("cargo:rustc-link-lib=dylib:+verbatim=libibverbs.so.1");
}
