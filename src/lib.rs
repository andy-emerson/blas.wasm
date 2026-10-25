//! blas-wasm: the wasm-native BLAS layer, one file per BLAS
//! routine per number type in netlib naming (d = f64, s = f32,
//! z = c64, c = c32 — naming convention and call graph in
//! src/README.md). All four type layers are built; unit stride today
//! (vector strides are planned: STATUS.md, todo S3) — callers pass
//! contiguous column slices. The tuned multi-column loop
//! shapes live in `kernels`; the SIMD lane types with their
//! bit-identical native emulation live in `lanes`; the complex
//! scalars (`C64`/`C32`, exported) in `c64`/`c32`. What is built:
//! README.md; why: DESIGN.md; open work: STATUS.md.
#![no_std]

// The four gemm dispatchers allocate two pack buffers above their
// packed threshold (and the `*gemm_packed` shapes always do); nothing
// else allocates. Link a global allocator if you call gemm. Removing
// this is STATUS.md todo S4.
extern crate alloc;

#[doc(hidden)]
pub mod math;
mod c32;
mod c64;
mod kernels;
mod lanes;

pub use c32::C32;
pub use c64::C64;

// Module names match the folder and table spelling (BLAS "Level 1"
// convention) — deliberately not snake_case.
#[allow(non_snake_case)]
pub mod L1;
#[allow(non_snake_case)]
pub mod L2;
#[allow(non_snake_case)]
pub mod L3;
