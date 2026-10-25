//! # blas.wasm — the BLAS for the browser
//!
//! A BLAS (the Basic Linear Algebra Subprograms: vector, matrix–vector
//! and matrix–matrix routines) written in Rust for WebAssembly, with
//! hand-written SIMD on every path, zero dependencies, zero imports,
//! and results that are bit-identical across native and every wasm
//! engine. It is a BLAS and nothing more: the decompositions (solve,
//! inverse, eigen, SVD) belong to a separate project that depends on
//! this crate.
//!
//! ## This documentation is the book
//!
//! Every routine is a module, and each module's page is its chapter:
//! the operation in notation, the rounding-order argument that makes
//! the bit-identity promise true for it, the lane geometry of its loop,
//! its dispatch thresholds where it has any, and its race record. The
//! source is one click away on every page, and every Rust example in
//! the prose is compiled and run by `cargo test`, so examples cannot
//! rot. Chapters that belong to no single routine are under [`book`].
//! Start with [`daxpy`](mod@L1::daxpy) for a complete chapter; the others are
//! being brought up to that standard family by family (progress is in
//! `STATUS.md` in the repository).
//!
//! Layout: one file per routine per number type in netlib naming,
//! `d` = f64, `s` = f32, `z` = c64, `c` = c32, under [`L1`], [`L2`]
//! and [`L3`]; the complex scalars [`C64`] and [`C32`] are exported.
//!
//! ## Conventions
//!
//! Matrices are column-major: column $j$ of an $m \times n$ matrix
//! $A$ stored at column stride $\mathrm{cs} \ge m$ occupies
//! `a[j*cs .. j*cs + m]`. Vectors are contiguous slices today; strides
//! are planned. $\mathrm{fl}(\cdot)$ denotes rounding to the working
//! precision under the round-to-nearest-even rule that IEEE 754 and
//! WebAssembly both require.
//!
//! Beside this book, three documents in the repository have one job
//! each: `README.md` says what is built and how to call it, for the
//! person using the library; `DESIGN.md` says why it is built this way,
//! for the person maintaining it; `STATUS.md` holds the open work, the
//! open decisions and the latest check results.
#![no_std]

// The four gemm dispatchers allocate two pack buffers above their
// packed threshold (and the `*gemm_packed` shapes always do); nothing
// else allocates. Link a global allocator if you call gemm. Removing
// this is STATUS.md todo S4.
extern crate alloc;

pub mod book;
pub mod error;
pub mod flags;
#[doc(hidden)]
pub mod math;
mod c32;
mod c64;
mod kernels;
mod lanes;

pub use c32::C32;
pub use c64::C64;
pub use error::{Error, ErrorKind};
pub use flags::{Diag, Side, Trans, Uplo};

// Module names match the folder and table spelling (BLAS "Level 1"
// convention) — deliberately not snake_case.
#[allow(non_snake_case)]
pub mod L1;
#[allow(non_snake_case)]
pub mod L2;
#[allow(non_snake_case)]
pub mod L3;
