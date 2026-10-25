# Introduction

blas.wasm is a BLAS — the Basic Linear Algebra Subprograms: vector,
matrix–vector and matrix–matrix routines — written in Rust for
WebAssembly, with hand-written SIMD on every path, zero dependencies,
zero imports, and results that are bit-identical across native and
every wasm engine. It ships as a Rust crate and as an npm package
wrapping the pre-built module. It is a BLAS and nothing more: the
decompositions (solve, inverse, eigen, SVD) belong to a separate
project that depends on this crate.

This book is where the mathematics lives. For each routine family it
gives the operation in notation, the rounding-order argument that
makes the bit-identity promise true for that family, the lane
geometry of the SIMD loop, the dispatch thresholds where there are
any, and the measured results. The code shown in a chapter is the
code that ships: it is included from the source tree by anchor at
build time, so a renamed or removed region fails the build rather
than going stale. Standalone examples in Rust are compiled and run on
every build of the book, so they cannot rot either.

The book is young. It has one routine chapter today; chapters are
added as the work in the repository's living status lands, and the
repository's READMEs shrink to pointers as their content moves here.

Three documents in the repository sit beside this book:

- [`README.md`](https://github.com/andy-emerson/blas.wasm/blob/main/README.md)
  says what is built and how to call it, for the person using the
  library.
- [`DESIGN.md`](https://github.com/andy-emerson/blas.wasm/blob/main/DESIGN.md)
  says why it is built this way — the product, the standing
  constraints, the claims and what makes each true, the design behind
  each part, and every decision closed, with the alternative that lost.
- [`STATUS.md`](https://github.com/andy-emerson/blas.wasm/blob/main/STATUS.md)
  holds the open work, the open decisions, and the latest check
  results.

## Conventions used in this book

Matrices are column-major: column \\(j\\) of an \\(m \times n\\) matrix
\\(A\\) stored at column stride \\(\mathrm{cs} \ge m\\) occupies
\\(A[j \cdot \mathrm{cs} \,..\, j \cdot \mathrm{cs} + m)\\). Routine
names follow netlib: a type prefix (`d` for f64, `s` for f32, `z` for
double complex, `c` for single complex) and the reference BLAS name,
so `daxpy` is the f64 axpy. \\(\mathrm{fl}(\cdot)\\) denotes rounding
to the working precision under the round-to-nearest-even rule that
IEEE 754 and WebAssembly both require.
