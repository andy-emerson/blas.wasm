# The call graph

Which operations call which. One node is one mathematically distinct
operation: the 23 netlib-named routines of a real type with the flag
variants split out, because a transposed product or a side swap is a
different result (gemv_t; symm/trmm/trsm left and right). gemm's
internal loop shapes are not extra nodes: they produce bit-identical
results, so the dispatch between them is invisible.

The two diagrams below are included from `src/README.md`, the
routine-naming and call-graph reference in the source tree, so the
book and the tree cannot drift apart. They cover the f64 and f32
layers, which have identical structure; the complex layers are the
same picture with renamed nodes (`dot` splits into `dotu` and `dotc`,
`ger` into `geru` and `gerc`, `gemv` gains the conjugate-transpose
form, and the symmetric family becomes Hermitian), as that file
describes.

## What Level 3 calls

{{#include ../../src/README.md:graph-l3}}

## What Level 2 calls

{{#include ../../src/README.md:graph-l2}}

`symv` calls nothing: its fused kernel does the work the
axpy-plus-dot composition would otherwise do. The Level 1 operations
are leaves, and `rotg`, `copy`, `swap`, `rot`, `nrm2`, `asum` and
`iamax` are called by consumers only.

Below the operations sits shared plumbing, deliberately not in the
diagrams: the private SIMD kernels (`src/kernels.rs`, blocked hot
loops several operations share), the lane types (`src/lanes.rs`),
gemm's internal shapes and dispatcher, and the small helpers.
