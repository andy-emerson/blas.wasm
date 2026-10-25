# The call graph

Which operations call which. One node is one mathematically distinct
operation: the 23 netlib-named routines of a real type with the flag
variants split out, because a transposed product or a side swap is a
different result (gemv_t; symm/trmm/trsm left and right). gemm's
internal loop shapes are not extra nodes: they produce bit-identical
results, so the dispatch between them is invisible.

The diagram below is included from `src/README.md`, the routine-naming
and call-graph reference in the source tree, so the two cannot drift
apart. It covers the f64 and f32 layers, which have identical
structure; the complex layers are the same picture with renamed nodes
(`dot` splits into `dotu` and `dotc`, `ger` into `geru` and `gerc`,
`gemv` gains the conjugate-transpose form, and the symmetric family
becomes Hermitian), as that file describes.

{{#include ../../src/README.md:graph}}

Below the operations sits shared plumbing, deliberately not in the
graph: the private SIMD kernels (`src/kernels.rs`, blocked hot loops
several operations share), the lane types (`src/lanes.rs`), gemm's
internal shapes and dispatcher, and the small helpers. `symv` has no
outgoing arrows because its fused kernel does the work the
axpy-plus-dot composition would otherwise do. `rotg` and copy/swap/rot/nrm2/asum/iamax
are leaves: nothing in the crate calls them; consumers do.
