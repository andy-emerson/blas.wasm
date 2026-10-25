//! # The call graph
//!
//! Which operations call which. One node is one mathematically
//! distinct operation: the 23 netlib-named routines of a real type with
//! the flag variants split out, because a transposed product or a side
//! swap is a different result (`gemv_t`; `symm`/`trmm`/`trsm` left and
//! right). gemm's internal loop shapes are not extra nodes: they produce
//! bit-identical results, so the dispatch between them is invisible.
//!
//! The diagrams cover the f64 and f32 layers, which have identical
//! structure. The complex layers are the same picture with renamed
//! nodes: `dot` splits into `dotu` and `dotc`, `ger` into `geru` and
//! `gerc`, `gemv` gains the conjugate-transpose form `gemv_c`, and the
//! symmetric family becomes Hermitian (`hemv`, `her`, `her2`, `hemm`,
//! `herk`, `her2k`). Six complex Level 1 routines are one-line
//! delegations onto the same-precision real routine over the
//! interleaved view (copy, swap, the real rotation, the real-α scale,
//! nrm2, asum) and are not nodes of their own.
//!
//! ## What Level 3 calls
//!
//! ```mermaid
//! graph LR
//!   subgraph L3["Level 3"]
//!     gemm
//!     symm_left
//!     symm_right
//!     syrk
//!     syr2k
//!     trmm_left
//!     trmm_right
//!     trsm_left
//!     trsm_right
//!   end
//!   subgraph L21["Operations called"]
//!     gemv
//!     symv
//!     trmv
//!     trsv
//!     axpy
//!     scal
//!   end
//!   gemm --> gemv
//!   symm_left --> symv
//!   symm_right --> axpy
//!   syrk --> axpy
//!   syr2k --> axpy
//!   trmm_left --> trmv
//!   trmm_left --> scal
//!   trmm_right --> axpy
//!   trmm_right --> scal
//!   trsm_left --> trsv
//!   trsm_left --> scal
//!   trsm_right --> axpy
//!   trsm_right --> scal
//! ```
//!
//! ## What Level 2 calls
//!
//! ```mermaid
//! graph LR
//!   subgraph L2["Level 2"]
//!     gemv
//!     gemv_t
//!     ger
//!     symv
//!     trmv
//!     trsv
//!     syr
//!     syr2
//!   end
//!   subgraph L1["Operations called"]
//!     axpy
//!     dot
//!   end
//!   gemv --> axpy
//!   gemv_t --> dot
//!   ger --> axpy
//!   trmv --> axpy
//!   trsv --> axpy
//!   syr --> axpy
//!   syr2 --> axpy
//! ```
//!
//! `symv` calls nothing: its fused kernel does the work the
//! axpy-plus-dot composition would otherwise do. The Level 1 operations
//! are leaves, and `rotg`, `copy`, `swap`, `rot`, `nrm2`, `asum` and
//! `iamax` are called by consumers only.
//!
//! Below the operations sits shared plumbing, deliberately not in the
//! diagrams: the private SIMD kernels (`kernels.rs`, blocked hot loops
//! several operations share), the lane types (`lanes.rs`), gemm's
//! internal shapes and dispatcher, and the small helpers.
