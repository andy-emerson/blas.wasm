# blas.wasm — the BLAS for the browser

blas.wasm is a BLAS (the Basic Linear Algebra Subprograms: vector,
matrix–vector and matrix–matrix routines) written in Rust for
WebAssembly: hand-written SIMD on every path, zero dependencies, zero
imports, and results that are bit-identical across native and every
wasm engine. Today the tree ships the Rust crate `blas-wasm` and a standalone
`blas.wasm` module attached to each tagged GitHub release; the npm
package that will wrap the module is open work, and nothing is
published to a registry (`STATUS.md`). It is a BLAS and nothing more: the
decompositions (solve, inverse, eigen, SVD) are a separate project
that depends on this crate. Why it is built this way is `DESIGN.md`;
what is open is `STATUS.md`.

Layout: one file per BLAS routine per number type in netlib naming
(`daxpy.rs`, `saxpy.rs`, … — convention: `src/README.md`), one folder
per level. Counting convention, used everywhere: **23 routines** per
real type (netlib names = files; flag variants folded, so gemv covers
N/T and symm/trmm/trsm cover both sides) which split into **27
distinct operations** (the call-graph nodes). The complex layers
count differently because reference BLAS itself splits there: **26
routines / 31 operations** each for c64 and c32 (dot → dotu/dotc,
ger → geru/gerc, scal gains a real-α form, gemv adds the
conjugate-transpose form, and the symmetric family becomes
Hermitian). Companion maps: `src/README.md` (who calls whom),
`bench/README.md` (the measured benchmarks), `tests/README.md` (the
bit-identical test results).

## Documents and their jobs

Each document has one job and one primary reader; the documentation
review (`AGENTS.md`) judges it against that reader.

| document | job | reader |
|---|---|---|
| `README.md` (this file) | what is built and how to call it | the person using the library |
| `DESIGN.md` | why it is built this way: product, constraints, claims, design by part, closed decisions with rejected alternatives and reopen triggers | the person building and maintaining it |
| `STATUS.md` | open work (todos, bugs, open decisions) and the latest check results | whoever picks the work up next |
| `src/README.md` | routine naming and the call graph | maintainer |
| `tests/README.md` | what each routine is tested to, and the cross-target bit patterns | maintainer |
| `bench/README.md` | the measurement method and the current scoreboard | maintainer |
| `AGENTS.md` | the working agreement between the Human and the Agent (unedited upstream release; `CLAUDE.md` points at it) | both |

## What is built

- **Four type layers**, f64 / f32 / c64 / c32, every routine of the
  counting convention above: 116 routine exports in the module. Unit
  stride: callers pass contiguous column slices with a column stride
  per matrix.
- **SIMD on every path**, tuned: four streaming-loop shapes plus one
  scalar function (`DESIGN.md`, "Kernels"), the column shapes blocked four
  columns at a time by shared kernels, and gemm size-dispatched
  between a register tile and a BLIS/Goto packed-panel shape in every
  type. Every tuned shape is bit-identical to its plain column-axpy
  reference except where an elimination order forbids it; those two
  cases are documented in their files and mirrored by their tests.
- **Zero runtime dependencies.** `no_std`; the four math entry points
  the layer needs are vendored in `src/math.rs` and cross-checked bit
  for bit against libm (a dev-dependency only) on every CI run.
- **Bit-identical native ↔ wasm**, by construction (`src/lanes.rs`
  emulates the SIMD lane structure elementwise off-wasm) and verified
  by 90 determinism probes on every gate run (`tests/README.md`).
- **The module.** `abi/` wraps every routine in a flat C ABI and
  compiles to the standalone `blas.wasm` (~220 KB, 116 routines plus
  `wasm_alloc`/`wasm_free`, raw wasm, zero imports); conventions in
  `abi/src/lib.rs`, smoke test `abi/check.mjs`, attached to the
  GitHub release for each tag by the release workflow. Rust consumers
  depend on the crate instead (same code, cross-crate inlining, no
  ABI boundary).
- **Measured against the machine's ceilings**, two runner draws per
  verdict; the scoreboard and the method are in `bench/README.md`.
  Verdicts that lost are recorded where they lost: the fused
  single-pass iamax (its module docs), the 4-column hemv grouping for
  c32 (`src/L2/chemv.rs`; it won for c64 and ships there).

## What is not built

Relative to a complete BLAS (`DESIGN.md`, constraint 6), all planned
and sequenced in `STATUS.md`: the transposed and conjugate-transposed
forms of gemm, trmv, trsv, trmm and trsm; the transposed forms of
syrk/syr2k/herk/her2k; vector strides (`incx`/`incy`); the complex
symmetric (non-Hermitian) symm/syrk/syr2k; the complex-s rotation
`zrot`/`crot`; argument errors are trapped rather than reported; the
module's exports use an invented argument order rather than CBLAS's;
packed gemm allocates two panel buffers. Fused multiply-add variants
measured faster for some routines and are deliberately not shipped:
relaxed-SIMD rounding is implementation-dependent and would forfeit
cross-target bit-identity (`DESIGN.md`, "Fast mode is a lever").
Banded and packed storage forms are not planned.

## Calling it

**From Rust.** Add the crate as a path or git dependency (it is not
on crates.io: `STATUS.md`, D2). Routines live under `L1`, `L2` and
`L3` by level and carry netlib names; every matrix is a column-major
slice with its own column stride (`cs ≥ rows`; column `j` of an
`rows × cols` matrix is `a[j*cs .. j*cs + rows]`), and every vector is
a contiguous slice. Flag variants are separate functions
(`dgemv`/`dgemv_t`, `dtrsm_left`/`dtrsm_right`); triangle and
unit-diagonal selection are `bool`s. The same example, compiled and
run on every build of the book, is the axpy chapter's.

```rust
use blas_wasm::{L1, L3};

let x = [1.0, 2.0, 3.0];
let mut y = [4.0, 5.0, 6.0];
L1::daxpy(2.0, &x, &mut y);              // y ← 2x + y = [6, 9, 12]

// C ← αAB + βC for 2×2 matrices, column-major, each with column stride 2.
let a = [1.0, 3.0, 2.0, 4.0];            // [[1, 2], [3, 4]]
let b = [5.0, 7.0, 6.0, 8.0];            // [[5, 6], [7, 8]]
let mut c = [0.0; 4];
L3::dgemm(1.0, 2, 2, 2, &a, 2, &b, 2, 0.0, &mut c, 2);
assert_eq!(c, [19.0, 43.0, 22.0, 50.0]); // [[19, 22], [43, 50]]
```

The `dgemm` argument order is `(alpha, m, k, n, a, acs, b, bcs, beta,
c, ccs)` for an `m×k` times `k×n` product; each routine's own
signature and rounding contract are in its file's module docs.

**From JavaScript, with the raw module.** `blas.wasm` has no imports,
so it instantiates with an empty import object. Arguments are flat
pointers into the module's memory, lengths and strides are `usize`,
flags are `u32`, complex scalars are `(re, im)` pairs, and
`wasm_alloc`/`wasm_free` provide 16-byte-aligned buffers; the full
conventions are the module doc of `abi/src/lib.rs`, and
`abi/check.mjs` is a complete worked example.

```js
const { instance } = await WebAssembly.instantiate(bytes, {});
const e = instance.exports;
const px = e.wasm_alloc(3 * 8), py = e.wasm_alloc(3 * 8);
new Float64Array(e.memory.buffer, px, 3).set([1, 2, 3]);
new Float64Array(e.memory.buffer, py, 3).set([4, 5, 6]);
e.daxpy(2, px, py, 3);                   // y ← 2x + y, in place at py
```

This argument order is the interim flat ABI; the module's exports
become the CBLAS names and order when todo C lands (`STATUS.md`), and
the npm package will hide the memory management entirely. Argument
errors trap today (the instance is lost; re-instantiate), and
memory-bounds violations always will.

## Testing contract — two axes, both required to land

**Correctness — `tests/` in this crate** (`cargo test --release`):
every routine is tested to the strongest standard its math allows — bit-for-bit against a same-order scalar replay
wherever the rounding order is deterministic, n-scaled error bounds
against higher-precision references where lane parallelism
legitimately reorders, exact-index/identity checks for iamax/rotg,
residuals for the solves, and native ↔ wasm bit-identity throughout.
The full rationale (what standard applies to which routine class and
why) and the per-routine table live in `tests/README.md`.

**Performance — `bench/` in this crate**: the score is **distance
from the machine's measured ceiling** — streaming ops against the
bandwidth ceiling, multiply-class ops against the arithmetic peak —
per the re-derived success metric, with same-machine interleaved A/B
rows and the verdict-stability rule (two runner draws per claim; the
dev container is iteration only). The harness is self-contained;
method and current scoreboard: `bench/README.md`. The market
comparison against faer was raced in the predecessor repository and
is quoted there, not reproducible from this tree.
