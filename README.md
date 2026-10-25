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
count differently because reference BLAS itself splits there: **30
routines / 36 operations** each for c64 and c32 (dot → dotu/dotc,
ger → geru/gerc, scal gains a real-α form, rot gains a complex-sine
form, gemv adds the conjugate-transpose form, and the symmetric
family exists twice, Hermitian and complex-symmetric). Companion maps: `src/README.md` (who calls whom),
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
| the crate documentation (`cargo doc --open`; GitHub Pages after merge) | the book: each routine's chapter beside its code, the mathematics and the rounding-order arguments, examples compiled by `cargo test` | the person using or studying the library |
| `AGENTS.md` | the working agreement between the Human and the Agent (unedited upstream release; `CLAUDE.md` points at it) | both |

## What is built

- **Four type layers**, f64 / f32 / c64 / c32, every routine of the
  counting convention above: 126 routine exports in the module. Unit
  stride: callers pass contiguous column slices with a column stride
  per matrix.
- **SIMD on every path**, tuned: four streaming-loop shapes plus one
  scalar function (`DESIGN.md`, "Kernels"), the column shapes blocked four
  columns at a time by shared kernels, and gemm size-dispatched
  between a register tile and a BLIS/Goto packed-panel shape in every
  type. Every tuned shape is bit-identical to its plain column-axpy
  reference except where an elimination order forbids it; those two
  cases are documented in their files and mirrored by their tests.
- **Every flag form.** Level 2: gemv's N/T/C, and the transposed trmv
  and trsv as dot-fold reductions down the stored columns. Level 3:
  `transa`/`transb` on gemm, `trans` on the syrk family, `transa` on
  trmm and trsm, each bit-identical to the
  no-transpose call on an explicit transpose and each the winner of a
  two-draw runner race between candidates (`bench/README.md`): large
  gemm packs the transposed operand from a transposed read, trmm/trsm
  read the stored matrix through transposed kernels on the right and
  in the real types on the left, and the complex left side and the
  rest copy the operand into the caller's workspace first.
- **Zero runtime dependencies, zero allocation.** `no_std` without
  `alloc`: the one routine family that needs scratch (gemm above its
  packed threshold) takes it from the caller, sized by a query. The
  four math entry points the layer needs are vendored in
  `src/math.rs` and cross-checked bit for bit against libm (a
  dev-dependency only) on every CI run.
- **Bit-identical native ↔ wasm** for finite results (NaN sign and
  payload bits are outside the claim), by construction (`src/lanes.rs`
  emulates the SIMD lane structure elementwise off-wasm) and verified
  by 126 determinism probes on every gate run (`tests/README.md`),
  every flag form included.
- **The module.** `abi/` wraps every routine in a flat C ABI (the
  interim argument order, to be replaced by CBLAS names and order in
  todo C) and compiles to the standalone `blas.wasm` (~290 KB, 126 routines plus
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
and sequenced in `STATUS.md`: vector strides (`incx`/`incy`); the
module's exports use an invented argument order rather
than CBLAS's and trap on an argument error instead of reporting it.
Fused multiply-add variants
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
a contiguous slice. Arguments follow the netlib order, with the
character flags as enums (`Trans`, `Side`, `Uplo`, `Diag`) and the
strides `incx`/`incy` not yet present. Every routine that checks a
precondition returns `Result<(), Error>`; an `Error` names the routine
and the position of the offending argument, and nothing is written when
one is returned. Routines that pack (gemm above its packed threshold)
take a caller-provided scratch slice sized by their `*_work` query. The
`daxpy` chapter in the crate documentation carries the same example,
compiled and run by `cargo test`.

```rust
use blas_wasm::{L1, L3, Trans};

let x = [1.0, 2.0, 3.0];
let mut y = [4.0, 5.0, 6.0];
L1::daxpy(2.0, &x, &mut y)?;             // y ← 2x + y = [6, 9, 12]

// C ← αAB + βC for 2×2 matrices, column-major, each with leading dimension 2.
let a = [1.0, 3.0, 2.0, 4.0];            // [[1, 2], [3, 4]]
let b = [5.0, 7.0, 6.0, 8.0];            // [[5, 6], [7, 8]]
let mut c = [0.0; 4];
let mut work = vec![0.0; L3::dgemm_work(Trans::NoTrans, Trans::NoTrans, 2, 2, 2)]; // 0 here
L3::dgemm(Trans::NoTrans, Trans::NoTrans, 2, 2, 2, 1.0, &a, 2, &b, 2, 0.0, &mut c, 2, &mut work)?;
assert_eq!(c, [19.0, 43.0, 22.0, 50.0]); // [[19, 22], [43, 50]]
```

The `dgemm` argument order is netlib's, `(transa, transb, m, n, k,
alpha, a, lda, b, ldb, beta, c, ldc)`, plus the trailing `work`; each
routine's own signature, error positions and rounding contract are in
its module documentation. Every Level 3 flag form is built; a
transposed form is bit-identical to the no-transpose call on an
explicit transpose, and the work query counts whatever scratch the
form routed needs. The transposed trmv and trsv reduce with the dot
kernels' fold, a documented rounding order of their own.

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
memory-bounds violations always will. The crate's checks are netlib's,
so a leading dimension of zero traps even for an empty matrix; pass
at least 1.

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
