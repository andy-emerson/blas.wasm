# blas.wasm — the BLAS for the browser

blas.wasm is a BLAS (the Basic Linear Algebra Subprograms: vector,
matrix–vector and matrix–matrix routines) written in Rust for
WebAssembly: hand-written SIMD on every path, zero dependencies, zero
imports, and results that are bit-identical across native and every
wasm engine. It ships as two products, the Rust crate `blas-wasm` and
an npm package wrapping the pre-built module; neither is published to
a registry yet (`STATUS.md`). It is a BLAS and nothing more: the
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
  scalar function (the taxonomy below), the column shapes blocked four
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
  compiles to the standalone `blas.wasm` (~217 KB, 116 routines plus
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

Hard-won build rule: simd128 is NOT in rustc's default wasm32 feature
set — every SIMD path must sit under `#[target_feature(enable =
"simd128")]` on the whole call chain (see `src/lanes.rs`), or the
intrinsics compile as out-of-line calls (measured 6.4× slowdown).

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

## Implementation taxonomy

The whole layer reduces to **four SIMD streaming-loop shapes plus one
scalar function** — the bold codes are the shorthand the tables below
use:

- **ES** — *elementwise stream*: one pass over the vector(s); lanes
  are transformed and written back (includes the fused y ← αx + y
  form).
- **RS** — *reduction stream*: one pass; parallel accumulator lanes,
  folded to a single number at the end.
- **CA** — *column-axpy*: the matrix operation runs as one
  elementwise/axpy stream per column.
- **DCA** — *divide-then-column-axpy*: triangular solves — divide by
  the diagonal entry, then stream the elimination update through the
  remaining columns.

**G** (`rotg`) is the sole exception: no arrays = no SIMD. Guarded
scalar arithmetic, inlined branch-free into the sweep loops that call
it (LAPACK's overflow guards kept — proven numerics).

The tuning campaign (2026-07-18/19) blocks the column shapes four
columns at a time without changing any of them — the shared kernels in
`src/kernels.rs` do the work — the remaining table codes:

- **+FO** — *4-column fan-out* (`axpy4`): one source column streamed
  once into four destination columns — source traffic 4× down.
- **+FI** — *4-column fan-in* (`axpy4in`): four source columns
  accumulated in one pass over the destination — destination
  read-modify-write traffic 4× down.
- **FS** — *fused symv pass* (`axpy_dot`/`axpy_dot4`): one column
  load serves both triangles of the symmetric update, four columns
  grouped per pass over x and y.
- **RT** — *register tile*: gemm's small-matrix micro-kernel (4×4 in
  f64, 8×4 in f32 — same register count at each lane width);
  `CA+RT/FO` means size-dispatched between the tile and fan-out at
  the measured crossover.
- **→d** / **→s** — *delegation*: the complex routine is a one-line
  call onto the tuned same-precision real routine over the
  interleaved 2n-real view (`src/c64.rs` / `src/c32.rs`) — a real
  scale/copy/swap/rotation/reduction of complex data IS the real
  operation on twice the elements, so speed, guards, and determinism
  are inherited rather than duplicated.

The fan-out/fan-in kernels preserve the per-element rounding
sequence, so tuned ops stay bit-for-bit against their plain
column-axpy replays (the two triangular right-side cases whose
elimination order forbids that document their reorder and the tests
mirror it; symv's fused pass legitimately re-folds its reduction and
is bounds-tested). The tables give each op's shorthand per number
type — matching cells across type columns are the clones doing their
job.

A three-way race measured fused-FMA better for `trmm`/`trsm`/`gemv`
and harmful for `syrk`; those variants are deliberately not shipped
(determinism trade-off; `DESIGN.md`, "Fast mode is a lever").
Banded/packed forms are not planned — no consumer demand. Evidence
per row: the July 2026 campaign record (not in this tree;
`STATUS.md`).

The type columns map to the routine-name prefixes (d/s/z/c — the
full convention with the per-routine deviations from reference BLAS:
`src/README.md`); a row names the routine family, so the f64 cell
of `axpy` describes `daxpy`, the f32 cell `saxpy`. In the c64 column
the symmetric rows describe the **Hermitian** z-routine (symv →
`zhemv`, syr/syr2 → `zher`/`zher2`, symm → `zhemm`, syrk → `zherk`,
syr2k → `zher2k`), per the counting note above.

## Level 1 — `src/L1/`

| BLAS | mathematical name | f64 | f32 | c64 | c32 |
|---|---|---|---|---|---|
| `axpy` | scaled vector addition (y ← αx + y) | ES | ES | ES | ES |
| `scal` | scalar × vector | ES | ES | ES ² | ES ² |
| `copy` | vector copy | ES | ES | →d | →s |
| `swap` | exchange two vectors | ES | ES | →d | →s |
| `rot` | apply a plane rotation | ES | ES | →d ³ | →s ³ |
| `dot` | dot product | RS | RS | RS ⁴ | RS ⁴ |
| `nrm2` | Euclidean length (ℓ² norm) | RS | RS | →d | →s |
| `asum` | sum of absolute values (ℓ¹ norm) | RS | RS | →d ⁵ | →s ⁵ |
| `iamax` | index of the largest element | RS | RS | RS ⁶ | RS ⁶ |
| `rotg` | generate a plane rotation | G | G | G | G |

² `zscal`/`cscal` (complex α); the real-α forms `zdscal`/`csscal`
are delegations.
³ `zdrot`/`csrot` — the real-c,s rotation; a real rotation acts on
re/im independently, so it IS `drot`/`srot` on the 2n-real view. The
complex-s apply (`zrot`) has no consumer yet (gap).
⁴ splits into `dotu` (xᵀy) and `dotc` (xᴴy) — different results,
both 4-accumulator reduction streams.
⁵ reference semantics Σ(|re|+|im|) — component magnitudes ARE
`dasum`/`sasum` of the 2n-real view.
⁶ `izamax`/`icamax` maximize |re|+|im| per element — own lane pass
(abs + swap-add), idamax's two-pass shape.

## Level 2 — `src/L2/`

| BLAS | mathematical name | f64 | f32 | c64 | c32 |
|---|---|---|---|---|---|
| `gemv` | matrix × vector | CA+FI | CA+FI | CA+FI | CA+FI |
| `gemv_t` | transposed-matrix × vector (Aᵀx, transpose never formed) | RS per column | RS per column | RS per column ⁷ | RS per column ⁷ |
| `ger` | outer-product update (rank-1) | CA | CA | CA ⁸ | CA ⁸ |
| `symv` | symmetric matrix × vector | FS | FS | FS ⁹ | FS ⁹ |
| `trmv` | triangular matrix × vector | CA+FI | CA+FI | CA+FI | CA+FI |
| `syr` / `syr2` | symmetric rank-1/2 updates | CA | CA | CA | CA |
| `trsv` | triangular solve, one vector | DCA+FI | DCA+FI | DCA+FI ¹⁰ | DCA+FI ¹⁰ |

⁷ two forms: `_t` (Aᵀx, `dotu` per column) and `_c` (Aᴴx, `dotc`
per column) — the conjugate transpose is the one complex algorithms
actually consume.
⁸ splits into `geru` (αxyᵀ) and `gerc` (αxyᴴ) — the conjugation
lands on the per-column scalar, never on a stream.
⁹ the Hermitian `hemv` fused pass is single-column
(`zaxpy_dotc`/`caxpy_dotc`); the 4-column grouping that pushed dsymv
to 2× is a recorded complex tuning lever.
¹⁰ complex division by Smith's algorithm (`C64`/`C32`'s guarded `/`
— the `dladiv` shape).

## Level 3 — `src/L3/`

| BLAS | mathematical name | f64 | f32 | c64 | c32 |
|---|---|---|---|---|---|
| `gemm` | matrix multiplication | CA+RT/FO ¹¹ | CA+RT/FO ¹¹ | CA+FO/RT ¹¹ | CA+FO/RT ¹¹ |
| `syrk` | Gram-matrix update (αAAᵀ + βC) | CA+FO | CA+FO | CA+FO ¹² | CA+FO ¹² |
| `trmm` | triangular matrix multiplication | CA+FO | CA+FO | CA+FO | CA+FO |
| `symm` / `syr2k` | symmetric multiply / rank-2k update | CA+FO ¹ | CA+FO ¹ | CA+FO ¹ | CA+FO ¹ |
| `trsm` | triangular solve, many right-hand sides | DCA+FO | DCA+FO | DCA+FO | DCA+FO |

¹ left-side symm/hemm is FS per column of B.
¹¹ gemm is size-dispatched in every type since the packed-panel race
(2026-07-20, docs step 14): d/s run the register tile small, the
BLIS-style packed-panel shape (`*gemm_packed`) large; z/c run the
col4 fan-out small and a packed-panel COMPLEX register tile large
(2×4 c64, 4×4 c32 — the geometry recorded as a non-starter unpacked;
packing removed the strided k-walk that made it one). All shapes
bit-identical to the `_colaxpy` reference, which stays bit-checked.
¹² the Hermitian `herk`/`her2k` take real α (herk) / real β per the
reference signatures, and maintain the Hermitian invariant: stored
diagonals end exactly real.
