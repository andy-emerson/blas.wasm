# DESIGN — blas.wasm

This document's job: the durable design record, written for the
person building and maintaining blas.wasm. It says what the product
is, the constraints every change honors, the claims the product makes
and what makes each one true, the design behind each part, and every
decision that has been closed — with the alternative that lost and
the condition that would reopen it. It describes the present. Open
work, open decisions and the latest check results live in
`STATUS.md`; what the code does, for the person using it, in
`README.md`; the per-folder conventions in `src/README.md`,
`tests/README.md` and `bench/README.md`.

## The product

**Audience.** Scientific computing in the browser: Pyodide,
JupyterLite, webR, notebook tools, statistics and optimization
libraries, education, interactive visualization. This audience is
f64-heavy and works at sizes from tiny to around a thousand.
Inference runtimes (ONNX Runtime Web, TensorFlow.js and their
relatives) are not consumers; the reasoning is under "Decisions
closed".

**Competitors.** OpenBLAS and BLIS as deployed in the browser:
compiled to wasm via Emscripten, generic C kernels, little SIMD. The
product claim is: **reproducible by construction, verified by test,
faster than everything else that is reproducible, and within 2× of
the fastest thing that is not.** It is not "the fastest gemm on the
web" (see XNNPACK under "Claims").

**Two products, nothing else.**

- *The Rust crate* (`blas-wasm`). Source, not binary. A Rust project
  adds it as a dependency and its own `cargo build` compiles it for
  whatever target that project uses, native or wasm. No C bindings,
  no wasm inside it, nothing pre-compiled. Idiomatic API: enums for
  transpose/side/triangle/diagonal, slices, explicit strides.
- *The npm package* (planned, todo J1; the name is a placeholder
  until publishing is decided, D2). Ready to run. One package containing the
  JavaScript loader, the TypeScript type definitions, and the two
  pre-built wasm files as assets inside the package. Works in Node
  and in browsers. A JavaScript project installs it as one dependency
  and calls functions; it never sees a wasm file. The wasm bytes are
  never inlined into the JavaScript as text; they are separate files
  inside the package, and the loader fetches the one the page can
  use.

Plus the book: the crate documentation, published to GitHub Pages
by the Pages workflow, and on docs.rs once the crate is published.

**One semantic surface, three projections.** The semantics are
netlib BLAS: names, argument meanings, flag meanings, quick-return
rules. Nothing is invented.

- *Rust projection*: the crate's API. Primary, because the code is
  Rust.
- *CBLAS projection* (ruling; today's module exports the interim
  flat ABI, todo C): the wasm module's export list —
  `cblas_dgemm(order, transA, transB, m, n, k, alpha, A, lda, B, ldb,
  beta, C, ldc)` and the rest of the standard, 32-bit `int`,
  `CblasRowMajor` by the standard operand swap. Produced by the
  `abi/` crate, which is a build step for the npm package and is
  never published on its own. These are export names, chosen because
  the module has to export something and the standard names are
  known to everyone; they are not a channel anyone maintains for C
  users.
- *JavaScript projection*: CBLAS names and argument order on typed
  arrays, memory management hidden. The model is stdlib's
  `@stdlib/blas`, the one existing package serving numerical users in
  JavaScript: its users already know these conventions. Whether
  threaded calls are asynchronous everywhere or under their own
  namespace is open (D6).

**What is not shipped, and why.** No static library, no `cblas.h`, no
Fortran-style `dgemm_` symbols, no Emscripten anything. A C
programmer who wants this code inside their own wasm program depends
on the crate, writes a handful of `extern "C"` wrapper lines (the
`abi/` source is a worked example), and builds as a library; the book
has a page for it. Pyodide-style consumers get the same page plus a
note that reference LAPACK links against any BLAS, so they do not
wait on the LAPACK project; their integration is their work.

**Why two wasm files.** A wasm file declares in its header whether
its memory is ordinary or shareable between workers, and the browser
enforces the declaration. A file declaring shared memory cannot be
loaded on a page without the cross-origin-isolation headers, not even
to run on one thread, because the page is forbidden to create the
shared memory object the file demands. A file declaring ordinary
memory cannot be opened by several workers at once. No single file
does both. So "threads when the headers are present, single-threaded
otherwise" forces two files from the same source, differing in one
linker setting: `blas.wasm` (zero imports) and `blas.threads.wasm`
(one import, the memory). The user never sees this: the npm loader
picks one at startup and the browser downloads only that one. The
cost is a doubled gate (both files built, audited, smoke-tested).

**What this shape is and is not.** It is the minimum for a Rust core
whose primary audience writes JavaScript, with precedent in
tree-sitter and Automerge (Rust or C core, crate, npm with wasm
inside). It is not how numerical libraries usually ship: OpenBLAS and
BLIS ship C, faer ships a crate, stdlib ships npm, and bindings are
left to the community. Inside each channel the calls look exactly
like BLAS, so a numerical analyst is not confused by either; but
they would not expect a BLAS to ship this way, and the README does
not pretend otherwise.

## Constraints (standing)

1. **Our build stays pure.** No Emscripten, no wasm-bindgen, no libc
   layer in the artifacts we build. Nothing is built *for* Emscripten
   either; a consumer who wants to link our code into their own
   program does so from the crate, not from an artifact we ship.
2. **Shared memory is a feature, never a requirement.** The host
   detects `crossOriginIsolated`; with it, the threaded artifact runs
   on one shared memory; without it, the single-threaded artifact
   runs unchanged. The single-threaded path stays first-class:
   measured, tested, and shipped on every release, never a fallback
   that stops being looked at.
3. **Bit-identity is kept, and includes thread count.** Same inputs,
   same bits, on every engine and target, and for any number of
   workers. No relaxed-SIMD instruction (fused multiply-add included)
   ever appears in a shipped artifact; the gate scans the binary for
   relaxed opcodes and fails on any. The promise is a property of the
   artifact, never a runtime flag. The thread-count clause is a
   target until the threaded artifact and its p=1 vs p=N test exist
   (todos T1, T5); today's artifact is single-threaded.
4. **Threads are a host concern; kernels never synchronize.** The
   module contains no atomics, no locks, no thread-local state. The
   host partitions disjoint output regions and joins. Parallelism is
   a property of the JavaScript package, never of the Rust code.
5. **Numbers before routing, measured where they apply.** Kernel
   dispatch thresholds (tile/packed) are properties of the code and
   are raced on the reference runners with two draws. Thread fan-out
   thresholds are properties of the user's machine and are calibrated
   at runtime; CI races verify scaling, they do not set the
   constants.
6. **A complete BLAS, netlib semantics.** Every op/side/uplo/diag
   flag, vector strides, quick-return rules (`alpha == 0`, `beta ==
   0` means C is not read, zero dimensions), and the reference
   argument checks. Where reference BLAS and this crate differ, the
   reference wins unless the deviation is recorded here with a
   reason. Today's surface lacks only vector strides (todo S3). Two
   recorded deviations, both about NaN: `nrm2` of a vector holding
   both an infinity and a NaN returns the infinity where the
   reference returns NaN (the scaled accumulation never compares the
   NaN); and the reference skips a column whose multiplier is exactly
   zero in ger, syr, syr2, trmv, trsv, trmm and trsm (`IF (X(J).NE.
   ZERO)`), so a NaN or infinity in that column never reaches the
   output there, while this crate's streams have no per-column branch
   and propagate it (0·NaN = NaN). The reference's gemv, symv, gemm,
   symm, syrk and syr2k have no such skip and agree with this crate.
   Checked against netlib 3.12 on the dev container; not mirrored,
   because the skip would put a branch inside every SIMD stream for a
   case no consumer has asked for.
7. **The crate never allocates.** Scratch space is the caller's,
   passed in and size-queried, LAPACK style: the crate is `no_std`
   without `alloc`, and the build is the check.

## Claims, and what makes each true

| claim | what makes it true |
|---|---|
| **Reproducible.** Finite results bit-identical across engines, targets, and thread counts (a NaN's sign and payload bits are outside the promise), tested, and auditable from the binary. | per-routine bitwise replays and native ↔ wasm probes on Node (`tests/README.md`), held for other engines by the wasm specification's deterministic float semantics; the p=1 vs p=N bitwise test per routine (todo T5, the thread-count part is a target until then); the relaxed-opcode scan in the gate (`abi/audit.mjs`) |
| **Dependency-free.** One raw wasm file, zero imports, no toolchain glue, no libc. | the import-count assertion in the gate; `blas.threads.wasm` imports exactly one thing (shared memory) and nothing else |
| **Faster than OpenBLAS and BLIS in the browser.** | the Step B table: same engine, same machine, two draws, single- and multi-threaded |

How far each claim currently stands is tracked in `STATUS.md`.

Not claimed: faster than XNNPACK on f32 gemm. XNNPACK uses the
relaxed-SIMD fused multiply-add, which doubles the arithmetic peak on
every modern core; a bit-identical kernel cannot use it. The
expectation is to trail on sgemm while leading on f64, complex, and
every non-gemm Level 3 routine; the row is published anyway. How far
behind is a measurement, not a slogan, and whether a kernel campaign
precedes that measurement is decision D11.

Where the reproducibility comes from, stated honestly: wasm's float
semantics are deterministic by specification, so any BLAS compiled to
wasm without relaxed instructions and run single-threaded is probably
bit-reproducible across machines by accident, OpenBLAS included. What
they do not have is a promise, a test suite that enforces it,
thread-count independence, a binary audit, or an engineering practice
that keeps it true as kernels change. The moat is rigor, not
impossibility. Say it that way, so nobody measures OpenBLAS and calls
the claim false.

The ceiling the promise imposes is narrow: bit-identity forbids
exactly the relaxed instructions and nothing else. Packing, blocking,
dispatch and thread partitioning are all available. Levels 1 and 2
and the bandwidth-bound Level 3 rows are memory-limited and lose
nothing; the tax is confined to compute-bound gemm-like shapes.

## Design by part

### Kernels

Four SIMD streaming-loop shapes plus one scalar function, blocked
four columns at a time by shared kernels, with gemm size-dispatched
between a register tile and a BLIS/Goto packed-panel shape. The
taxonomy and the per-routine cells follow; the call graph and the
naming convention are in `src/README.md`. Every tuned shape is bit-identical to
its plain column-axpy reference except where an elimination order
forbids it; those two cases are documented in their files and
mirrored by their tests. Lane emulation off-wasm (`src/lanes.rs`)
reproduces the SIMD lane structure elementwise, which is what makes
native and wasm agree bit for bit.

The hard-won build rule: simd128 is not in rustc's default wasm32
feature set, so every SIMD path sits under `#[target_feature(enable
= "simd128")]` on the whole call chain, or the intrinsics compile as
out-of-line calls (measured 6.4× slowdown).

#### Implementation taxonomy

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

The tuned kernels block the column shapes four columns at a time
without changing any of them — the shared kernels in
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
Banded/packed forms are not planned — no consumer demand. Evidence per
row: the campaign record (`STATUS.md`, "Not in this tree").

The type columns map to the routine-name prefixes (d/s/z/c — the
full convention with the per-routine deviations from reference BLAS:
`src/README.md`); a row names the routine family, so the f64 cell
of `axpy` describes `daxpy`, the f32 cell `saxpy`. In the c64 column
the symmetric rows describe the **Hermitian** z-routine (symv →
`zhemv`, syr/syr2 → `zher`/`zher2`, symm → `zhemm`, syrk → `zherk`,
syr2k → `zher2k`), per the counting note above. The complex symmetric
`zsymm`/`zsyrk`/`zsyr2k` (and the c twins) exist beside them with the
same codes: `zsymm` is the `zhemm` shape unconjugated, `zsyrk` and
`zsyr2k` the `dsyrk`/`dsyr2k` shape on complex lanes.

#### Level 1 — `src/L1/`

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
re/im independently, so it IS `drot`/`srot` on the 2n-real view.
`zrot`/`crot` (real c, complex s) are routines of their own, the
scalar definition per element with no fold (`src/L1/zrot.rs`).
⁴ splits into `dotu` (xᵀy) and `dotc` (xᴴy) — different results,
both 4-accumulator reduction streams.
⁵ reference semantics Σ(|re|+|im|) — component magnitudes ARE
`dasum`/`sasum` of the 2n-real view.
⁶ `izamax`/`icamax` maximize |re|+|im| per element — own lane pass
(abs + swap-add), idamax's two-pass shape.

#### Level 2 — `src/L2/`

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
⁹ c64 `zhemv` ships the 4-column grouped fused pass
(`zaxpy_dotc4`); c32 `chemv` keeps the single-column pass
(`caxpy_dotc`) because the grouping lost its c32 race
(`src/L2/chemv.rs`).
¹⁰ complex division by Smith's algorithm (`C64`/`C32`'s guarded `/`
— the `dladiv` shape).

#### Level 3 — `src/L3/`

| BLAS | mathematical name | f64 | f32 | c64 | c32 |
|---|---|---|---|---|---|
| `gemm` | matrix multiplication | CA+RT/FO ¹¹ | CA+RT/FO ¹¹ | CA+FO/RT ¹¹ | CA+FO/RT ¹¹ |
| `syrk` | Gram-matrix update (αAAᵀ + βC) | CA+FO | CA+FO | CA+FO ¹² | CA+FO ¹² |
| `trmm` | triangular matrix multiplication | CA+FO | CA+FO | CA+FO | CA+FO |
| `symm` / `syr2k` | symmetric multiply / rank-2k update | CA+FO ¹ | CA+FO ¹ | CA+FO ¹ | CA+FO ¹ |
| `trsm` | triangular solve, many right-hand sides | DCA+FO | DCA+FO | DCA+FO | DCA+FO |

¹ left-side symm/hemm is FS per column of B.
¹¹ gemm is size-dispatched in every type: d/s run the register tile small, the
BLIS-style packed-panel shape (`*gemm_packed`) large; z/c run the
col4 fan-out small and a packed-panel COMPLEX register tile large
(2×4 c64, 4×4 c32 — the geometry recorded as a non-starter unpacked;
packing removed the strided k-walk that made it one). All shapes
bit-identical to the `_colaxpy` reference, which stays bit-checked.
¹² the Hermitian `herk`/`her2k` take real α (herk) / real β per the
reference signatures, and maintain the Hermitian invariant: stored
diagonals end exactly real.

### The surface

- **Level 3 op flags.** The test is the definition: a flagged call
  equals the no-transpose call on an explicitly transposed (and
  conjugated — an exact sign flip) copy, bitwise, for every flag
  combination; every form is bit-identical to it by construction,
  because each one performs the same per-element arithmetic sequence
  as the no-transpose kernel. Two candidates exist for every form and
  the wrappers route the winners of decision D3's race (`DESIGN.md`,
  "Decisions closed"; the numbers in `bench/README.md`); the losers
  stay exported and tested. Above the packed threshold gemm packs the
  transposed operand into the panel format from a transposed read
  (op(A)'s row is a stored column; the three tails that read operands
  directly are gathered into the workspace, and a transposed A's
  column tail rides a zero-padded fourth column group); below it, for
  the syrk family, and for c32 gemm when only B is transposed, the
  operand is transposed into the caller's workspace ahead of the
  existing shape (a cost of order mk against mnk of arithmetic). trmm
  and trsm route per side between the same copy and no-copy kernels:
  a transposed accessor on the right (the right-side kernels only
  ever read A as scalars) in every type, and on the left
  column-reduction kernels that walk op(A)'s rows as stored columns,
  lanes across four right-hand sides, in f64 and f32; the complex
  left side copies.
- **Level 2 op flags.** gemv has N/T/C. The transposed trmv and trsv
  are the `gemv_t`-shaped reduction per stored column: each new
  element is the Level 1 dot kernel's fold of the column's triangle
  segment with `x` (op(A)'s row read contiguously, SIMD in the fold),
  so their contract is that fold's deterministic order — replayed bit
  for bit by the tests — rather than the sweep's sequence, which no
  single-vector SIMD reduction can keep. The alternative, a copy of
  op(A), costs as much as the operation itself and was not raced.
- **Workspace.** Scratch is a caller-provided slice, `work: &mut
  [T]`, on gemm, the syrk family, trmm and trsm, beside a per-routine
  size query that takes the flags (`dgemm_work(transa, transb, m, n,
  k) -> usize`, `dtrmm_work(side, transa, m, n)`), because the routed
  candidate decides the size. The query is zero for every
  no-transpose call below gemm's packed threshold and for the
  transposed forms that need no copy, so those calls pass an empty
  slice; a slice shorter than the query is an argument error, never a
  silent slower path, and the query does not see α, so the slice is
  required even for a call with α = 0. The crate never allocates and has no `alloc`
  (constraint 7, required by threading: no allocator inside a shared
  memory): the packed gemm shape carves its panels from the slice,
  and the tests hand it a NaN-filled slice of exactly the queried
  size so a read of unpacked scratch would show. This shape is
  settled by constraint 7 together with the no-silent-miss rule
  rather than by a separate ruling: an optional workspace would hide
  a performance cliff, an owning value would need to allocate or be
  the same slice underneath, and exported block constants push the
  arithmetic onto every consumer. In the module, whose signatures
  carry no workspace argument, each wrapper today allocates the
  queried scratch per call inside the module (the crate itself never
  allocates); the CBLAS surface (todo C) replaces that with a scratch
  region the host registers once per instance through an exported
  `set_workspace(ptr, bytes)`, a wrapper whose registered region is
  too small writing the error slot and returning. Per-instance
  registration is a mutable global, which wasm already makes per
  instance, so threads need no synchronization.
- **Reference semantics.** `alpha == 0` reads neither operand,
  `beta == 0` never reads the output, `k == 0` scales, a zero
  dimension returns, NaN propagates as in reference BLAS (the one
  recorded deviation is under constraint 6); the complex symmetric
  (non-Hermitian) symm/syrk/syr2k; `zrot`/`crot`; iamax's NaN rule
  (a NaN first element wins, any other NaN is skipped).
- **Argument errors are reported.** The crate keeps trapping on
  memory-bounds violations: a trap is the right answer to an invalid
  pointer. Argument errors (negative n, bad flag, lda < m) are
  reported, not trapped: every netlib-named routine returns
  `Result<(), Error>`, and `Error` names the routine and the position
  of the offending argument, so it maps one to one onto xerbla.
  Every checkable precondition returns `Err`, a slice shorter than
  the declared dimensions included (D4, under "Decisions closed");
  the exported raced kernels assert theirs. The module writes the
  error in a per-instance slot (`error_arg`, `error_kind`,
  `error_routine`, `error_clear`) and returns without computing,
  mirroring reference BLAS's xerbla-then-return; the JavaScript layer
  reads it and throws. *Recorded decision:* a zero-import module
  cannot call a host error handler, so the slot replaces the
  callback.
- **Strides** (todo S3). `incx`/`incy` on every Level 1 and Level 2
  vector argument. Unit stride keeps the SIMD streams. Non-unit
  stride takes a path that reuses the lane emulation so the fold
  order is identical and results stay bit-identical to the
  unit-stride call on a gathered copy. Negative strides per reference
  semantics. How the stride appears in the Rust signatures is
  decision D7.

The testing contract is the same for every form: bitwise replay
against the same-order scalar definition, bounds against
higher-precision references, residuals for solves, native ↔ wasm
probes for every form. The module's expected import and export counts
have one owner, `abi/surface.json` (0 and 133 today), read by the
audit in the gate and the release workflow; the file moves with the
surface, deliberately.

### The module's export surface (todo C)

The `abi/` crate exports the CBLAS names and argument order
(32-bit `int`, the full standard), replacing the invented flat
signatures it has today. `CblasRowMajor` is the standard operand
swap into the column-major kernels; no row-major kernels. Argument
validation writes the error slot and returns. Each wrapper is a few
lines generated mechanically from the Rust signatures; the uniform
shape is kept. Whether `wasm_alloc`/`wasm_free` remain as exports is
decision D5. No header is shipped; the standard `cblas.h` already
describes these exports. Fortran-style symbols are not built.

### Threading by shared memory (todo group T)

- **The threaded artifact.** A second build of the identical kernels
  with the atomics, bulk-memory and mutable-globals target features,
  memory imported and shared, data segments initialized once. The
  kernels use no atomics; the features exist only so the module can
  be instantiated N times over one `WebAssembly.Memory({shared:
  true})`. Each instance gets its own stack region (mechanism:
  decision D10). The build needs a nightly toolchain with
  `-Zbuild-std`, because the linker refuses shared memory against
  objects compiled without atomics and the prebuilt standard library
  is one; the single-threaded artifact stays on stable, and the gate
  pins the nightly.
- **The partitioner** lives in the JavaScript package. A worker pool,
  each worker holding one instance over the shared memory. Fork-join
  on disjoint output regions: gemm, symm/hemm left, trmm/trsm left by
  column slabs of the output (each column is computed by the same
  kernel in the same order as the whole call, so the result is
  bit-identical for any slab width; the size-dispatch invisibility
  proven in `tests/README.md` is the same property); trmm/trsm right
  and symm/hemm right by row slabs (the kernels take a column stride,
  so a row slab is a view); syrk/herk/syr2k/her2k by column blocks of
  the stored triangle with widths chosen for equal triangular work;
  Levels 1 and 2 never (memory-bound; workers share one bus). Without
  a shared packed panel, which constraint 4 forbids, each worker
  packs its own copy of the A panel; the scaling race measures what
  that costs.
- **Call shape and thread count.** From the main thread a threaded
  call is asynchronous (the main thread may not block in
  `Atomics.wait`); from inside a worker the package offers the
  synchronous fork-join as well. Default thread count from
  `navigator.hardwareConcurrency`, capped, overridable by the caller.
  Without `crossOriginIsolated`, every entry point runs
  single-threaded on `blas.wasm` with the same signatures. The
  signature shape is decision D6.
- **Thresholds by runtime calibration.** Fan-out thresholds per
  routine class come from the user's machine, not from CI; CI races
  verify the shape of the scaling curve and catch regressions. Probe
  versus cache versus table is decision D9.
- **The reproducibility test.** For every Level 3 routine, every
  type, odd sizes that exercise tails and block edges: p=1 vs p=2 vs
  p=N, compared bitwise, native and wasm. It lands with the
  partitioner, not after. It is what turns constraint 3's
  thread-count clause from an argument into a fact.

Expectations, to be replaced by measurements: Level 3 above a few
hundred rows, roughly 3–5× on eight cores for gemm, less for the
bandwidth-bound rows. Levels 1–2 unchanged. Users without the headers:
unchanged.

### The book (todo group D)

The crate documentation is the book. Rust's own literate facility,
rustdoc, renders every module's prose as a page with the source one
click away, and compiles and runs every Rust example in the prose
under `cargo test`, so examples cannot rot. Each routine's module
page is its chapter: the operation in notation, the rounding-order
argument that makes the bit-identity claim true for that family, the
lane geometry, the dispatch thresholds and the measured results.
Chapters that belong to no single routine are doc-only modules under
`blas_wasm::book`. Mathematics and diagrams come from one header file
(`rustdoc/header.html`, KaTeX and Mermaid from a CDN, pinned),
injected with `--html-in-header`; the gate builds the documentation
with warnings denied, so a broken link or bad markup fails; the Pages
workflow runs `cargo doc` and publishes `target/doc`. Nothing else is
added to the repository for the book.

- One chapter per routine, written in the routine's file. `daxpy`
  sets the standard; the rest are brought to it family by family as
  work on them lands.
- The content of the four READMEs migrates into the crate docs and
  the READMEs shrink to a paragraph and a link.
- The API reference and the book are the same site, so there is
  nothing to cross-link.

Why not tangling (Markdown as the source of truth, Rust files
generated from its code blocks): for this codebase it means
rewriting all 106 routine files as Markdown, compiler and clippy
diagnostics that point into generated files while kernels are being
tuned, and a tangler in CI; the paradigm is legitimate and the
reasons are about this codebase. Why not mdBook with live includes:
it was built and replaced. It is a standard tool, but here it was a
second toolchain beside the code (a build tool, two preprocessors,
generated assets, an include checker) for prose that rustdoc already
places next to the code.

### Measurement (todo group B)

The score is distance from the machine's measured ceiling: streaming
operations against the bandwidth ceiling, multiply-class operations
against the arithmetic peak, same-run, with two independent runner
draws per verdict; the dev container is for iteration only. Method
and current scoreboard: `bench/README.md`.

The competitor table: same engine, same machine, two draws,
single-threaded and threaded (p = hardware threads), all four gemm
types plus trsm and syrk. Rows: OpenBLAS compiled to wasm via
Emscripten (no official target upstream; the exact configuration is
recorded so the row is reproducible); BLIS the same way; XNNPACK f32
gemm as used by TensorFlow.js and ONNX Runtime Web (the row expected
to lose, published so the "within 2×" claim is a measurement);
stdlib's `@stdlib/blas` (faithful, reference semantics, pure
JavaScript, no SIMD, no threads: it scopes the JavaScript claim to
"no *fast* BLAS exists for JavaScript"); a pure-JS matrix library
such as ml-matrix; faer as the historical row. No transitive
inference from any native comparison is valid.

Why we win where we win: hand-written simd128 kernels on every path,
not the absence of Emscripten. Emscripten does not slow a hot loop,
and the claim is never phrased as if it did.

### The npm package (todo group J)

Contents: the loader, TypeScript type definitions, `package.json`,
and the two wasm files as assets. ESM, works in Node and browsers.
Loads `blas.wasm` or `blas.threads.wasm` by feature detection;
exposes the CBLAS-named functions on typed arrays with explicit dtype,
dimensions, strides and flags, following stdlib's argument
conventions; validates arguments in JavaScript before calling (a trap
kills the instance; after a trap the package re-instantiates); reads
the error slot and throws. Views over `memory.buffer` are never
cached across a call that may grow memory. One opt-in construct from
the product layer: a resident buffer handle for callers who chain
operations (copy in once, compute many, copy out once); the full
residency/handle API belongs to the deferred product layer.

## Verification

Every code pass is verified by an independent reviewer briefed to
find where the work is wrong, not to confirm that it builds: results
are checked against the reference definitions (bitwise where the
rounding order is deterministic, bounded where it is not), and
performance against the machine's measured ceiling, by someone who
did not write the code. A check that only proves a build succeeded is
not evidence for a claim about what the code computes.

## Decisions closed

Each entry: the ruling, the alternative that lost, and the reopen
trigger where one is worth keeping. Open decisions are in
`STATUS.md`.

- **D0 — Attribution.** The Human is the author of record on every
  commit; the Agent is co-author on every commit, recorded as a
  `Co-authored-by` trailer. Holds for every session until the Human
  says otherwise.
- **D1 — Where the records live.** Three files, one job each:
  `README.md` for the user, `DESIGN.md` for the maintainer,
  `STATUS.md` for open work and latest checks. Rejected: keeping a
  roadmap that mixes design, todo and chronology; GitHub Issues as
  living status (allowed by the agreement, but it moves open work off
  the clone).
- **D3 — The transposed triangular forms are chosen by a race, not
  by argument.** All candidate implementations are built and raced on
  the reference runners with two draws: the transposed copy into
  caller workspace feeding the existing kernels, new row-streaming
  kernels, and trmm/trsm rebuilt on the packing layer so the
  transposed forms ride the packer as gemm's do. The winner is
  routed; the losers stay in the record with their numbers, as the
  existing kernels' losers do. Rejected: ruling on a dev-container
  measurement of one candidate against an inference about the
  others. Every routine that needs scratch takes the workspace slice
  described under "The surface". The verdicts and the routing are in
  `bench/README.md`; the third candidate's standing is the closing
  question in `STATUS.md`.
- **D4 — The Rust crate's error convention.** `Result<(), Error>` on
  every routine, the error carrying the routine and the argument
  position, with every checkable precondition reported and no panic
  on input. Rejected: keeping panics in the crate and reporting only
  at the module boundary (the crate alone would not be a complete
  BLAS under constraint 6, and a panic kills a wasm instance);
  validated view types (a product-layer vocabulary, not a BLAS one;
  it can be layered on top later); `debug_assert` only (silent wrong
  answers in release). Reopen trigger for adding `unsafe`
  unchecked twins: a consumer measures the argument checks as a cost
  on small calls.
- **D2 — Publishing to the registries.** Nothing is published to npm
  or crates.io for now; the package names are placeholders;
  `publish = false` stays, and no compatibility layer is kept for
  signature changes, since no one depends on the crate. Milestone M3 produces
  release-ready artifacts attached to the GitHub release only. Reopen
  trigger: the Human says the surface is ready to publish; the
  publish jobs are then a short code pass on the release workflow.
- **Scope is BLAS.** The LAPACK subset is a separate project that
  depends on this crate; the two have different shapes
  (OpenBLAS-style library versus decomposition product) and different
  audiences, and folding them together diluted both. Its scope rule,
  for when it starts: the routines `numpy.linalg` calls, per type
  (gesv/getrf/getrs, potrf/potrs, geqrf/orgqr/ormqr, gels, syev/heev,
  gesvd, geev), sequenced L1 factorizations → symmetric eigen → SVD →
  general eigen; it drives BLAS gaps back here; its own conventions
  must include `info` return.
- **The consumer is scientific computing in the browser; inference
  runtimes are not consumers.** Their hot path is f32, f16 and int8
  gemm where the relaxed dot-product and fused multiply-add
  instructions are the whole game; token-by-token decoding is
  matrix-times-vector, memory-bound, and won by weight quantization
  rather than a better Level 3; a BLAS is a small fraction of a
  runtime that is also a graph compiler with hundreds of operators and
  a memory planner. They already carry hand-tuned relaxed-SIMD kernels
  with shared-memory threads. A reproducible float BLAS has nothing to
  offer them.
- **WebGPU is out.** No consumer identified. The audience is
  f64-heavy and WGSL has no f64; its matrices rarely exceed ~1024 and
  at that size the single-threaded CPU path is already well under a
  second; the one honest GPU argument in numerical work is many small
  matrices, which is a batched API that neither Pyodide nor reference
  LAPACK calls. A GPU path also forfeits bit-identity (vendor-specific
  accumulation and contraction, no WGSL strict mode), so it could
  never be automatic. Reopen trigger: a named consumer with a
  batched-shaped need. (A WGSL-shaped GPU BLAS for inference runtimes
  would be a sister project sharing nothing with this one but the
  measurement discipline.)
- **Fast mode is a lever, not a product.** The relaxed-SIMD fused
  multiply-add is worth up to 2× on compute-bound Level 3 (less where
  memory binds; expect 1.3–1.7× on dgemm) and is the only route to
  parity with XNNPACK on sgemm. Not shipped, because (a) the
  product's promise is reproducibility and a mode that forfeits it
  dilutes the one claim no competitor can match; (b) for the named
  audience, which is f64, reproducible mode is already the fastest
  option in the browser; (c) it is a second kernel set with its own
  tolerance contract. Precedent: Intel MKL offers Conditional
  Numerical Reproducibility as a slower opt-in; this project inverts
  the default and omits the fast side. Reopen trigger: a named
  consumer asks; it then ships as a separate artifact under a
  different name with its own tolerance contract, never as a flag on
  the reproducible build, so the relaxed-opcode gate on `blas.wasm`
  and `blas.threads.wasm` stays absolute.
- **Shared memory is a feature, never a requirement; message-passing
  workers are rejected for the flat API.** Mandatory isolation
  headers break third-party iframes, ads, embedded video, payment and
  OAuth popups, and cannot be set on many static hosts. Message
  passing copies A into each worker's memory (p copies, done serially
  on the main thread) and copies C slabs back; memory multiplies by
  p; the practical ceiling is ~4× at n = 1024 with a break-even of
  several hundred rows; the host grows a marshaling layer for every
  routine; and it buys no synchrony. Its one advantage, reaching
  users without the headers, serves the population whose matrices are
  too small to benefit. Message-passing *residency* (workers that
  permanently own column slabs, the ScaLAPACK model) is deferred to
  the product layer with the handle API; it is not a BLAS concern.
- **Two products; the Emscripten distribution is dropped.** A static
  library, `cblas.h`, a pthreads threading shim and a Pyodide
  integration proof were the highest-risk and most time-consuming
  step (Emscripten's linker against a Rust object, Pyodide's build
  system, NumPy's test suite), and their consumer had been demoted. A
  C programmer depends on the crate and writes the wrapper lines.
  Reopen trigger: a named consumer asks; a `staticlib` is a one-line
  Cargo change and the header is the standard one.
- **Fortran-style symbols (`dgemm_` by reference) are not built.** No
  consumer in scope; a day of mechanical work from the CBLAS surface
  if one appears.
- **The module will export CBLAS names** (todo C), replacing the
  hand-written flat ABI with its invented argument order. The standard names are known
  to everyone and the mapping from the Rust signatures is mechanical.
- **Row-major is the operand swap**, as OpenBLAS does it. No
  row-major kernels.
- **The book is the crate documentation.** rustdoc renders the prose
  beside the code and compiles the examples; nothing is added for it
  but a header file and a Pages workflow. Rejected: tangling Markdown
  into Rust, and an mdBook site with live includes, which was built
  and replaced; reasons under "The book".
- **Zero runtime dependencies.** The four libm entry points the layer
  needs are vendored in `src/math.rs` (sqrt via the hardware
  instruction, which IEEE requires correctly rounded; hypot ported
  verbatim) and cross-checked bit for bit against libm as a
  dev-dependency on every CI run. `no_std` stays; consumers inherit
  zero transitive baggage.
- **Interleaved complex storage.** Planar storage forces a copy at
  every boundary; interleaved (re, im) is what every consumer holds.
- **No integer BLAS.** No multiply-add throughput gain; overflow
  instead of rounding; the one real integer win (int8 quantized gemm)
  is a different library.
- **No WASI, no memory64.** No syscalls are needed; 4 GB exceeds any
  page's working set; memory64 costs bounds checks and engine
  coverage.
- **No JavaScript in the hot path, no JavaScript LAPACK.** Crosses
  the boundary per call; cannot hold the bit-identity contract.
- **No Rayon on wasm, no threading inside the Rust code.** Rayon
  arrives only via wasm-bindgen-rayon (bindgen glue plus a thread-safe
  allocator in the module), against constraints 1 and 4; atomics and
  work queues inside the kernels are against constraint 4. All
  partitioning is the host's.
- **The wasm is never inlined into the JavaScript as text.** It
  inflates the bytes by a third and prevents the browser from
  compiling while downloading.
- **"Fastest on the web" is not the claim.** Not true on f32 gemm
  against fused-multiply-add kernels while bit-identity holds; the
  claim is reproducibility, zero dependencies, and faster than the
  two named competitors in the browser.
- **Banded and packed storage forms are not planned.** No consumer
  demand.

Two rulings from the tuning campaign are recorded in the code and
stay there: the fused single-pass iamax lost its race and the loss is
in its module docs; the 4-column grouped fused hemv won for c64 and
lost for c32, and `src/L2/chemv.rs` records why it keeps the
single-column pass. The unit-stride-only convention the code
currently follows is superseded by constraint 6 and closes with todo
S3.
