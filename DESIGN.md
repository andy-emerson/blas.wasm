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
- *The npm package* (name is a placeholder until publishing is
  decided; see D2). Ready to run. One package containing the
  JavaScript loader, the TypeScript type definitions, and the two
  pre-built wasm files as assets inside the package. Works in Node
  and in browsers. A JavaScript project installs it as one dependency
  and calls functions; it never sees a wasm file. The wasm bytes are
  never inlined into the JavaScript as text; they are separate files
  inside the package, and the loader fetches the one the page can
  use.

Plus the book (`book/`, GitHub Pages) and the crate's API reference on
docs.rs, which renders itself.

**One semantic surface, three projections.** The semantics are
netlib BLAS: names, argument meanings, flag meanings, quick-return
rules. Nothing is invented.

- *Rust projection*: the crate's API. Primary, because the code is
  Rust.
- *CBLAS projection*: the wasm module's export list —
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
   artifact, never a runtime flag.
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
   reason.
7. **The crate never allocates.** Scratch space is the caller's,
   passed in and size-queried, LAPACK style. (Today's packed gemm
   allocates two panel buffers; closing that gap is todo S4 and the
   shape of the argument is decision D8.)

## Claims, and what makes each true

| claim | what makes it true |
|---|---|
| **Reproducible.** Bit-identical results across engines, targets, and thread counts, promised, tested, and auditable from the binary. | per-routine bitwise replays and cross-target probes (`tests/README.md`); the p=1 vs p=N bitwise test per routine (todo T5); the relaxed-opcode scan in the gate (todo 0c) |
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
between a register tile and a BLIS/Goto packed-panel shape; the
taxonomy and the per-routine cells are in `README.md`, the call graph
and naming in `src/README.md`. Every tuned shape is bit-identical to
its plain column-axpy reference except where an elimination order
forbids it; those two cases are documented in their files and
mirrored by their tests. Lane emulation off-wasm (`src/lanes.rs`)
reproduces the SIMD lane structure elementwise, which is what makes
native and wasm agree bit for bit.

The hard-won build rule: simd128 is not in rustc's default wasm32
feature set, so every SIMD path sits under `#[target_feature(enable
= "simd128")]` on the whole call chain, or the intrinsics compile as
out-of-line calls (measured 6.4× slowdown).

### Completing the surface (todo group S)

- **Level 3 op flags through packing.** The transposed or conjugated
  operand is packed into the same panel format from a transposed
  read; the kernel is unchanged and never knows. Conjugation is an
  exact sign flip. Below the packed threshold the transposed read is
  a strided gather into a small scratch, then the existing shape.
  The test is the definition: `gemm(T, N)` equals `gemm(N, N)` on an
  explicitly transposed copy, bitwise, for every flag combination.
  This covers gemm and the syrk/syr2k/herk/her2k family. trmm and
  trsm have no packed path; how their transposed forms are built is
  decision D3.
- **Level 2 op flags.** gemv already has N/T/C as separate entry
  points. The transposed trmv and trsv are the `gemv_t`-shaped
  reduction stream per column: new kernels, same contract.
- **Strides.** `incx`/`incy` on every Level 1 and Level 2 vector
  argument. Unit stride keeps the SIMD streams. Non-unit stride takes
  a path that reuses the lane emulation so the fold order is
  identical and results stay bit-identical to the unit-stride call on
  a gathered copy. Negative strides per reference semantics. How the
  stride appears in the Rust signatures is decision D7.
- **Workspace.** Packed gemm's panel buffers become caller-provided,
  with a query form that returns the size; afterwards the crate never
  allocates and `alloc` is dropped (constraint 7, required by
  threading: no allocator inside a shared memory). Scope (D3) and
  shape (D8) are open.
- **Reference semantics.** Quick returns, `beta == 0` not reading C,
  zero dimensions, NaN propagation as reference BLAS; the complex
  symmetric (non-Hermitian) symm/syrk/syr2k; `zrot`/`crot`.
- **Argument errors are reportable.** The crate keeps trapping on
  memory-bounds violations: a trap is the right answer to an invalid
  pointer. Argument errors (negative n, bad flag, lda < m) are
  reported, not trapped. The module gets an exported error slot and
  returns without computing, mirroring reference BLAS's
  xerbla-then-return; the JavaScript layer reads it and throws.
  *Recorded decision:* a zero-import module cannot call a host error
  handler, so the slot replaces the callback. The Rust crate's own
  convention is decision D4.

The testing contract carries over unchanged for every new form:
bitwise replay against the same-order scalar definition, bounds
against higher-precision references, residuals for solves, native ↔
wasm probes extended to the new forms. The export count grows well
past 116; the gate's assertion moves with it, deliberately.

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

An mdBook site: a folder of Markdown chapters plus a `SUMMARY.md`,
built by one command into a static site with sidebar, search, themes
and navigation; deployed to GitHub Pages by a workflow on every push
to `main`; previewed locally with `mdbook serve`.

- One chapter per routine family per level, giving the mathematics in
  LaTeX (`mdbook-katex`, rendered at build time, no client script),
  the rounding-order argument that makes the bit-identity claim true
  for that family, the lane geometry, the dispatch thresholds and the
  measured results. A chapter each on the testing contract, the
  measurement method, the deployment shape, and a page for C
  programmers.
- Source regions are marked `// ANCHOR: name` / `// ANCHOR_END: name`
  and pulled into chapters by mdBook's include directive. The code in
  the book is the code that ships. mdBook renders a missing anchor as
  an empty include without complaint, so `book/check-includes.mjs`
  verifies every include's file and anchor pair and the gate fails on
  a removed or renamed anchor.
- `mdbook test` compiles and runs the Rust code blocks written
  directly in chapters, so standalone examples cannot rot. The gate
  builds and tests the book; the Pages workflow deploys it.
  `mdbook-mermaid` renders the call-graph diagrams.
- The content of the four READMEs migrates into chapters and the
  READMEs shrink to a paragraph and a link. The API reference stays
  in rustdoc on docs.rs, with the KaTeX header so module docs render
  the same mathematics; book and reference link to each other.

Why a book and not literate programming in the literal sense
(Markdown tangled into Rust): Rust has no line directive, so every
compiler error, clippy warning, test failure and panic would cite the
generated file and be mapped back by hand; rust-analyzer, rustfmt and
go-to-definition would work on generated code; diffs would show prose
churn or generated code; and literate layouts are a known barrier to
outside contributors, which a crate and an npm package are asking
for. Knuth's reason for tangling, ordering code for the reader rather
than the compiler, does not apply in Rust, where item order is free
and one file per routine already groups by concept. The book is the
inverse: prose includes code, code stays canonical.

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
  the clone). The former `ROADMAP.md` is retired into this file and
  `STATUS.md`.
- **D2 — Publishing to the registries.** Nothing is published to npm
  or crates.io for now; the package names are placeholders. The
  crates.io publish is out of the housekeeping step, `publish =
  false` stays, and no alias layer is kept for the S0 signature
  change, since no one depends on the crate. Milestone M3 produces
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
  and `blas.threads.wasm` stays absolute. This supersedes the earlier
  "FMA variants deferred" note from the tuning campaign.
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
- **The module exports CBLAS names**, replacing the hand-written flat
  ABI with its invented argument order. The standard names are known
  to everyone and the mapping from the Rust signatures is mechanical.
- **Row-major is the operand swap**, as OpenBLAS does it. No
  row-major kernels.
- **The documentation is a book, not a tangled literate program.**
  Reasons under "The book". Tangling earns its place only when the
  exposition is the primary artifact and the code secondary; here the
  code is primary.
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
