# ROADMAP — blas.wasm, the BLAS for the browser

Plan of record as of 2026-10-25 (Andy, architect; drafted from the
2026-10-17 and 2026-10-24 design conversations; revised 2026-10-24
after the consumer and fast-mode discussion and 2026-10-25 after the
deployment-shape discussion). Supersedes the 2026-10-17 roadmap. `README.md` is the record of what is BUILT; this
file is the record of what comes next and of what was considered,
deferred, or rejected. Dates and decisions go here first; README rows
move as work lands.

## What changed since 2026-10-17, and why

- **Scope narrows to BLAS.** The LAPACK subset becomes a separate
  project that depends on this crate. The two have different shapes
  (OpenBLAS-style library vs decomposition product) and different
  audiences; folding them together diluted both.
- **WebGPU is dropped: no consumer identified.** The audience for
  this BLAS (below) is f64-heavy and works at sizes from tiny to
  around a thousand; WGSL has no f64, and the GPU pays for itself only
  on counts or sizes that audience does not reach. Recorded under
  "considered and rejected" with the reasoning, so it is not
  re-litigated without a named consumer.
- **The consumer is named: scientific computing in the browser.**
  Pyodide, JupyterLite, webR, notebook tools, statistics and
  optimization libraries, education, interactive visualization.
  Inference runtimes (ONNX Runtime Web, TensorFlow.js and their
  relatives) are explicitly OUT of scope as consumers (below).
- **Fast mode is not a product.** The relaxed-SIMD fused
  multiply-add is a known lever worth up to 2× on compute-bound
  Level 3; it is deliberately unused because the product's promise
  is reproducibility and, for the named audience, reproducible mode
  is already the fastest option in the browser. Recorded as a lever,
  not a mode.
- **Shared-memory threading is admitted as a feature, never as a
  requirement.** The old constraint 2 rejected SharedArrayBuffer
  because the cross-origin-isolation headers are a burden on the
  user's deployment. That burden is real; the resolution is feature
  detection with a single-threaded fallback, so no user is asked for
  headers who did not want threads. Message-passing workers are
  rejected for the flat API (reasons below).
- **The competitors are named: OpenBLAS and BLIS**, as deployed in
  the browser (compiled to wasm via Emscripten, generic C kernels).
  The product claim is: **reproducible by construction, verified by
  test, faster than everything else that is reproducible, and within
  2× of the fastest thing that is not.** It is NOT "the fastest gemm
  on the web" (see XNNPACK, Step B).
- **Two products, nothing else (2026-10-25).** A Rust crate on
  crates.io and a JavaScript package on npm. The Emscripten
  distribution (static library, header, threading shim, Pyodide
  integration) is dropped from the plan entirely: highest risk, most
  time, and its consumer was demoted. C programmers get a page in the
  book, not an artifact. The Fortran-style symbols and the current
  hand-written flat ABI are dropped too; the wasm module exports CBLAS
  names. See "Products and surfaces".
- **The documentation becomes a book (2026-10-25).** An mdBook site
  with LaTeX and live code includes, replacing the three READMEs as
  the place the mathematics and the rounding-order arguments are
  explained. Chosen over literal literate programming (tangling
  Markdown into Rust) because it keeps every Rust tool working on
  real source; see Step D and "considered and rejected".

## Where we are

`blas.wasm` is finished as a single-threaded module: four type
layers (f64, f32, c64, c32), 116 routines, SIMD on every path, zero
runtime dependencies, zero imports, ~217 KB, bit-identical across
native and every wasm engine. Raw wasm, no bindgen, no glue. The
crate is `no_std` + `alloc`, has no global state, and allocates only
inside packed gemm (two panel buffers).

Gaps relative to a complete BLAS, recorded in README: transpose and
conjugate-transpose forms of gemm/trmv/trsv/trmm/trsm are not built;
syrk/syr2k/herk/her2k have the N form only; vectors are unit stride
(no `incx`/`incy`); no row-major entry; the module's exports use an
invented argument order rather than CBLAS's; argument errors trap
rather than report. Competing with OpenBLAS means closing all of
these.

## Products and surfaces

**Two products.**

- **The Rust crate** (`blas-wasm`, crates.io). Source, not binary. A
  Rust project adds it as a dependency and its own `cargo build`
  compiles it for whatever target that project uses, native or wasm.
  No C bindings, no wasm inside it, nothing pre-compiled. Idiomatic
  API: enums for transpose/side/triangle/diagonal, slices, explicit
  strides.
- **The npm package** (scoped name, decided before first publish).
  Ready to run. One package containing the JavaScript loader, the
  TypeScript type definitions, and the two pre-built wasm files as
  assets inside the package. Works in Node and in browsers. A
  JavaScript project installs it as one dependency and calls
  functions; it never sees a wasm file. The wasm bytes are NOT
  inlined into the JavaScript as text (that inflates them by a third
  and defeats streaming compilation); they are separate files inside
  the package, and the loader fetches the one the page can use.

Plus the book (Step D) on GitHub Pages and the crate's API reference
on docs.rs, which renders itself.

**One semantic surface, three projections.** The semantics are
netlib BLAS: names, argument meanings, flag meanings, quick-return
rules. Nothing is invented.

- *Rust projection*: the crate's API above. Primary, because the code
  is Rust.
- *CBLAS projection*: the wasm module's export list. `cblas_dgemm(
  order, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C,
  ldc)` and the rest of the standard, 32-bit `int`, `CblasRowMajor`
  by the standard operand swap. Produced by the `abi/` crate in this
  repo, which is a BUILD STEP for the npm package and is never
  published on its own. These are export NAMES, chosen because the
  module has to export something and the standard names are known to
  everyone; they are not a channel anyone maintains for C users.
- *JavaScript projection*: CBLAS names and argument order on typed
  arrays, memory management hidden, promises on threaded calls. The
  model is stdlib's `@stdlib/blas`, the one existing package serving
  numerical users in JavaScript: its users already know these
  conventions.

**What is not shipped, and why.** No static library, no `cblas.h`,
no Fortran-style `dgemm_` symbols, no Emscripten anything. A C
programmer who wants this code inside their own wasm program depends
on the crate, writes a handful of `extern "C"` wrapper lines (the
`abi/` source is a worked example), and builds as a library; the book
has a page for it. Anyone who is considering a C dependency already
knows how to do that. Pyodide-style consumers get the same page plus
a note that reference LAPACK links against any BLAS, so they do not
wait on the LAPACK project; their integration is their work, recorded
here as such.

**Why two wasm files, stated once.** A wasm file declares in its
header whether its memory is ordinary or shareable between workers,
and the browser enforces the declaration. A file declaring shared
memory cannot be loaded on a page without the isolation headers, not
even to run on one thread, because the page is forbidden to create
the shared memory object the file demands. A file declaring ordinary
memory cannot be opened by several workers at once. No single file
does both. So the decision "threads when the headers are present,
single-threaded otherwise" forces two files from the same source,
differing in one linker setting. The user never sees this: the npm
loader picks one at startup and the browser downloads only that one.
The cost is a doubled gate (both files built, probed, smoke-tested).

**What this shape is and is not.** It is the minimum for a Rust core
whose primary audience writes JavaScript, with precedent in
tree-sitter and Automerge (Rust or C core, crate, npm with wasm
inside). It is NOT how numerical libraries usually ship: OpenBLAS and
BLIS ship C, faer ships a crate, stdlib ships npm, and bindings are
left to the community. A numerical analyst will not be confused by
either channel, because inside each one the calls look exactly like
BLAS, but they would not expect a BLAS to ship this way, and the
README should not pretend otherwise. The user this serves is the one
stdlib already serves: a numerical analyst working in JavaScript. For
that person the deployment is JavaScript-shaped and the API is
BLAS-shaped, and that satisfies both halves.

## Constraints (standing, architect decisions, revised 2026-10-24)

1. **Our build stays pure.** No Emscripten, no wasm-bindgen, no libc
   layer in the artifacts we build. Nothing is built FOR Emscripten
   either; a consumer who wants to link our code into their own
   program does so from the crate (book page), not from an artifact
   we ship.
2. **Shared memory is a feature, never a requirement.** The host
   detects `crossOriginIsolated`; with it, the threaded artifact runs
   on one shared memory; without it, today's single-threaded artifact
   runs unchanged. The single-threaded path stays first-class: it is
   measured, tested, and shipped on every release, never a fallback
   that stops being looked at.
3. **Bit-identity is kept, and now includes thread count.** Same
   inputs, same bits, on every engine and target, and for any number
   of workers. No relaxed-SIMD instruction (fused multiply-add
   included) ever appears in a shipped artifact; the gate scans the
   binary for relaxed opcodes and fails on any (Step 0). The promise
   is a property of the artifact, never a runtime flag.
4. **Threads are a host concern; kernels never synchronize.** The
   module contains no atomics, no locks, no thread-local state. The
   host partitions disjoint output regions and joins. Parallelism is
   a property of the JS package, never of the Rust code.
5. **Numbers before routing, measured where they apply.** Kernel
   dispatch thresholds (tile/packed) are properties of the code and
   are raced on the reference runners with two draws, as before.
   Thread fan-out thresholds are properties of the user's machine and
   are calibrated at runtime (Step T); CI races verify scaling, they
   do not set the constants.
6. **A complete BLAS, netlib semantics.** Every op/side/uplo/diag
   flag, vector strides, quick-return rules (`alpha == 0`, `beta == 0`
   means C is not read, zero dimensions), and the reference argument
   checks. Where reference BLAS and this crate differ today, the
   reference wins unless the deviation is recorded here with a reason.

## The three claims, and what each requires

| claim | status | what makes it true |
|---|---|---|
| **Reproducible.** Bit-identical results across engines, targets, and thread counts, promised, tested, and auditable from the binary. Nobody else in the browser offers this. | true single-threaded today; thread-count part untested; binary audit not yet in the gate | Step T's p=1 vs p=N bitwise tests per routine; Step 0's relaxed-opcode scan |
| **Dependency-free.** One raw wasm file, zero imports, no toolchain glue, no libc. | true | stays true for `blas.wasm`; `blas.threads.wasm` imports exactly one thing (shared memory) and nothing else |
| **Faster than OpenBLAS and BLIS in the browser.** | plausible, unmeasured | Step B, same engine, same machine, two draws, both single- and multi-threaded |

Not claimed: faster than XNNPACK on f32 gemm. XNNPACK uses the
relaxed-SIMD fused multiply-add, which doubles the arithmetic peak on
every modern core; a bit-identical kernel cannot use it. The honest
expectation is ~2× behind on sgemm while ahead on f64, complex,
and every non-gemm Level 3 routine. Publish the row anyway.

Where the reproducibility comes from, stated honestly: wasm's float
semantics are deterministic by specification, so ANY BLAS compiled to
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
nothing; the tax is confined to compute-bound gemm-like shapes and is
at most 2× there.

## Step 0 — housekeeping (before anything is announced)

- README: remove the dangling parent-repo paths (`../STATUS.md`,
  `../docs/…`, `../bench/`, `cd blas`), the stale "CI gate line still
  needs adding" note (the gate runs `cargo test` already), and the
  references to the LAPACK layer "in the parent repo". A stranger
  must be able to read it end to end.
- Publish `blas-wasm` on crates.io (drop `publish = false`); Rust
  consumers are told to depend on the crate, so it must exist.
- README's first paragraph states that the decompositions (solve,
  inverse, eigen, SVD) are a separate project, so nobody installs
  this looking for `solve`. It also states the deployment shape
  plainly: a crate and an npm package, and why.
- Gate: assert the export count and the import count (zero) of the
  built module, so Step S cannot silently change either.
- Gate: scan every shipped artifact for relaxed-SIMD opcodes
  (`wasm-objdump` or a small reader over the code section) and fail
  on any occurrence. This turns constraint 3 into something anyone
  with the binary can verify.

## Step S — complete the BLAS surface

The largest step and the precondition for every claim. Done through
the packing layer wherever possible so the microkernels do not change.

**S0 — Rust-side flags as enums.** The crate API gains `Trans`,
`Side`, `Uplo`, `Diag` enums (netlib meanings) in place of the
current booleans and separate-function pairs, so the forms added
below have one idiomatic home and the CBLAS projection is a mechanical
mapping. Existing entry points stay as thin aliases until the next
major version.

**S1 — Level 3 op flags via packing.** gemm gains `transa`/`transb`
in {N, T, C}; syrk/syr2k/herk/her2k gain `trans`; trmm/trsm gain
`transa` (and C for complex). The transposed operand is packed into
the SAME panel format from a transposed read; the kernel is unchanged
and never knows. Below the packed threshold (tile/col4 zones) the
transposed read is a strided column gather into a small scratch, then
the existing shape. Bit-identity test: `gemm(T, N)` equals
`gemm(N, N)` on an explicitly transposed copy, BITWISE, because the
panel contents are identical. Same for every flag combination.

**S2 — Level 2 op flags.** gemv already has N/T/C as separate entry
points; trmv/trsv gain `trans` (the transposed triangular forms are
the `gemv_t`-shaped reduction stream per column — RS, not CA; new
code, same contract).

**S3 — vector strides.** `incx`/`incy` on every Level 1 and Level 2
vector argument. Unit stride keeps the SIMD streams. Non-unit stride
takes a path that reuses `lanes.rs`'s lane emulation so the fold
order is identical and results stay bit-identical to the unit-stride
call on a gathered copy (tested bitwise). Negative strides per
reference semantics. This also unblocks LU row swaps downstream.

**S4 — workspace, no allocation.** Packed gemm's two panel buffers
become a caller-provided workspace (`work`, `lwork`), LAPACK style,
with a query form that returns the size. After S4 the crate never
allocates: `alloc` is dropped. Required by Step T (no allocator
inside a shared memory).

**S5 — reference semantics.** Quick returns, `beta == 0` not reading
C, zero dimensions, NaN propagation as reference BLAS; the complex
symmetric (non-Hermitian) symm/syrk/syr2k, which exist in reference
BLAS and are the one family missing outright; `zrot`/`crot`
(complex-s rotation).

**S6 — argument errors are reportable.** The crate keeps trapping on
memory-bounds violations (a trap is the right answer to an invalid
pointer). Argument errors (negative n, bad flag, lda < m) are
reported, not trapped: the module gets an exported error slot and
returns without computing, mirroring reference BLAS's
xerbla-then-return; the JavaScript layer reads it and throws.
Decision recorded: a zero-import module cannot call a host error
handler, so the slot replaces the callback.

Testing contract carries over unchanged for every new form: bitwise
replay against the same-order scalar definition, bounds against
higher-precision references, residuals for solves, native ↔ wasm
probes extended to the new forms. Export count grows well past 116;
the gate assertion in Step 0 moves with it, deliberately.

## Step C — the module's export surface becomes CBLAS

The `abi/` crate is rewritten so the wasm module exports the CBLAS
names and argument order (`cblas_dgemm(order, transA, transB, m, n,
k, alpha, A, lda, B, ldb, beta, C, ldc)`, 32-bit `int`, the full
standard), replacing today's invented flat signatures. `CblasRowMajor`
is the standard operand swap into the column-major kernels; no
row-major kernels. Argument validation writes the error slot (S6) and
returns. Each wrapper is a few lines generated mechanically from the
Rust signatures, the same way the current ABI was written; the
uniform shape is kept. `wasm_alloc`/`wasm_free` stay as the two
non-BLAS exports.

This is a naming change to a build step, not a new channel: the
`abi/` crate remains unpublished and exists only to produce the two
wasm files for the npm package. No header is shipped; the standard
`cblas.h` already describes these exports for anyone who wants one.
Fortran-style symbols are not built (no consumer in scope; a day of
mechanical work from this surface if one appears).

## Step T — threading by shared memory

**T1 — the threaded artifact.** A second build of the identical
kernels: `blas.threads.wasm`, target features `+atomics,+bulk-memory,
+mutable-globals`, memory IMPORTED and shared, data segments
initialized once (the linker's shared-memory mode). The kernels use
no atomics; the features exist only so the module can be instantiated
N times over one `WebAssembly.Memory({shared: true})`. Each instance
gets its own stack region (exported stack-pointer setter or linker
flag, decided at implementation); after S4 there is no allocator and
no global state, so nothing else is per-instance. Two artifacts ship:
`blas.wasm` (zero imports, as today) and `blas.threads.wasm` (one
import: the memory). Both built and smoke-tested by the gate.

**T2 — the partitioner (JS package, Step J).** A worker pool, each
worker holding one instance over the shared memory. Fork-join on
disjoint output regions:

- gemm, symm_left/hemm_left, trmm_left, trsm_left: column slabs of
  the output. Each column is computed by the same kernel in the same
  order as the whole call, so the result is bit-identical for any
  slab width (the size-dispatch invisibility already proven in
  `tests/README.md` is the same property).
- trmm_right, trsm_right, symm_right/hemm_right: columns are coupled;
  partition by ROW slabs (pointer offset + the same column stride —
  the kernels already take `cs`, so a row slab is just a view).
- syrk/herk/syr2k/her2k: column blocks of the stored triangle, widths
  chosen so the triangular work per block is equal, not the column
  count.
- Levels 1 and 2: never. Memory-bound; workers share one bus.

**T3 — call shape and thread count.** From the main thread a threaded
call is asynchronous (the main thread may not block in
`Atomics.wait`); from inside a worker the package offers the
synchronous fork-join as well. Default thread count from
`navigator.hardwareConcurrency`, capped, overridable by the caller
(the `OPENBLAS_NUM_THREADS` equivalent). Without
`crossOriginIsolated`, every entry point runs single-threaded on
`blas.wasm` with the same signatures; the caller's code does not
change.

**T4 — thresholds by runtime calibration.** A short probe at package
init (one gemm at two sizes, p=1 vs p=N), cached per origin, sets the
fan-out threshold per routine class. CI runs the scaling race on the
reference runners with two draws to VERIFY the shape of the curve and
catch regressions, never to bake in a constant: GitHub runners are
not the user's phone or workstation.

**T5 — the reproducibility test.** For every Level 3 routine, every
type, odd sizes that exercise tails and the block edges: p=1 vs p=2
vs p=N, compared BITWISE, native and wasm. This test is what turns
constraint 3's thread-count clause from an argument into a fact.
Lands with T2, not after.

Honest expectations, to be replaced by measurements: Level 3 above a
few hundred rows, roughly 3–5× on eight cores for gemm, less for the
bandwidth-bound rows (syrk/syr2k sit at ~40% of peak in f64 because
they are memory-limited and will scale worst). Levels 1–2 unchanged.
Users without the headers: unchanged.

## Step D — the book

An mdBook site (the tool behind the Rust Book, Rust by Example, the
Cargo book): a folder of Markdown chapters plus a `SUMMARY.md` table
of contents, built by one command into a static site with a sidebar,
search, light/dark themes, and next/previous navigation; deployed to
GitHub Pages by a workflow on every push to main; previewed locally
with `mdbook serve`.

Why this and not literate programming in the literal sense (Markdown
source tangled into Rust): Rust has no line directive, so every
compiler error, clippy warning, test failure and panic would cite the
generated file and be mapped back by hand; rust-analyzer, rustfmt and
go-to-definition would work on generated code; diffs would show prose
churn or generated code; and literate layouts are a known barrier to
outside contributors, which a crate and an npm package are asking
for. Knuth's reason for tangling, ordering code for the reader rather
than the compiler, does not apply in Rust, where item order is free
and one file per routine already groups by concept. The book is the
inverse: prose includes code, code stays canonical.

- **D1 — structure.** One chapter per routine family (per level),
  each giving the mathematics in LaTeX (`mdbook-katex`, rendered at
  build time, no client script), the rounding-order argument that
  makes the bit-identity claim true for that family, the lane
  geometry, the dispatch thresholds and the measured results. A
  chapter on the testing contract, one on the measurement method, one
  on the deployment shape, and one page for C programmers (how to
  wrap the crate in `extern "C"` and build a library, using `abi/` as
  the worked example).
- **D2 — live code.** Source regions are marked with `// ANCHOR:
  name` / `// ANCHOR_END: name` comments and pulled into chapters by
  mdBook's include directive. The code in the book is the code that
  ships; a removed or renamed anchor fails the build.
- **D3 — verification.** `mdbook test` compiles and runs the Rust
  code blocks written directly in chapters, so standalone examples
  cannot rot. The gate builds the book; the Pages workflow deploys
  it. `mdbook-mermaid` renders the call-graph diagrams now in
  `src/README.md`.
- **D4 — migration.** The content of `README.md`, `src/README.md`,
  `tests/README.md` and `bench/README.md` moves into chapters; the
  READMEs shrink to a paragraph and a link. The API reference stays
  in rustdoc on docs.rs, with the KaTeX header so module docs render
  the same mathematics; book and reference link to each other.

## Step B — benchmarks owed

Same engine, same machine, two runner draws, single-threaded AND
threaded (p = hardware threads), all four gemm types plus trsm and
syrk. Rows:

- **OpenBLAS compiled to wasm via Emscripten.** No official wasm
  target exists upstream; record the exact configuration used
  (generic target, flags, whether its pthreads build works under
  Emscripten) so the row is reproducible. This is the OpenBLAS users
  actually get in a browser: generic C with little SIMD.
- **BLIS compiled the same way.** Same note: generic configuration,
  record it. Expected to land near OpenBLAS.
- **XNNPACK f32 gemm** (as used by TensorFlow.js / ONNX Runtime Web).
  The row we expect to lose, by ~2×, for the fused-multiply-add
  reason above. Published so the "within 2× of the fastest thing that
  is not reproducible" claim is a measurement, not a slogan.
- **stdlib's `@stdlib/blas`**, the one existing BLAS for JavaScript:
  faithful, reference semantics, well tested, pure JavaScript with a
  few wasm variants, no SIMD, no threads. This row scopes the
  JavaScript claim: not "no BLAS exists for JavaScript" but "no FAST
  one does".
- **A pure-JS matrix library** (e.g. ml-matrix), because that is what
  much of the web audience currently uses.
- **faer** stays as the historical row (`bench/README.md`); no
  transitive inference from any native comparison is valid.

Prior until measured: ahead of OpenBLAS and BLIS in wasm on complex
and on every non-gemm Level 3 routine; ahead on dgemm; near or ahead
on sgemm against their generic kernels; behind XNNPACK on sgemm.
Why we win where we win: hand-written simd128 kernels on every path,
not the absence of Emscripten — Emscripten does not slow a hot loop,
and the claim must not be phrased as if it did.

## Step J — the npm package (thin)

- Scoped npm name (decided before first publish; `linalg.js` and
  `linalg` are taken; a scoped name is the safe default).
- Contents: the loader, TypeScript type definitions, `package.json`,
  and the two wasm files as assets. ESM, works in Node and browsers.
  The wasm is never inlined into the JavaScript.
- Loads `blas.wasm` or `blas.threads.wasm` by feature detection
  (T3); exposes the CBLAS-named functions on typed arrays with
  explicit dtype, dimensions, strides and flags, following stdlib's
  argument conventions; validates arguments in JS before calling (a
  trap kills the instance; after a trap the package re-instantiates);
  reads the error slot (S6) and throws. Views over `memory.buffer`
  are never cached across a call that may grow memory.
- Opt-in resident buffer handle for callers who chain operations
  (copy in once, compute many, copy out once). This is the ONLY
  product-layer construct in scope; the full residency/handle API
  belongs to the deferred product layer.
- Publishing: a job on the release workflow publishes to npm on tag,
  alongside the crates.io publish, so one tag is one version in both
  places. The release page keeps the tag's build outputs attached
  because the existing workflow does that; nothing on it is a
  product.

## Milestone 1 — "the reproducible BLAS for the web"

Done when all of the following hold:

1. Full netlib surface, all flags and strides, four types, in the
   Rust API and as CBLAS-named module exports (S, C).
2. `blas.wasm` and `blas.threads.wasm` both built, both tested, both
   bit-identical to native and to each other at every thread count,
   both free of relaxed opcodes (0, T1, T5).
3. The Step B table is published with OpenBLAS, BLIS, XNNPACK and
   stdlib rows, both thread modes, two draws.
4. The crate is on crates.io and the package is on npm, one tag, one
   version (0, J).
5. The book is live on GitHub Pages with every routine family
   covered and the READMEs reduced to pointers (D).

Sizing, honestly (engineer-days of focused work; calendar time with
the agent-assisted workflow this repo has used runs far shorter):

| step | days | risk |
|---|---|---|
| 0 housekeeping, crates.io, gate assertions, opcode scan | 1–2 | low |
| S0 enums | 1–2 | low |
| S4 workspace | 1 | low |
| S1 Level 3 flags via packing, four types, full contract | 15–25 | medium |
| S2 Level 2 flags | 3–5 | low |
| S3 strides | 5–8 | low |
| S5 reference semantics, complex symmetric family, zrot/crot | 4–6 | low |
| S6 error slot | 1–2 | low |
| C CBLAS exports | 2–4 | low |
| T threading: build, partitioner, API, calibration, p=1 vs p=N | 10–15 | medium |
| B benchmarks: build OpenBLAS and BLIS for wasm, stdlib, XNNPACK via TF.js, two draws | 5–8 | medium |
| J npm package | 3–5 | low |
| D book: tooling, chapters, anchors, migration | 6–10 | low |

Roughly 60–95 days. S is the bulk and the critical path, and also the
lowest risk because every form inherits an existing pattern and
contract. T1 has fiddly linker work (per-instance stacks, shared data
segments). B depends on OpenBLAS and BLIS building under Emscripten
at all, which neither officially supports. Nothing is research; all
of it is known engineering with a clear finish line.

## Order of work

1. **Step 0** (housekeeping, crates.io).
2. **D1–D3 scaffolding** (the book exists and builds in CI, even if
   thin) so every later step writes its chapter as it lands rather
   than migrating prose afterwards.
3. **S0, S4** (enums; workspace, no allocation) — small, and S4
   unblocks T.
4. **S1–S3, S5, S6** (surface completion), racing each new L3 form
   against the existing one so no regression hides in the packing.
5. **C** (CBLAS exports), mechanical once S is done.
6. **T1–T5** (threaded artifact, partitioner, calibration, the
   reproducibility test).
7. **B** (the full table) — last, because it measures the finished
   thing; the OpenBLAS/BLIS single-threaded rows may be taken early
   as a cheap sanity check, but the claim is made from the final run.
8. **J** publishes alongside 6–7; **D4** migration completes with it.

## Deferred (not rejected; scope fixed here for when they restart)

- **The LAPACK subset** — separate project, consumer of this crate.
  Scope rule stays: the routines `numpy.linalg` calls, per type
  (gesv/getrf/getrs, potrf/potrs, geqrf/orgqr/ormqr, gels, syev/heev,
  gesvd, geev), sequenced L1 factorizations → symmetric eigen → SVD →
  general eigen. It drives BLAS gaps back here; after Step S the known
  ones (transposes, row swaps via strides) are closed. Its own
  conventions must include `info` return; the trap-on-error style
  does not carry over.
- **Message-passing residency** — workers that permanently own
  column slabs of a distributed matrix (the ScaLAPACK model) are the
  right substrate for headerless parallelism under a handle API.
  Belongs to the product layer with the handle API; not a BLAS
  concern.

## Recorded levers, deliberately unused

- **Relaxed-SIMD fused multiply-add ("fast mode").** Worth up to 2×
  on compute-bound Level 3 (less where memory binds; expect 1.3–1.7×
  on dgemm), and the only route to parity with XNNPACK on sgemm.
  Not shipped, because (a) the product's promise is reproducibility
  and a mode that forfeits it dilutes the one claim no competitor can
  match; (b) for the named audience, which is f64, reproducible mode
  is already the fastest option in the browser, so the lever would
  make us faster than ourselves, not catch anyone; (c) it is a second
  kernel set with its own tolerance contract. Precedent noted: Intel
  MKL offers Conditional Numerical Reproducibility as a slower
  opt-in; we invert that default and omit the fast side. If a NAMED
  consumer ever asks, it ships as a separate artifact under a
  different name with its own tolerance contract, never as a flag on
  the reproducible build, so the relaxed-opcode gate on `blas.wasm`
  and `blas.threads.wasm` stays absolute.

## Considered and rejected (so they are not re-litigated)

- **Message-passing workers for the flat BLAS API.** Every call
  copies A into each worker's memory (p copies, done serially on the
  main thread) and copies C slabs back; memory multiplies by p; the
  practical ceiling is ~4× at n = 1024 with a break-even of several
  hundred rows; the host grows a marshaling layer for every routine;
  and it buys no synchrony (a threaded call from the main thread is
  async under both designs). Its one advantage, reaching users
  without the isolation headers, serves the population whose matrices
  are too small to benefit. Shared memory with a single-threaded
  fallback dominates it for one-shot calls.
- **Mandatory isolation headers.** COEP breaks third-party iframes,
  ads, embedded video, payment and OAuth popups, and cannot be set on
  many static hosts. Threads are a feature, never a requirement
  (constraint 2).
- **Emscripten / pthreads emulation for OUR build.** A libc port
  plus JS shims; nothing it provides is needed by the kernels.
  Constraint 1.
- **An Emscripten-linkable distribution (static library, `cblas.h`,
  pthreads threading shim, Pyodide integration proof).** Rejected
  2026-10-25. It was the highest-risk and among the most
  time-consuming steps (external toolchains: Emscripten's linker
  against a Rust object, Pyodide's build system, NumPy's test suite),
  and its consumer had already been demoted. A C programmer who
  wants this code in their program depends on the crate and writes
  the wrapper lines (book page). Pyodide-style projects can do the
  same, and reference LAPACK links against any BLAS so they need not
  wait on the LAPACK project; that integration is their work. If a
  named consumer asks, a `staticlib` is a one-line Cargo change and
  the header is the standard one.
- **Fortran-style symbols (`dgemm_` by reference).** No consumer in
  scope after the above; a day of mechanical work from the CBLAS
  surface if one appears.
- **Literate programming by tangling Markdown into Rust.** No line
  directive in Rust, so every diagnostic cites generated code;
  rust-analyzer, rustfmt and go-to-definition work on the wrong file;
  diffs and review degrade; outside contributors are deterred. The
  book with live includes (Step D) gives the reader the same
  experience while the code stays canonical. Tangling earns its place
  only when the exposition is the primary artifact and the code
  secondary, as in a textbook or thesis; here the code is primary.
- **Inlining the wasm into the JavaScript as text.** Inflates the
  bytes by a third and prevents the browser from compiling while
  downloading. Separate files inside the npm package is the standard.
- **Rayon on wasm.** Only via wasm-bindgen-rayon: bindgen glue plus a
  thread-safe allocator in the module. Constraint 1 and 4; after S4
  there is no allocator to make thread-safe.
- **Threading inside the Rust code (atomics, work queues).**
  Constraint 4: the kernels never synchronize. All partitioning is
  the host's.
- **One wasm file for both threaded and single-threaded use.** Not
  possible under browser policy: a file declaring shared memory
  cannot load on a page without the isolation headers, even to run
  on one thread, and a file declaring ordinary memory cannot be
  shared by workers. Loading a shared-memory file against ordinary
  memory is rejected as a type mismatch; defining the memory inside
  the module instead of importing it changes nothing. Two files from
  one source, chosen by the loader, invisible to the user.
- **Row-major kernels.** CBLAS row-major is the standard operand swap
  into column-major kernels, as OpenBLAS does it.
- **WASI or memory64.** No syscalls needed; 4 GB exceeds any page's
  working set; memory64 costs bounds checks and engine coverage.
- **Integer BLAS.** No multiply-add throughput gain; overflow instead
  of rounding; the one real integer win (int8 quantized gemm) is a
  different library.
- **Planar complex storage.** Forces a copy at every boundary;
  interleaved (re, im) is what every consumer holds.
- **A JavaScript LAPACK, or a JavaScript anything in the hot path.**
  Crosses the boundary per call; cannot hold the bit-identity
  contract.
- **"Fastest on the web" as the claim.** Not true on f32 gemm against
  fused-multiply-add kernels while bit-identity holds; the claim is
  reproducibility, zero dependencies, and faster than the two named
  competitors in the browser.
- **WebGPU in this project.** No consumer identified. The audience is
  f64-heavy and WGSL has no f64; its matrices rarely exceed ~1024 and
  at that size the single-threaded CPU path is already well under a
  second; the one honest GPU argument in numerical work is many small
  matrices (a gemm per particle or pixel), which is a batched API that
  neither Pyodide nor reference LAPACK calls. A GPU path also forfeits
  bit-identity (vendor-specific accumulation and contraction, no WGSL
  strict mode), so it could never be automatic. Revisit only with a
  named consumer and a batched-shaped need. (A WGSL-shaped GPU BLAS
  for inference runtimes would be cuBLASLt-shaped — f16/f32, int8 and
  int4 weights dequantized in-kernel, batched, fused epilogues,
  asynchronous — a sister project sharing nothing with this one but
  the measurement discipline, and one whose consumer must be
  confirmed before it is started.)
- **Inference runtimes (ONNX Runtime Web, TensorFlow.js, and
  relatives) as consumers.** Their hot path is f32, f16 and int8 gemm
  where the relaxed dot-product and fused multiply-add instructions
  are the whole game; token-by-token decoding is matrix-times-vector,
  memory-bound, and won by weight quantization rather than a better
  Level 3; and a BLAS is a small fraction of a runtime that is also a
  graph compiler, hundreds of operators, convolution, attention and a
  memory planner. They already carry hand-tuned relaxed-SIMD kernels
  with shared-memory threads. A reproducible float BLAS has nothing
  to offer them, and "a better ONNX on top of this" is a project ten
  times this size.
