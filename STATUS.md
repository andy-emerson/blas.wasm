# STATUS — blas.wasm living status

This document's job: the open work and the latest check results, for
anyone picking the project up — the Human, the Agent, or a
contributor. Three species of entry, per the working agreement
(`AGENTS.md`): a **todo** names the claim it will earn and the
evidence that will earn it; a **bug** is built but wrong, and closing
it leaves the test that would have caught it; a **decision** is a
design fork left open, naming its options and what it gates, and only
the Human closes it. Closed decisions, with their rejected
alternatives and reopen triggers, move to `DESIGN.md`. Nothing here is
a changelog; history is the commit log.

Last updated 2026-10-25.

## Latest checks

| check | result | run |
|---|---|---|
| `cargo test --release` (native, 156 tests: L1 64, L2 40, L3 48, vendored math 4) | all pass | local, 2026-10-25, head `d803ae5` |
| gate workflow on `main` (tests, clippy host + wasm, 90 cross-target determinism probes, C-ABI build + smoke test) | success | run 15, head `d803ae5`, 2026-10-25, github.com/andy-emerson/blas.wasm/actions/runs/40159219042 |
| bench workflow (rooflines, packed race) | on demand; latest scoreboard in `bench/README.md` is from the July 2026 campaign draws | GitHub Actions, `workflow_dispatch` |

The three claims (`DESIGN.md`, "Claims") stand as follows:
*reproducible* is true single-threaded and tested per routine; the
thread-count part is untested (no threaded artifact yet) and the
binary audit is not yet in the gate (todo 0c). *Dependency-free* is
true: zero imports, verified by the smoke test instantiating with no
import object. *Faster than OpenBLAS and BLIS in the browser* is
plausible and unmeasured (todo B).

## Open decisions

Only the Human closes these. Each names its options, the
recommendation, and what it gates; the fuller framing (user and
developer points of view) is in the plan the Human approved on
2026-10-25 and is reproduced here in brief so the repository, not a
chat log, holds it.

- **D3 — Scratch space and the transposed triangular forms.** Gates
  S4's API, S1's size, T1. S1 gets transposes through packing, which
  exists for gemm and the syrk family but not for trmm/trsm. Options:
  (1) workspace on every routine that needs scratch, and transposed
  trmm/trsm by a transposed copy into it — bit-identical to the
  definition by construction, zero new kernels, O(n²) workspace on
  the triangular Level 3 routines; (2) workspace on gemm only, and
  new transposed kernels for trmm/trsm — a tuning campaign per type.
  Recommendation: 1; reopen if a consumer measures the copy as the
  bottleneck at its sizes.
- **D4 — The Rust crate's error convention.** Gates S0 (both
  signature changes in one pass), S6, C. Options: (1) `Result<(),
  Error>` on every routine carrying the reference xerbla argument
  index, with memory-bounds violations still asserting; (2) panics in
  the crate, reporting only at the module boundary; (3)
  `debug_assert` only. Recommendation: 1, landed with S0.
- **D5 — Allocation in the threaded artifact.** Gates C's export
  list, T1, the J1 loader. The allocator's state would live in the
  shared memory, unsynchronized across instances. Options: (1) no
  allocation exports in either artifact — JavaScript owns linear
  memory, the module exports only BLAS, the abi crate becomes
  `no_std`; (2) keep `wasm_alloc`/`wasm_free` with a contract that
  only the coordinating instance calls them; (3) a locked allocator
  in the threaded build. Recommendation: 1.
- **D6 — The JavaScript call shape.** Gates J1's type definitions,
  T3. A function that sometimes returns a promise is not a signature.
  Options: (1) every routine returns a promise; (2) synchronous
  everywhere, threaded Level 3 under an explicit asynchronous
  namespace that also resolves on the single-threaded path; (3)
  synchronous only, threads only from inside a worker.
  Recommendation: 2, with 3's synchronous fork-join also exported for
  callers already in a worker.
- **D7 — How strides appear in the Rust API.** Gates S3. Options:
  (1) netlib integers (`incx: isize` beside the slice, length
  checked) — the existing column-stride convention extended to
  vectors; (2) a strided view type; (3) unit stride only in Rust.
  Recommendation: 1, confirming the convention already in the code.
- **D8 — The workspace API's shape.** Gates S4, C, J1. Options: (1)
  LAPACK style, `work: &mut [T]` plus a per-routine size query; (2) a
  `Workspace` value the caller reuses; (3) exported block-size
  constants only. Recommendation: 1.
- **D9 — Runtime calibration policy.** Gates T4, J2. Options: (1)
  probe at every init, store nothing; (2) cache in the page's storage
  per origin; (3) no probe, thresholds from `hardwareConcurrency` and
  size with an override. Recommendation: 1 with an optional
  host-supplied cache hook, falling back to 3 if the probe is noisy on
  the reference runners.
- **D10 — How a threaded instance is initialized.** Gates T1 and the
  threaded file's export list. Options: (1) export the stack-pointer
  global and let the host set it; (2) an exported
  `thread_init(stack_top)` without which the instance writes the
  error slot and computes nothing; (3) the linker default, one static
  stack region shared by every instance (unsound; listed to be
  rejected). Recommendation: 2.
- **D11 — Level 3 tuning before Step B.** Gates K1, K2 and how
  strongly the speed claim can be stated. At n = 512 dgemm and sgemm
  sit at 55–57% of the measured arithmetic peak and the syrk family
  and trmm/trsm at 38–50% in f64 (`bench/README.md`); XNNPACK runs
  near the fused-multiply-add peak, twice ours, so "within 2×" is at
  risk as written. Options: (1) two campaigns before B — K1 moves the
  syrk family and trmm/trsm onto the packing layer, K2 is a gemm
  micro-kernel campaign, both raced two draws; (2) no kernel work,
  weaken the claim to the measurement; (3) measure first (todo B0),
  then rule. Recommendation: 3. Either way: never write success for a
  miss; the claim is weakened before Step B if the campaigns are
  declined or lose.

## Todos, by milestone

Effort is in engineer-days of focused work, low to high. Each todo
names the claim it earns and the evidence. Order of work is top to
bottom.

### M0 — Readable to a stranger, agreement in force

- [x] **0b — The records exist.** `DESIGN.md` and this file, each
  with a named job and reader; `ROADMAP.md` retired into them.
  Evidence: the documentation review's placement check, read against
  each file's job.
- [x] **0a — No dangling paths or stale notes anywhere in the tree.**
  README opens on what the product is and where the decompositions
  live. Evidence: a grep for the old parent-repository path patterns
  is empty; README read end to end on a fresh clone. (1–1.5)
- [ ] **0c — The gate audits the binary.** Relaxed-SIMD opcode scan
  over the code section of every shipped artifact; import and export
  counts asserted explicitly. Evidence: the scan fails on a
  deliberately relaxed build and passes on `main`. (0.5–1)
- [ ] **`main` is protected** (pull request required, no direct or
  force pushes, merge commits only). Human action, GitHub settings.

### M1 — The book builds

- [ ] **D1–D3 — Book scaffold.** mdBook with KaTeX and Mermaid, at
  least one chapter including source by anchor, `mdbook test` in the
  gate, Pages deploy on push to `main`. Evidence: the gate builds and
  tests the book; a renamed anchor fails the build; the Pages site is
  live. (2–3) Human action after the merge: set the repository's
  Pages source to GitHub Actions.

### M2 — Complete BLAS surface, netlib semantics

- [ ] **S0 — Enums for Trans, Side, Uplo, Diag; error convention per
  D4.** Evidence: the existing 156 tests pass unchanged through the
  new signatures. Gated by D4. (1–2)
- [ ] **S4 — Caller-provided workspace; the crate never allocates.**
  Evidence: no `alloc` in the crate; packed-gemm probe values
  unchanged. Gated by D3, D8. (1–2)
- [ ] **S1 — Level 3 op flags on gemm, the syrk family, trmm, trsm,
  four types.** Evidence: the flagged call equals the unflagged call
  on an explicitly transposed copy, bitwise, every combination; each
  new form raced against the existing one, two draws. (15–28)
- [ ] **B0 — Early sanity rows: OpenBLAS, BLIS, XNNPACK
  single-threaded**, to inform D11. Evidence: same engine, same
  machine, two draws; configurations recorded. (2–3)
- [ ] **S2 — Level 2 op flags: transposed trmv and trsv.** Evidence:
  bitwise replay plus residuals. (3–5)
- [ ] **S3 — Vector strides on every Level 1 and 2 vector argument.**
  Evidence: the strided call equals the unit-stride call on a gathered
  copy, bitwise; negative strides versus reference. Gated by D7. (5–8)
- [ ] **S5 — Reference semantics**: quick returns, NaN propagation,
  the complex symmetric symm/syrk/syr2k, `zrot`/`crot`. Evidence:
  per-rule tests against the reference definitions. (4–6)
- [ ] **S6 — Argument errors reported with the xerbla index; the
  module error slot.** Evidence: every check path returns, never
  traps. (1–2)

### M3 — Release-ready: crate and single-threaded package, nothing published

- [ ] **C — CBLAS-named exports replace the flat ABI.** Evidence: the
  export list equals the standard list for four types; the smoke test
  rewritten; the count asserted. Gated by D5. (2–4)
- [ ] **J1 — npm package, single-threaded**: loader, types,
  validation; the release workflow attaches it; no registry publish
  (D2). Evidence: `npm pack` contents; Node and browser smoke tests; a
  tag produces the release attachments. Gated by D6. (3–5)

### K — Optional Level 3 tuning campaigns (only if B0 and D11 call for them)

- [ ] **K1 — syrk family and trmm/trsm onto the packing layer**, raced
  per type. Evidence: two-draw races; routed only at measured wins;
  replays bit-identical. Gated by D11. (10–20)
- [ ] **K2 — gemm micro-kernel campaign**: blocking, broadcast,
  unroll, raced. Evidence: two-draw races against the shipping kernel;
  losses recorded in the module docs. Gated by D11. (8–15)

### M4 — Reproducible at any thread count

- [ ] **T1 — `blas.threads.wasm`**: nightly build-std, shared memory,
  per-instance stacks. Evidence: both artifacts built, audited and
  smoke-tested in the gate; the threaded file has exactly one import.
  Gated by D10. (4–6)
- [ ] **T2 + T5 — Fork-join partitioner and the p=1 vs p=N bitwise
  test**, landed together. Evidence: bitwise equality for every Level
  3 routine and type at odd sizes, native and wasm. (4–6)
- [ ] **T3 + T4 — Call shape, thread count, runtime calibration.**
  Evidence: scaling race on the reference runners, two draws. Gated by
  D9. (2–3)
- [ ] **J2 — The loader picks the artifact by feature detection.**
  Evidence: a page without isolation headers runs every call
  single-threaded with identical signatures. (1–2)

### M5 — Measured against the named competitors; the book is complete

- [ ] **B — The competitor table**: OpenBLAS and BLIS under
  Emscripten, XNNPACK, stdlib, both thread modes, two draws. Evidence:
  the published table with configurations and run identifiers. (5–8)
- [ ] **D4 — Book migration**: the READMEs become pointers, every
  routine family has a chapter. Evidence: `mdbook test` green;
  merge-scale documentation review. (3–5)

Committed total: 58–99 engineer-days; optional campaigns 18–35 more.

## Levers recorded, not yet raced

From the July 2026 tuning campaign; each is a todo whose evidence is a
two-draw race on the reference runners. `bench/README.md` has the
current numbers.

- The `i*amax` first-index rescans (worst: isamax at 8% of the stream
  ceiling; icamax inherits it). A fused single-pass candidate already
  lost its f64 race.
- Packed-gemm tuning constants: KC and MC are first guesses, not
  swept.
- The unmeasured dispatch gaps: f64 below 256³ versus the tiled
  shape; c32 between 2 MB and 8 MB of A.
- The c32 `hemv` grouping lost (~2%, both draws); `chemv` keeps the
  single-column pass. Re-race only if the lane product form changes.

## Bugs

None open.

## Not in this tree

The step-by-step evidence trail of the July 2026 campaign (the A/B
tables with run identifiers that `bench/README.md` cites by step
number) lives in the predecessor repository the layer was transplanted
from, not here. The numbers quoted in `bench/README.md` and the probe
bit patterns in `tests/README.md` are the parts that were carried
over; the bench workflow regenerates both on demand.
