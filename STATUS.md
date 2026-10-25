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
| `cargo test --release` (native, 199 tests: L1 64, L2 53, L3 77, vendored math 4, one doctest) | all pass | local, 2026-10-25, after S2 |
| 12 determinism probe suites, 126 probes (every flag form included) | native ↔ wasm bit-identical | local, 2026-10-25, after S2 |
| `bench.yml` tri-race and gemm-t-race (two draws each) | every fold identical; verdicts in `bench/README.md` | runs 40208127781, 40208130046, 40208724791, 40208727304 |
| gate workflow on `main` (tests, clippy host + wasm, 90 cross-target determinism probes, C-ABI build + smoke test; before the audit and book steps existed) | success | run 15, head `d803ae5`, 2026-10-25, github.com/andy-emerson/blas.wasm/actions/runs/40159219042 |
| gate workflow on this branch, dispatched by hand | success, every step: tests, clippy host and wasm, 12 determinism probe suites (126 probes), C-ABI build and smoke test, binary audit, the book with warnings denied | run 21, head `40782a0`, github.com/andy-emerson/blas.wasm/actions/runs/40210413099 |
| the book: `cargo doc --no-deps` with the header and warnings denied; the doctests in `cargo test --release` | builds clean; 157 tests pass, the `daxpy` chapter's example among them | local, 2026-10-25, after the rustdoc switch |
| binary audit: `node abi/audit.mjs --self-test` and the audit of the built module | self-test 3 of 3; 0 imports, 119 exports, 265 functions, 100732 instructions decoded, 0 relaxed-SIMD | local, 2026-10-25, head `641cecc` |
| bench workflow (rooflines, packed race) | on demand; latest scoreboard in `bench/README.md` is from the July 2026 campaign draws | GitHub Actions, `workflow_dispatch` |

The three claims (`DESIGN.md`, "Claims") stand as follows:
*reproducible* is true single-threaded and tested per routine; the
binary audit is in the gate (`abi/audit.mjs`); the thread-count part
is untested (no threaded artifact yet). *Dependency-free* is
true: zero imports, verified by the smoke test instantiating with no
import object. *Faster than OpenBLAS and BLIS in the browser* is
plausible and unmeasured (todo B).

## Open decisions

- **D3, closing question — is the packed trmm/trsm candidate still
  owed?** D3 named three candidates for the transposed triangular
  forms; two were built and raced (the workspace copy and the no-copy
  kernels), the wrappers route their winners, and the third, trmm/trsm
  rebuilt on the packing layer, was not built: it would change the
  no-transpose shapes as well, and the race margins were set by the
  copy's cost and the kernels' load ratios. Options: close D3 on the
  two-candidate verdicts (the routing stands; the packed candidate
  becomes the lever recorded below), or require the third candidate
  before closing (the routing is provisional until then). Gates: the
  M2 merge recommendation.

Only the Human closes these. D3, D4 and D8 are closed and recorded in
`DESIGN.md`; D8 turned out to be settled by constraint 7 rather than
by a ruling, and D3 is a race, not a pick. Each names its options, the
recommendation, and what it gates; the fuller framing (user and
developer points of view) is in the plan the Human approved on
2026-10-25 and is reproduced here in brief so the repository, not a
chat log, holds it.

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

Done on this branch except the human action below; the exit evidence
is in the commit log and the gate run listed under "Latest checks".

- [ ] **`main` is protected** (pull request required, no direct or
  force pushes, merge commits only). Human action, GitHub settings.

### M1 — The book builds

The book is the crate documentation (rustdoc), per the Human's ruling
of 2026-10-25, replacing the mdBook site. Done on this branch except
the human action below: the crate page, the call-graph chapter and
the `daxpy` chapter render with mathematics and diagrams, the example
runs under `cargo test`, and the gate builds the documentation with
warnings denied.

- [ ] **Pages source set to GitHub Actions** (Settings → Pages), so
  `.github/workflows/pages.yml` can publish `cargo doc` on the first
  push to `main` after the merge. Human action.

### M2 — Complete BLAS surface, netlib semantics

- [x] **S0 — Enums, netlib argument order, `Result` on every routine,
  workspace slices and size queries.** Done on this branch.
- [x] **S4 — The crate never allocates.** Done on this branch:
  `extern crate alloc` is gone (the build is the check), the packed
  gemm shapes run on the caller's slice, probe bit patterns unchanged.
- [x] **S1 — Level 3 op flags on gemm, the syrk family, trmm, trsm,
  four types.** Done on this branch: every flagged call equals the
  unflagged call on an explicitly transposed copy, bitwise, every
  combination (`tests/L3`); the candidates raced on two runner draws
  (`bench/README.md`, D3 in `DESIGN.md`), winners routed, losers
  exported. Not built: trmm/trsm on the packing layer (a lever
  below).
- [ ] **B0 — Early sanity rows: OpenBLAS, BLIS, XNNPACK
  single-threaded**, to inform D11. Evidence: same engine, same
  machine, two draws; configurations recorded. (2–3)
- [x] **S2 — Level 2 op flags: transposed trmv and trsv.** Done on
  this branch: dot-fold reductions, replayed bit for bit (`tests/L2`),
  probed native versus wasm; no routine reports `Unsupported` any
  more and the error kind is gone.
- [ ] **S3 — Vector strides on every Level 1 and 2 vector argument.**
  Evidence: the strided call equals the unit-stride call on a gathered
  copy, bitwise; negative strides versus reference. Gated by D7. (5–8)
- [x] **S5 — Reference semantics.** Done on this branch: the
  `alpha == 0` / `beta == 1` / `k == 0` rules in every wrapper (what the
  reference does not read is NaN-poisoned in the tests), the complex
  symmetric symm/syrk/syr2k and `zrot`/`crot` in both complex types,
  iamax's NaN rule. One known deviation: `nrm2` of a vector holding
  both an infinity and a NaN returns the infinity where the reference
  returns NaN (its scaled accumulation never compares the NaN); every
  other NaN input propagates as in the reference.
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
  routine has a chapter to the `daxpy` standard. Evidence: `cargo doc`
  clean with warnings denied; every chapter's example runs under
  `cargo test`; merge-scale documentation review. (3–5)

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
- A lane version of the complex left transposed trmm/trsm kernels
  (`ztrmm_left_t_k` and twins): the scalar one lost its race to the
  copy 0.26–0.76× (both draws); the real-type lane kernels won 1.3–2.8×,
  so the complex lane form is the obvious next candidate.
- trmm/trsm rebuilt on the packing layer (D3's third candidate), which
  would also change the no-transpose shapes; motivated only if the
  B0 rows show the triangular routines far from the ceiling.
- The f32 left transposed trmm/trsm ascending-walk cases (trmm upper,
  trsm lower) sit at 0.85–1.0 of the copy on 1024² while the
  descending walks win 1.3×; the asymmetry is unexplained.

## Bugs

None open.

## Not in this tree

**The campaign record.** The step-by-step evidence trail of the July
2026 tuning campaign — the A/B tables with their GitHub Actions run
identifiers, numbered by step — lives in the predecessor repository
the layer was transplanted from, not here. `DESIGN.md`,
`bench/README.md`, `tests/README.md` and the module docs under `src/`
cite it as "campaign record step N". The numbers they quote, the run
identifiers, and the probe bit patterns are the parts that were
carried over; the bench workflow regenerates the scoreboard and the
probes on demand.
