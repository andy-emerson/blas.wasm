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
| `cargo test --release` (native, 218 tests: L1 68, L2 54, L3 91, vendored math 4, one doctest) | all pass | local, 2026-10-25, head `96b3248` |
| 12 determinism probe suites, 126 probes (every flag form included) | native ↔ wasm bit-identical | local, 2026-10-25, after S5 |
| `bench.yml` tri-race and gemm-t-race (two draws each, candidates called directly, workspaces outside the timed call) | every fold identical; verdicts and tables in `bench/README.md` | runs 40212210771, 40212213162, 40212214964, 40212217524 |
| `bench.yml` b0 (two draws: OpenBLAS, BLIS, blas.wasm on one runner, one thread) | rows in `bench/README.md` | runs 40213117656, 40213120395 |
| gate workflow on `main` (tests, clippy host + wasm, 90 cross-target determinism probes, C-ABI build + smoke test; before the audit and book steps existed) | success | run 15, head `d803ae5`, 2026-10-25, github.com/andy-emerson/blas.wasm/actions/runs/40159219042 |
| gate workflow on this branch, dispatched by hand | success, every step: tests, clippy host and wasm, 12 determinism probe suites (126 probes), C-ABI build and smoke test (the error slot included), binary audit against `abi/surface.json` (133 exports), the book with warnings denied | run 24, head `3d63750`, github.com/andy-emerson/blas.wasm/actions/runs/40216386043 (run 23 at `96b3248`, the last code commit, also green) |
| the book: `cargo doc --no-deps` with the header and warnings denied; the doctests in `cargo test --release` | builds clean; the one doctest (the `daxpy` chapter's example) passes | local, 2026-10-25, head `767299a` |
| binary audit: `node abi/audit.mjs --self-test` and the audit of the built module | self-test 3 of 3; 0 imports, 133 exports, 333 functions, 128455 instructions decoded, 0 relaxed-SIMD | local, 2026-10-25, head `767299a` |
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

Four more gate the M2 merge. Each was settled by the Agent during the
M2 reviews and is reopened here because it is the Human's to close;
the fuller framing is owed before a ruling.

- **D12 — The reference's zero-multiplier skips.** Reference BLAS
  skips a column whose multiplier is exactly zero in ger, syr, syr2,
  trmv, trsv, trmm and trsm (`IF (X(J).NE.ZERO)`), so a NaN or
  infinity in that column never reaches the output there; this
  crate's streams have no per-column branch and propagate it
  (verified against netlib 3.12). Options: (1) record the deviation
  under constraint 6 and leave the streams branch-free (the current
  text); (2) mirror the skip in those seven routines, a per-column
  test outside the SIMD loop, and race it. Gates: the M2 merge; the
  wording of "reference semantics" in README and DESIGN.
- **D13 — `nrm2` with both an infinity and a NaN.** The scaled
  accumulation returns the infinity where the reference returns NaN.
  Options: (1) record the deviation (current); (2) make the
  accumulation NaN-aware (a small change to the guard, then the
  determinism probes re-run). Gates: the M2 merge.
- **D14 — Where the module's expected export count lives.** The
  reviews found the release workflow asserting a stale count; the
  branch now has `abi/surface.json` as the one owner, read by the
  audit in both workflows. Options: (1) keep the file; (2) revert to
  the two workflow literals and a review rule. Gates: the M2 merge.
- **D15 — When the book's chapters are written.** Today the crate
  documentation is the crate page, the call-graph chapter and one
  routine chapter (`daxpy`); every other routine page carries its
  module header. The approved plan writes the chapters in M5 (todo
  D4). Options: (1) leave D4 in M5; (2) pull it forward, before the
  M2 merge or as M2's successor. Gates: the plan's order; what README
  may call the documentation until then.

Only the Human closes these. D3's ruling (a race, not a pick) is
closed in `DESIGN.md`; what remains open is the closing question
above. D8 (the workspace shape) was settled by constraint 7 and is
recorded as design in `DESIGN.md`, "The surface". Each entry below
names its options, the recommendation, and what it gates.

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
  risk as written. B0 is measured (two draws, `bench/README.md`):
  single-threaded on the runner, blas.wasm sits at 17–28% of native
  OpenBLAS and BLIS on every Level 3 row — gemm near the quarter the
  instruction set alone allows (128-bit lanes without fused
  multiply-add against AVX-512 with it) for the complex types and
  about three quarters of it for the real ones, the triangular and
  rank-k routines at 17–20%, below gemm's own share. Those are native
  rows; the browser rows of Step B, where the competitors also lose
  the fused multiply-add, are the only measurement that can speak to
  "within 2×". Options now: (1) two campaigns before B — K1 moves the
  syrk family and trmm/trsm onto the packing layer, K2 is a gemm
  micro-kernel campaign, both raced two draws; (2) no kernel work,
  weaken the claim to the measurement. Recommendation: 1, K1 first,
  since the triangular and rank-k routines are the larger lever on
  this evidence. Either way: never write success for a miss; the
  claim is weakened before Step B if the campaigns are declined or
  lose.

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

Done on this branch except S3: every flag form in four types with the
D3 race's winners routed (S0, S4, S1, S2), the reference alpha, beta,
dimension and NaN rules, the complex symmetric family and `zrot`/`crot`
(S5), the module error slot (S6), and the B0 rows (gemm at 17–28% of
native OpenBLAS and BLIS, the triangular and rank-k routines at
17–20%; XNNPACK is not packaged for the runner and has no row). What
is built is in `README.md` and `DESIGN.md`, the numbers in
`bench/README.md`, the evidence in the commit log and the checks
above.

- [ ] **S3 — Vector strides on every Level 1 and 2 vector argument.**
  Evidence: the strided call equals the unit-stride call on a gathered
  copy, bitwise; negative strides versus reference. Gated by D7. (5–8)
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
- trmm/trsm rebuilt on the packing layer (D3's third candidate, the
  K1 campaign's shape), which would also change the no-transpose
  shapes; the B0 rows put the triangular and rank-k routines at
  17–20% of the native libraries, below gemm's 17–28%, which is the
  motivation D11 asked for.
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
