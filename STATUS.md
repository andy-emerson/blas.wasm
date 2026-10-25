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
| `cargo test --release` (native, 228 tests: L1 68, L2 54, L3 101, vendored math 4, one doctest) | all pass | local, 2026-10-25, head `e4850c1`; gate run 26 |
| 12 determinism probe suites, 126 probes (every flag form included) | native ↔ wasm bit-identical | gate run 26, head `e4850c1`; the two roofline runs below re-check them |
| `bench.yml` k2-race (two draws: the broadcast kernel against the pre-K2 packed shape, both called directly) | 1.73–1.91× on every shape, folds identical; table in `bench/README.md` | runs 40226735030, 40226834978 |
| `bench.yml` k1-race (two draws: the packed syrk, trmm and trsm against the exported sweeps, both called directly) | 1.36–2.63× on every shape, folds identical; table in `bench/README.md` | runs 40231405123, 40232089887 |
| `bench.yml` rooflines (two draws at the routed head) | probes bit-identical; Level 3 table in `src/L3/README.md`; the real gemm, syrk, trmm and trsm rows read 86–111% of the peak probe, which is bug B1 | runs 40232849090, 40233130801 |
| `node tools/depmap.mjs --check` (the dependency blocks in the 27 source READMEs against the code) | match | gate run 28, head `260c50a`; local at the Level 3 doc pass |
| `bench.yml` tri-race and gemm-t-race (two draws each, candidates called directly, workspaces outside the timed call) | every fold identical; verdicts and tables in `bench/README.md` | runs 40212210771, 40212213162, 40212214964, 40212217524 |
| `bench.yml` b0 (two draws: OpenBLAS-wasm, BLIS-wasm, blas.wasm in one Node on one runner, one thread) | rows and verdict in `src/L3/README.md`: ties OpenBLAS on dgemm and dsyrk, 1.1–1.27× behind on sgemm, dsymm, dtrmm, dtrsm, ahead on the complex types; ahead of BLIS on every row | runs 40232073684, 40232628850 (pre-campaign draws 40220022112, 40220778110) |
| gate workflow on `main` (tests, clippy host + wasm, 90 cross-target determinism probes, C-ABI build + smoke test; before the audit and book steps existed) | success | run 15, head `d803ae5`, 2026-10-25, github.com/andy-emerson/blas.wasm/actions/runs/40159219042 |
| gate workflow on this branch, dispatched by hand | success, every step: tests, clippy host and wasm, 12 determinism probe suites (126 probes), C-ABI build and smoke test (the error slot included), binary audit against `abi/surface.json` (133 exports), the dependency maps, the book with warnings denied | run 28, head `260c50a` (the folder move, the last code commit), github.com/andy-emerson/blas.wasm/actions/runs/40239509801 |
| the book: `cargo doc --no-deps` with the header and warnings denied; the doctests in `cargo test --release` | builds clean; the one doctest (the `daxpy` chapter's example) passes | gate run 26, head `e4850c1`; local at the doc pass that followed |
| binary audit: `node abi/audit.mjs --self-test` and the audit of the built module | self-test 3 of 3; 0 imports, 133 exports, 0 relaxed-SIMD | gate run 26, head `e4850c1` |

The three claims (`DESIGN.md`, "Claims") stand as follows:
*reproducible* is true single-threaded and tested per routine; the
binary audit is in the gate (`abi/audit.mjs`); the thread-count part
is untested (no threaded artifact yet). *Dependency-free* is
true: zero imports, verified by the smoke test instantiating with no
import object. *Faster than OpenBLAS and BLIS in the browser* is
measured single-threaded (B0, `bench/README.md`, two runner draws at
the head that routes K2 and K1): true against BLIS on every row;
against OpenBLAS's wasm port true for zgemm (1.05×) and cgemm
(1.5–1.7×), a tie on dgemm and dsyrk (0.99–1.05×), false on sgemm
(1.11–1.17× behind), dtrmm and dtrsm (1.15–1.23×) and dsymm
(1.24–1.27×). The multi-threaded part is untested (todo B).

## Open decisions

- **D3, closing question — is the transposed trmm/trsm race still
  owed?** D3 named three candidates for the transposed triangular
  forms: the workspace copy, the no-copy kernels, and trmm/trsm
  rebuilt on the packing layer. The first two were raced and the
  wrappers route their winners; the third now exists (K1) for the
  no-transpose forms, and the copy candidate runs it on the copy, so
  the transposed race as recorded (`bench/README.md`, "The
  transposed-form races") predates the kernel on its copy side, and
  the packed kernel on a transposed read (what K2 does for gemm) is
  unbuilt for trmm and trsm. Options: close D3 on the recorded
  verdicts (the routing stands; the re-race and the transposed-read
  candidate become the lever recorded below), or require the re-race
  before closing (the transposed routing is provisional until then).
  Gates: the M2 merge recommendation.

Three more gate the M2 merge. Each was settled by the Agent during
the M2 reviews and is reopened here because it is the Human's to
close; the fuller framing is owed before a ruling. (D15, the book's
chapters, was closed by the Human on 2026-10-25: the family and level
READMEs under `src/` are the chapters, written now; `DESIGN.md`,
"Decisions closed".)

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
- **D11 — Level 3 tuning before Step B: confirm the campaigns, and
  what the speed claim says now.** Gates how the speed claim is
  stated and whether more kernel work precedes Step B. Where it
  stood: B0 measured the way the plan asked (OpenBLAS and BLIS
  compiled to wasm in the same Node on the same runner, two draws)
  put OpenBLAS's own wasm target 1.9–2.3× ahead of this module on
  every f64 and f32 Level 3 row and this module ahead on the complex
  types, with the same simd128 lanes on both sides. The Human's "can
  you fix it?" was read as choosing option 1 of the fork (the K2 and
  K1 campaigns before B) and both ran on this branch: K2 (the gemm
  micro-kernel rebuilt on the mechanics of OpenBLAS's kernel, 1.7–1.9×
  over the pre-K2 shape, two runner draws) and K1 (syrk, trmm and
  trsm onto the packing layer, 1.4–2.6× over the sweeps, two runner
  draws), every result bit-identical to the shapes that shipped. B0
  at the routed head, two draws (`bench/README.md`): a tie with
  OpenBLAS-wasm on dgemm and dsyrk (0.99–1.05×), 1.11–1.17× behind
  on sgemm, 1.15–1.23× on dtrmm and dtrsm, 1.24–1.27× on dsymm; ahead
  1.05× on zgemm and 1.5–1.7× on cgemm; ahead of BLIS on every row.
  Open now: (a) confirm that reading of "fix it" — the routing of
  the campaign kernels is the Agent's recommendation, not yet the
  Human's ruling, and the pre-campaign shapes stay exported so a
  reversal is a dispatch change; (b) whether the remaining 1.1–1.27×
  on sgemm, dsymm, dtrmm and dtrsm is pursued before B (the levers
  below: sgemm's 8×4 tile against OpenBLAS's, the triangular
  diagonal routines, dsymm onto the packing layer) or the claim is
  stated as "ahead of BLIS; ahead of OpenBLAS on the complex types,
  level on dgemm and dsyrk, within 1.3× elsewhere" and B proceeds.
  Recommendation: (a) yes; (b) state the claim as measured and
  proceed to B, racing the levers only if B's multi-threaded table
  reopens the question. Rejected along the way: fusing multiply and
  add (relaxed SIMD) — it would double the arithmetic peak and break
  the reproducibility claim (constraint 2).

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
(S5), the module error slot (S6), the B0 rows (OpenBLAS and BLIS
compiled to wasm in the same Node) and, read from D11, the K2 and K1
campaigns, after which this module ties OpenBLAS-wasm on dgemm and
dsyrk, trails it 1.1–1.27× on sgemm, dsymm, dtrmm and dtrsm, leads it
on the complex types and leads BLIS everywhere (XNNPACK is not
built). What
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

### K — Level 3 tuning campaigns (run under D11's option 1; the ruling is open)

- [x] **K2 — gemm micro-kernel campaign.** Done for f64 and f32: α
  folded into the B panel, B broadcast by lane shuffles, no scalar
  work in the k loop; 1.73–1.91× over the pre-K2 packed shape on every
  shape (runs 40226735030, 40226834978) and 1.2–2.3× over the register
  tile from 128³ up (container), so the tile's zone closed; a KC/MC
  sweep moved nothing. Bit-identical to the shipping shapes (tests).
  The losers stay exported. Not done: the complex gemms keep their
  kernels (they already lead OpenBLAS-wasm).
- [x] **K1 — syrk, trmm and trsm onto the packing layer.** Done for
  f64 and f32, both sides, both triangles: 1.36–2.63× over the sweeps
  on every shape (runs 40231405123, 40232089887), bit-identical to the
  sweeps, which stay exported. Not done: syr2k (still the sweep,
  38–56% in the Level 3 table), the complex triangular routines, the
  transposed forms (they run the packed kernel on a transposed copy,
  the stream-versus-copy race for them not re-run since K1).

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
- [ ] **D4 — The chapters** (pulled forward from M5 by D15; in
  progress on this branch): one README per family and per level under
  `src/`, included as the module's documentation, with the generated
  dependency maps. Done: the three level chapters and all 23 family
  chapters (Level 3: gemm, symm, syrk, syr2k, trmm, trsm; Level 2:
  gemv, ger, symv, syr, syr2, trmv, trsv; Level 1: axpy, scal, copy,
  swap, dot, nrm2, asum, iamax, rot, rotg), the source map
  (`src/README.md`, also the `book::call_graph` chapter), the
  scoreboards moved out of `bench/README.md` into the level chapters,
  the Level 3 routine headers trimmed to what differs by type. Not
  done: the Level 2 and Level 1 routine headers still carry their
  pre-chapter prose (true, but a second copy of what the chapters
  say); a runnable example per chapter to the `daxpy` standard; a
  Level 1 and 2 scoreboard draw at the current head (the rows are the
  July 2026 campaign's). Evidence: `cargo doc` clean with warnings
  denied; `tools/depmap.mjs --check`; merge-scale documentation
  review. (3–5)

Committed total: 58–99 engineer-days; optional campaigns 18–35 more.

## Levers recorded, not yet raced

From the July 2026 tuning campaign; each is a todo whose evidence is a
two-draw race on the reference runners. `bench/README.md` has the
current numbers.

- The `i*amax` first-index rescans (worst: isamax at 8% of the stream
  ceiling; icamax inherits it). A fused single-pass candidate already
  lost its f64 race.
- The unmeasured dispatch gap: c32 between 2 MB and 8 MB of A. (The
  real types' gap closed with K2: the packed kernel wins at every
  size; KC and MC were swept 120–256 / 64–256 and moved nothing.)
- The c32 `hemv` grouping lost (~2%, both draws); `chemv` keeps the
  single-column pass. Re-race only if the lane product form changes.
- A lane version of the complex left transposed trmm/trsm kernels
  (`ztrmm_left_t_k` and twins): the scalar one lost its race to the
  copy 0.26–0.76× (both draws); the real-type lane kernels won 1.3–2.8×,
  so the complex lane form is the obvious next candidate.
- What is left against OpenBLAS-wasm after K2 and K1 (B0, two draws):
  sgemm 1.11–1.17× — the 8×4 f32 tile against theirs (same geometry;
  the difference is in the loads or the panel layout, unread); dtrmm
  and dtrsm 1.15–1.23× — the diagonal blocks (the lane routines and
  the sweep's column code) and the per-k-block packing of B, about
  10% of the time at 512² against the gemm tile's rate; dsymm
  1.24–1.27× — still the Level 2 fused pass per column, the natural
  next routine for the packing layer (pack the symmetric A as a full
  panel from both triangles). syr2k on the packing layer as two
  rank-k updates (38–56% in the Level 3 table, the one real Level 3
  row still on the sweep).
- The transposed trmm/trsm forms since K1: the stream candidates
  (D3's winners for the left real forms) are raced against the old
  copy candidate; the copy now runs the packed kernel, so the race
  is stale and the no-transpose packed kernel on a transposed read
  (K2 did it for gemm) is the unraced third candidate.
- The f32 left transposed trmm/trsm ascending-walk cases (trmm upper,
  trsm lower) sit at 0.85–1.0 of the copy on 1024² while the
  descending walks win 1.3×; the asymmetry is unexplained.

## Bugs

- **B1 — the arithmetic-peak probe measures below the kernels.**
  `run_ceiling_flops` steps `acc = acc·m + a` on 8 accumulators, so
  its multiply is on the dependency chain and it measures a latency
  bound; the K2 kernel keeps its multiplies off the chain and reads
  102–111% of the probe in both roofline draws (runs 40232849090,
  40233130801), as do the K1 rows in f32. Fix: a probe whose multiply
  takes independent operands (products from a rotating register set,
  adds on the chain), re-measured against the kernel in one run; the
  test that would have caught it: the roofline script fails any row
  above 100% of the peak. Until then the Level 3 "% of peak" column is
  relative to the gemm kernel, not to the machine.
- Closed on this branch, recorded because the commit that opened it
  (`409158e`) states the wrong premise: the B0 rows and the K1 race
  rows seemed to disagree 1.15–1.3× on the same kernel. They were
  compared across runs on different runners, whose speed differs by
  up to 1.5× (roofline peaks 14.9 and 22.4 GFLOP/s in two draws); in
  one run the two paths agree within 1% (run 40233496997). The direct
  rows stay in the B0 script as the check that would have caught it,
  and no claim in this tree compares absolute times across runs.

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
