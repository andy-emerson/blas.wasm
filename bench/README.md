# bench/ — the measured benchmarks

The BLAS layer's measurement harness and its current results.
Self-contained — depends only on `blas-wasm` — so it builds in
seconds and the repository is the complete product: code (`../src`),
correctness (`../tests`), measurement (here). The step-by-step
evidence trail with raw tables and run IDs is the July 2026 campaign
record, which lives in the predecessor repository and not in this
tree (`../STATUS.md`); the tables below quote its tables and
name the GitHub Actions runs they came from.

## The scoreboard

Timing runs in the wasm runtime on the reference CI machines. Score =
**% of the machine's same-run measured ceiling**: Levels 1–2 against
the fastest same-run memory stream (GB/s), Level 3 against the
same-run register-resident arithmetic peak (GFLOP/s) — per the
re-derived success metric (distance from the ceiling; scipy/faer are
market comparisons only). Every figure is the range across **two
independent runner draws**.

### Level 1 (n = 2048, % of fastest same-run stream)

| routine | f64 | f32 | c64 | c32 |
|---|---|---|---|---|
| copy | 52–57% | 45–54% | 62–63% | 52–58% |
| swap | 93–97% | 73–76% | 100% | 89–98% |
| scal | 100% | 100% | 70–75% | 77–78% |
| dscal | | | 79–83% | 93–100% |
| rot | 85–86% | 66–67% | 97–99% | 86–100% |
| axpy | 76–77% | 59–68% | 81–83% | 58–60% |
| dot (complex: dotu / dotc) | 53–57% | 39–43% | 53–57% / 53–57% | 46–54% / 46–54% |
| nrm2 | 47–53% | 46% | 45% | 45–48% |
| asum | 44–50% | 42–45% | 41–42% | 44–46% |
| iamax | 17–21% | 8% | 24–25% | 13–16% |

The reductions (dot/nrm2/asum) read one array; their honest ceiling
is the READ path (the triad), not the fastest read-modify-write
stream — against the triad they sit at 73–100% in both types.
iamax's two-pass shape is priced conservatively (only one pass
counted); the f32 row's 8% is a recorded lever — the scalar
first-index rescan is per-element and dominates (a fused
single-pass candidate already lost its f64 race, campaign record step 9).
The complex delegations earn their keep: swap/rot/the real-α scal
ride the tuned real streams at 86–100% of the ceiling in both
complex types; the genuinely complex streams pay the shuffle work of
the lane product form. icamax inherits isamax's known weakness (the
per-element scalar rescan — recorded lever).

### Level 2 (n = 2048, % of fastest same-run stream)

| routine | f64 | f32 | c64 | c32 |
|---|---|---|---|---|
| gemv | 57–64% | 59–64% | 50–56% | 60–63% |
| gemv_t (complex: also gemv_c) | 43–52% | 66–68% | 39–42% / 39–42% | 39–40% / 38–41% |
| ger (complex: geru / gerc) | 79–90% | 100% | 65–70% / 65–70% | 88–91% / 91–92% |
| symv (complex: hemv) | 53–54% | 48% | 16–21% ¹ | 12% |
| trmv | 60–66% | 54–59% | 47–52% | 60–65% |
| trsv | 59–65% | 53–57% | 47–53% | 54–56% |
| syr (complex: her) | 100% | 87–90% | 60–64% | 85–93% |
| syr2 (complex: her2) | 50–51% | 45–46% | 32–35% | 47–49% |

¹ zhemv ships the 4-column grouped fused pass: ~13% faster in ms
than the single-column shape on both draws (runs
29705606221/29705603966); its % range spans one normal-class and one
slow-class machine draw. The same grouping lost for c32 (~2% slower,
both draws, overruling the dev container's prediction), so chemv
keeps the single-column pass; its 12% marks the c32 hemv gap honestly:
at two complexes per register the extra blend work costs more than
the saved traffic. The remaining recorded levers here are the i*amax
rescans.

### Level 3 (n = 512, % of same-run arithmetic peak)

The f64 peak measured 13.3–15.3 GFLOP/s across draws; the f32 peak
26.6–30.6 (~2× — four lanes vs two). Each complex type scores
against its own precision's real peak (complex arithmetic IS real
arithmetic; a complex multiply-add counts 8 real FLOPs).

| routine | f64 | f32 | c64 | c32 |
|---|---|---|---|---|
| gemm | 55–56% ³ | 56–57% | 88–94% ³ | 86–89% |
| symm_left (complex: hemm_left) | 81–84% | 74–77% | 59–68% ² | 53–62% |
| syrk (complex: herk) | 38–42% | 48–49% | 66–68% | 78–82% |
| syr2k (complex: her2k) | 38–42% | 47% | 67–68% | 76–81% |
| trmm_left | 46–50% | 44–46% | 72–75% | 77–78% |
| trsm_left | 47–50% | 46–47% | 72–75% | 78% |
| trmm_right | 47–48% | 55% | 69–76% | 87–89% |
| trsm_right | 47–48% | 54–55% | 69–76% | 88–89% |

² hemm_left rides zhemv; its range is from runs
29705911344/29705909050.
³ gemm rows, and the whole table, are from runs
29721461249/29721465793 (campaign record step 14). At n=512 the
dispatch routes packed for f64 and for c64 (the packed complex
register tile); f32 and c32 route their pre-packed shapes at this
size because their packed zones start higher; the larger f64 packed
wins are at 1024³ and deep-K, above this table's size. Same-run-
relative numbers move with the ceiling draw (this pair measured f64
peaks of 15.2/17.0 GFLOP/s against 13.3–15.3 on the Level 2 pair), so
rows compare within a pair, not across pairs; absolute ms/call in the
run logs is the stable cross-check.

The complex families sit far closer to the arithmetic ceiling than
the real layers (~66–94% vs 38–57% outside the symm/hemm rows) —
complex arithmetic does 4× the
FLOPs per byte moved, so the same fan-out shapes shift from
memory-limited toward compute-bound. zgemm at 88–94% of peak (packed
tile) and cgemm at 86–89% (col4 fan-out, ~26–30 GFLOP/s — the
fastest absolute rows on the board) are effectively at the ceiling.
The remaining below-family rows are the hemm_lefts, riding their
hemv kernels (see the Level-2 note).

Market comparison (not the metric): the layer's gemm beats faer's
blocked gemm in every type raced, two draws each. dgemm 1.25–1.8× at
every measured size (campaign record step 6, raced with the
predecessor repository's `gemm-tune-ab.mjs` — pre-packed; packed adds 1.24–1.44× on top at 512³+); zgemm
1.49–1.71× at n=256–768 on both draws (n=128 split across draws:
1.70×/0.96× — call it a tie at the smallest size; packed adds
1.08–1.33× on top); cgemm 3.11–3.67× at every size including 128,
unanimous (campaign record step 13, the predecessor repository's
`cplx-gemm-ab.mjs` — conservative against us: the blas rows do the full αAB+βC blend
where faer's row is a plain replace). The packed-vs-incumbent race
itself: `packed-gemm-ab.mjs`, runs 29720778259/29720782633, full
tables in campaign record step 14.

### The transposed-form races (decision D3)

Two candidates per transposed form, both bit-identical to the
definition (the no-transpose call on an explicit transpose), raced on
the runners with two draws each; the netlib wrappers route the winner
and the loser stays exported. Figures are `stream ÷ copy` (trmm/trsm)
or `pack ÷ copy` (gemm) wall time, min-of-iters, the range across the
two draws; `copy` is the operand transposed into the workspace ahead
of the existing kernel. Both candidates are called directly and take
their workspace from the bench's setup state, grown on the warmup
call, so no allocation lands inside a timed call (an earlier pair of
draws, runs 40208127781/40208130046 and 40208724791/40208727304, had
charged the copy a zero-filling allocation and, for gemm, raced the
wrapper after it already routed the pack; they are superseded, and
every routing they chose survived the re-run except c32 gemm's B-only
form).

**trmm/trsm** (`tri-t-ab.mjs`, runs 40212210771 / 40212213162;
`stream` = the no-copy candidate: a transposed accessor on the right,
column-reduction kernels on the left; A at a 1024 leading dimension, B
refilled per call):

| side, shape | f64 trmm | f64 trsm | f32 trmm | f32 trsm | c64 trmm | c64 trsm | c32 trmm | c32 trsm |
|---|---|---|---|---|---|---|---|---|
| left 256² (U / L) | 1.34–1.54 / 1.61–1.93 | 1.88–2.04 / 1.29–1.45 | 0.98–1.01 / 1.14–1.23 | 1.34–1.37 / 0.93–0.97 | 0.60–0.80 / 0.48–0.60 | 0.54–0.63 / 0.45–0.51 | 0.31–0.40 / 0.29–0.34 | 0.28–0.34 / 0.28–0.34 |
| left 1024² (U / L) | 1.38–1.48 / 1.75–1.85 | 1.94–1.97 / 1.40–1.47 | 0.94–0.95 / 1.04–1.18 | 1.33–1.36 / 0.85–0.91 | — | — | — | — |
| left 1024×16 (U / L) | 2.04–2.35 / 2.51–2.75 | 2.69–3.38 / 2.00–2.14 | 1.59–1.68 / 2.06–2.08 | 2.29–2.33 / 1.49–1.56 | 0.90–1.28 / 0.79–1.10 | 0.82–1.10 / 0.88–0.94 | 0.48–0.72 / 0.43–0.69 | 0.46–0.66 / 0.44–0.63 |
| left 1024×128 (U / L) | 1.52–1.56 / 1.92–1.95 | 2.08–2.19 / 1.53–1.54 | 1.01–1.04 / 1.16–1.27 | 1.44–1.45 / 0.95–0.98 | 0.63–0.78 / 0.55–0.68 | 0.53–0.64 / 0.50–0.52 | 0.32–0.41 / 0.28–0.33 | 0.28–0.36 / 0.27–0.31 |
| right 256² (U / L) | 1.03–1.05 / 1.02–1.04 | 1.06–1.07 / 1.03–1.07 | 1.04 / 1.04 | 1.04 / 1.04 | 0.98–1.07 / 1.05–1.06 | 1.04–1.05 / 0.99–1.11 | 1.04–1.07 / 1.05–1.06 | 1.05–1.06 / 1.04–1.07 |
| right 1024² (U / L) | 1.02 / 1.02 | 1.03 / 1.03 | 1.03 / 1.02–1.03 | 1.02–1.03 / 1.03 | — | — | — | — |
| right 16×1024 (U / L) | 1.22–1.49 / 1.21–1.42 | 1.30–1.46 / 1.29–1.47 | 1.30–1.49 / 1.30–1.52 | 1.23–1.52 / 1.29–1.44 | 1.37–1.73 / 1.30–1.64 | 1.42–1.50 / 1.43–1.75 | 1.40–1.66 / 1.35–1.58 | 1.42–1.64 / 1.47–1.60 |
| right 128×1024 (U / L) | 1.11–1.19 / 1.09–1.17 | 1.11–1.22 / 1.10–1.22 | 1.11–1.22 / 1.11–1.21 | 1.10–1.20 / 1.09–1.19 | 1.11–1.14 / 1.10–1.13 | 1.12 / 1.10–1.11 | 1.12–1.22 / 1.12–1.20 | 1.09–1.22 / 1.13–1.22 |

Verdict and routing: `stream` on the right in every type (f64, f32 and
c32 never below 1.02; c64 0.98–0.99 on two 256² rows in one draw each,
≥ 1.03 everywhere else; up to 1.75 where B is skinny); `stream` on the
left for f64 (1.29–3.38, every row) and f32 (23 of 32 rows ≥ 1, the
skinny shapes 1.5–2.3×; the ascending-walk cases — trmm upper, trsm
lower — sit at 0.85–1.0 on 256² and 1024²); the `copy` on the left for
c64 and c32, where the scalar complex reduction kernel loses every
256² and 1024×128 row in both draws (0.45–0.80 c64, 0.27–0.41 c32) and
wins only some c64 1024×16 rows in one draw. The 64² rows (f64
0.67–1.97, f32 0.64–1.57, c64 0.36–1.05, c32 0.24–1.23) are too small
to be verdicts. A lane version of the complex left kernel is a
recorded lever (`STATUS.md`).

**gemm** (`gemm-t-ab.mjs`, runs 40212214964 / 40212217524; `pack` =
`*gemm_packed_op`, the packed shape packing from a transposed read;
`copy` = `*gemm_copy_op`; every shape is above the type's packed
threshold, β = 0, the flags with a transposed operand — T stands for C
on the complex types):

| m×k×n | f64 TN / NT / TT | f32 TN / NT / TT | c64 CN / NC / CC | c32 CN / NC / CC |
|---|---|---|---|---|
| 256³ | — | — | 1.02 / 1.01 / 1.02–1.03 | — |
| 512³ | 1.02–1.10 / 1.01–1.25 / 1.01–1.26 | — | 1.01 / 1.00 / 1.01–1.02 | — |
| 1024³ | 1.01–1.26 / 1.00–1.26 / 1.01–1.27 | 0.90–1.01 / 0.91–1.01 / 0.90–1.01 | — | — |
| 1024×1024×256 | — | — | — | 0.94 / 0.90–0.96 / 0.93–1.00 |
| 1024×1024×8 (c64: 512×512×8; f32: 2048×1024×8) | 1.35–1.89 / 1.06–1.24 / 1.30–2.16 | 1.26–1.72 / 0.95–1.05 / 1.28–1.71 | 1.43–1.47 / 1.05–1.13 / 1.38–1.57 | 1.38–1.42 / 0.97–0.99 / 1.37–1.54 |
| 1024×1024×64 (c64: 512×512×64) | 1.08–1.34 / 1.02–1.25 / 1.08–1.39 | 0.96–1.13 / 0.91–1.01 / 0.96–1.13 | 1.07–1.08 / 1.01–1.03 / 1.09–1.10 | 0.98–1.04 / 0.91–0.96 / 1.00–1.08 |
| 512×2048×16 (f32, c32: 1024×2048×16; c64: 256×1024×16) | 1.29–1.65 / 1.02–1.24 / 1.31–1.75 | 1.14–1.47 / 0.90–1.03 / 1.14–1.48 | 1.12–1.22 / 1.03–1.07 / 1.19–1.31 | 1.32–1.43 / 0.92–0.97 / 1.31–1.49 |

Verdict and routing: f64 and c64 route `pack` for any transposed
operand (f64 1.00–2.16, never a loss; c64 1.00–1.57, the one 0.997 a
tie). f32 routes `pack` as a tie on the square shapes — draw 1 had it
at 0.90–0.96 on five rows, draw 2 at ≥ 1.01 on every row, the two
draws having landed on runners of different speed — and a win of
1.1–1.7× where A is transposed and n is small. c32 routes `pack` only
when A is transposed (1.3–1.5× at n = 8 and 16, 0.94 at n = 256 in
both draws) and the `copy` for a B-only transposed operand, which lost
every c32 row in both draws (0.90–0.99); below the threshold every
type copies ahead of the register tile.

## Running a roofline (per level, per type)

```sh
cd bench
cargo build --release --target wasm32-unknown-unknown --lib
cargo run --release --bin native l3-bits > /tmp/bits.txt
node l3-roofline.mjs target/wasm32-unknown-unknown/release/blas_bench.wasm /tmp/bits.txt
# f32: use l3-bits-f32 and add --f32 to the script
# c64: use l3-bits-z and add --c64; c32: l3-bits-c and --c32
```

Each script first verifies the level's determinism probes — the wasm
build must reproduce the native bit patterns exactly, and a mismatch
kills the run (expected values: `../tests/README.md`) — then times
every routine of its level and scores it against the same-run
ceiling.

## What's measured how

- **State** (`setup(n)`): plain column-major Vecs, `cs = n` — a, b
  (inputs), sym (SACRIFICIAL: triad destination and L2/L3 mutation
  target), tri (dominant diagonal, keeps solves bounded), rhs, and
  their f32 casts. Same LCG recipe and seeds as the predecessor
  repository's harness, so the probe bit patterns carry over. The c64 twins (az/bz/symz/triz/rhsz) are own
  LCG fills, re/im interleaved draws, same roles; the c32 twins are
  those fills cast to f32.
- **Ceilings**: `run_ceiling_bw` (pure single-pass triad into sym,
  3·8·n² bytes) and `run_ceiling_flops[_f32]` (register-resident
  mul+add chains, 8 accumulators). L1/L2 scores use the fastest
  same-run stream (a single triad under-caps read-modify-write mixes
  — the STREAM rationale); L3 scores use the arithmetic peak.
- **Verdict rules** (the standing method): reference machines only
  for verdicts, two draws per claim, sub-1.3× margins are
  direction-only unless unanimous; the dev container is for
  iteration. Runner draws come from the bench workflow
  (`../.github/workflows/bench.yml`, dispatched twice). Update the
  scoreboard above only from runner draws.
