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
same-run register-resident arithmetic peak (GFLOP/s); the success
metric is distance from the ceiling (scipy/faer are market
comparisons only). Every figure is the range across **two
independent runner draws**. The Level 3 tables moved to
`src/L3/README.md` and its family READMEs, which own those numbers;
the Level 1 and 2 tables below move the same way when their READMEs
are written.

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
both draws), so chemv
keeps the single-column pass; its 12% marks the c32 hemv gap honestly:
at two complexes per register the extra blend work costs more than
the saved traffic. The remaining recorded levers here are the i*amax
rescans.

### Level 3

The Level 3 scoreboard (n = 512 against the same-run peak probe, two
draws at the head that routes K2 and K1) and the B0 table against
OpenBLAS-wasm and BLIS-wasm live in `src/L3/README.md`, the level's
chapter; each family's race tables (K2 for gemm, K1 for syrk, trmm and
trsm, the transposed-form races of decision D3) live in that family's
README under `src/L3/`. One owner per number: the tables are there and
not repeated here.

### B0 — OpenBLAS and BLIS compiled to WebAssembly, same engine, same machine

The rows decision D11 asked for: OpenBLAS and BLIS built to wasm and
run in the same Node as this module, on the same runner and the same
shapes, single-threaded, no relaxed SIMD anywhere (`bench/b0/build-wasm.sh`,
`bench/b0/b0.c`, `bench/b0-rows.mjs`, the `b0` suite of `bench.yml`).
OpenBLAS is its own `WASM128_GENERIC` target (hand-written simd128
gemm kernels; develop at `5bab605`); BLIS is its generic C
configuration autovectorized by clang under `-msimd128` (master at
`a8037cf`). Each library runs 1 warm call then 9 (5 at 1024³) timed
calls per row, min taken; the three libraries are interleaved over two
rounds; the table takes the min over the rounds and the range across
two runs. The script also prints dsyrk, dtrmm and dtrsm through the
race's direct path (`blas.wasm-direct-lda512` and `-lda2048` rows: the
kernel without the wrapper, A at the race's leading dimension) as a
standing cross-check of the harness: in run 40233496997 the two paths
agreed within 1%. Absolute times are not comparable across runs: the
runner pool's speed varies by up to 1.5× (the two roofline draws
measured peaks of 14.9 and 22.4 GFLOP/s), which is why every verdict
is a same-run ratio.

The table and the verdict are in `src/L3/README.md`; the per-family
reading in each family's README. XNNPACK is not built. The earlier
native rows (OpenBLAS and BLIS as Ubuntu packages, AVX2/AVX-512 with
fused multiply-add, runs 40213117656 / 40213120395) measured a
different question and are not kept.

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
