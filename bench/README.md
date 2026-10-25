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
independent runner draws**. The tables themselves live in the level
and family READMEs under `src/`, which own those numbers; this page
keeps the method, the instruments and their defects.

### Levels 1 and 2

The Level 1 and Level 2 scoreboards (n = 2048 against the fastest
same-run memory stream, two draws from the July 2026 campaign record)
live in `src/L1/README.md` and `src/L2/README.md`, the levels'
chapters; each family's rows and the reading of them are in the family
READMEs under those folders.

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
