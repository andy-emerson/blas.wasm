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
both draws), so chemv
keeps the single-column pass; its 12% marks the c32 hemv gap honestly:
at two complexes per register the extra blend work costs more than
the saved traffic. The remaining recorded levers here are the i*amax
rescans.

### Level 3 (n = 512, % of same-run arithmetic peak)

Runs 40232849090 and 40233130801 at the head that routes K2 and K1
(`e4850c1`), two different runners (f64 peak probe 14.9 and 22.4
GFLOP/s, f32 28.8 and 42.2); each complex type scores against its own
precision's real peak (complex arithmetic IS real arithmetic; a
complex multiply-add counts 8 real FLOPs). Rows compare within a run,
so the range is across the two runs' percentages; absolute ms/call is
in the run logs and, for the B0 rows, in the B0 table below.

| routine | f64 | f32 | c64 | c32 |
|---|---|---|---|---|
| gemm | 102–107% | 103–109% | 92–95% | 84–101% |
| symm_left (complex: hemm_left) | 81–98% | 75–95% | 60–79% | 60–76% |
| syrk (complex: herk) | 98–102% | 99–110% | 72–95% | 76–94% |
| syr2k (complex: her2k) | 38–49% | 47–56% | 71–92% | 74–92% |
| trmm_left | 86–92% | 89–103% | 77–94% | 75–92% |
| trsm_left | 78–89% | 82–92% | 77–93% | 75–90% |
| trmm_right | 96–100% | 97–111% | 78–97% | 85–102% |
| trsm_right | 94–96% | 94–104% | 77–97% | 86–101% |

**The percentages above 100% are a defect of the instrument, not a
result.** The peak probe (`run_ceiling_flops`: 8 accumulators, each
stepping `acc = acc·m + a`) has its multiply on the accumulator's
dependency chain, so it measures a latency bound that the K2 kernel,
whose multiplies are off the chain (8 independent adds per step, the
products from loads), exceeds by up to 11%. The probe is recorded as
a bug in `STATUS.md`; until it is fixed the column says how the rows
stand to each other and to the gemm kernel, not to the machine. What
the table does show: the real gemm, syrk, trmm and trsm now run
within 10% of each other (K1 put the triangular routines on the gemm
tile: left-side trsm is the slowest of them, its diagonal solve on
lanes), syr2k is the one real Level 3 row still on the column sweep
(38–56%), and the symm/hemm_left rows ride their Level 2 kernels. The
complex rows are unchanged by K1 and K2 and sit at 72–101%.

Market comparison (not the metric): the layer's gemm beats faer's
blocked gemm in every type raced, two draws each. dgemm 1.25–1.8× at
every measured size (campaign record step 6, raced with the
predecessor repository's `gemm-tune-ab.mjs` — pre-packed; packed adds 1.24–1.44× on top at 512³+); zgemm
1.49–1.71× at n=256–768 on both draws (n=128 split across draws:
1.70×/0.96× — call it a tie at the smallest size; packed adds
1.08–1.33× on top); cgemm 3.11–3.67× at every size including 128,
unanimous (campaign record step 13, the predecessor repository's
`cplx-gemm-ab.mjs` — conservative against us: the blas rows do the full αAB+βC blend
where faer's row is a plain replace). The packed race itself:
`packed-gemm-ab.mjs`, runs 29720778259/29720782633, full tables in
campaign record step 14. The script now calls the 4-column fused
shape (`*gemm_col4`) and the packed shape directly with the workspace
hoisted out of the timed call, so a re-run races the two shapes and
not the routed wrapper against itself; the recorded verdict predates
that change and stands until a re-run.

### The transposed-form races (decision D3)

Two candidates per transposed form, both bit-identical to the
definition (the no-transpose call on an explicit transpose), raced on
the runners with two draws each; the netlib wrappers route the winner
and the loser stays exported. Figures are `stream ÷ copy` (trmm/trsm)
or `pack ÷ copy` (gemm) wall time, min-of-iters, the range across the
two draws; `copy` is the operand transposed into the workspace ahead
of the existing kernel. Both candidates are called directly and take
their workspace from the bench's setup state, grown on the warmup
call, so no allocation lands inside a timed call.

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

### The K2 race — the gemm micro-kernel (decision D11)

OpenBLAS's wasm gemm kernel showed the real types had 2× left in the
micro-kernel on the same simd128 lanes. The candidate
(`*gemm_packed_bcast`) keeps the packed shape's blocking and
per-element rounding sequence and changes the k loop to the same
mechanics: α folded into the B panel at pack time, B's four values
loaded as vectors and broadcast by lane shuffles, no scalar load,
multiply or scalar-to-vector move left in the loop (the emitted wasm:
4 loads, 4 shuffles, 8 multiplies, 8 adds per k step, accumulators in
locals). `k2-ab.mjs` races it against the pre-K2 packed shape and,
below the old threshold, against the register tile, both called
directly with the workspace hoisted; folds asserted identical.
Figures are `old ÷ new` wall time, the range across runs 40226735030
and 40226834978:

| shape | f64 vs packed | f32 vs packed |
|---|---|---|
| 256³ | 1.87–1.90 | 1.77–1.91 |
| 512³ | 1.86–1.89 | 1.78–1.91 |
| 1024³ | 1.82–1.86 | 1.79–1.91 |
| 128×2048×2048 | 1.80–1.83 | 1.73–1.86 |

Against the register tile below the old threshold (dev container,
one draw, the runner race not taken because the margin at every
size is far outside its noise): f64 1.22× at 128³ rising to 2.27× at
512³, f32 1.22–1.55×. Verdict: the broadcast kernel routes at every
size in both real types; the tile's zone is closed; the pre-K2 packed
shape and the tile stay exported. A KC/MC sweep (120–256 / 64–256)
moved nothing and the block sizes stay.

### The K1 race — syrk, trmm and trsm on the packing layer (decision D11)

The candidates run the triangular routines' off-diagonal work through
the K2 tile and keep each sweep's per-element term order exactly
(`src/L3/dsyrk.rs`, `dtrmm.rs`, `dtrsm.rs` and the s twins describe
the mechanics); the sweeps stay exported as `*_sweep` and are the
"sweep" side here, both sides called directly with the workspace
hoisted, folds asserted identical. Figures are `sweep ÷ packed` wall
time, upper / lower, the range across the two runner draws:

| routine | shape (n×k or m×n) | f64 upper | f64 lower | f32 upper | f32 lower |
|---|---|---|---|---|---|
| syrk | 256² | 2.06–2.09 | 1.93–1.97 | 2.10–2.13 | 2.33–2.41 |
| syrk | 512² | 2.05–2.07 | 1.88–1.90 | 2.04 | 2.43–2.47 |
| syrk | 1024² | 2.37–2.43 | 2.02–2.04 | 1.99–2.00 | 2.58–2.62 |
| syrk | 512×2048 | 2.05–2.06 | 1.87–1.90 | 2.06 | 2.47–2.49 |
| syrk | 2048×128 | 2.32 | 1.96–1.97 | 1.99 | 2.59–2.63 |
| trmm left | 256² | 1.84–1.85 | 1.92–1.93 | 1.85–1.86 | 1.90 |
| trmm left | 512² | 1.76–1.77 | 1.92–1.94 | 1.91–1.94 | 2.03–2.05 |
| trmm left | 1024² | 1.84–1.87 | 2.06–2.09 | 1.87–1.91 | 2.08–2.09 |
| trmm left | 512×64 | 1.63–1.67 | 1.82–1.85 | 1.75–1.78 | 1.88–1.90 |
| trmm right | 256² | 1.83–1.86 | 1.85–1.87 | 1.84–1.97 | 1.82–1.97 |
| trmm right | 512² | 1.84–1.86 | 1.86–1.87 | 2.04–2.05 | 2.04–2.06 |
| trmm right | 1024² | 1.95 | 1.99–2.05 | 2.09 | 2.10–2.12 |
| trmm right | 512×64 | 1.36–1.39 | 1.36–1.40 | 1.61–1.62 | 1.61 |
| trsm left | 256² | 1.64–1.68 | 1.70–1.77 | 1.58–1.61 | 1.63–1.67 |
| trsm left | 512² | 1.51–1.52 | 1.66–1.67 | 1.67–1.72 | 1.81 |
| trsm left | 1024² | 1.66–1.68 | 1.86–1.90 | 1.51–1.58 | 1.73–1.76 |
| trsm left | 512×64 | 1.47–1.48 | 1.64–1.68 | 1.56–1.62 | 1.68–1.72 |
| trsm right | 256² | 1.75–1.78 | 1.74–1.76 | 1.87–1.93 | 1.86–1.88 |
| trsm right | 512² | 1.80–1.83 | 1.80–1.83 | 1.97–2.01 | 1.96–2.00 |
| trsm right | 1024² | 1.94 | 1.97–2.00 | 2.08–2.10 | 2.07–2.10 |
| trsm right | 512×64 | 1.48–1.53 | 1.45–1.50 | 1.52–1.56 | 1.52–1.55 |

Verdict: the packed shapes route in both real types on both sides and
both triangles; the sweeps stay exported. The remaining overhead
against the gemm tile's rate is the diagonal routines and the panel
packing (about 10% at 512²); the complex types, whose kernels already
beat OpenBLAS-wasm, keep their sweeps.

### B0 — OpenBLAS and BLIS compiled to WebAssembly, same engine, same machine

The rows D11 asked for (`STATUS.md`): OpenBLAS and BLIS built to
wasm and run in the same Node as this module, on the same runner and
the same shapes, single-threaded, no relaxed SIMD anywhere
(`bench/b0/build-wasm.sh`, `bench/b0/b0.c`, `bench/b0-rows.mjs`, the
`b0` suite of `bench.yml`). OpenBLAS is its own `WASM128_GENERIC`
target (hand-written simd128 gemm kernels; develop at `5bab605`);
BLIS is its generic C configuration autovectorized by clang under
`-msimd128` (master at `a8037cf`). Milliseconds per call, min over
the iterations and over the two interleaved rounds of a run, the
range across the two runs at the head that routes K2 and K1 (runs
40232073684 and 40232628850); the ratio columns are this module's
time over theirs, so above 1 means this module is slower.

| op | blas.wasm | OpenBLAS-wasm | BLIS-wasm | ours ÷ OpenBLAS | ours ÷ BLIS |
|---|---|---|---|---|---|
| dgemm 512³ | 17.5–19.1 | 17.5–18.2 | 31.5–35.0 | 1.00–1.05 | 0.54–0.56 |
| sgemm 512³ | 9.0–9.1 | 7.9–8.3 | 13.4–13.9 | 1.11–1.15 | 0.66–0.67 |
| zgemm 512³ | 79.9–81.1 | 83.8–86.3 | 84.6–91.1 | 0.94–0.95 | 0.89–0.94 |
| cgemm 512³ | 41.5–45.3 | 63.6–69.7 | 132.8–134.0 | 0.65 | 0.31–0.34 |
| dsymm left 512 | 21.9–23.2 | 17.7–18.3 | 36.5–40.7 | 1.24–1.27 | 0.57–0.60 |
| dsyrk 512 | 9.2–9.9 | 9.2–9.4 | 18.9–19.2 | 0.99–1.05 | 0.49–0.51 |
| dtrmm left 512 | 10.4–11.3 | 9.0–9.2 | 14.1–15.5 | 1.16–1.23 | 0.73 |
| dtrsm left 512 | 11.4–12.5 | 9.9–10.2 | 17.0–18.5 | 1.15–1.22 | 0.67–0.68 |
| dgemm 1024³ | 142.3–151.1 | 140.6–144.3 | 291.9–334.7 | 1.01–1.05 | 0.45–0.49 |
| sgemm 1024³ | 72.3–73.1 | 62.0–62.5 | 106.7–110.9 | 1.17 | 0.66–0.68 |
| zgemm 1024³ | 642.8–644.7 | 668.7–686.4 | 667.4–722.0 | 0.94–0.96 | 0.89–0.96 |
| cgemm 1024³ | 314.9–330.9 | 506.3–556.8 | 1060.8–1071.8 | 0.59–0.62 | 0.29–0.31 |

Verdict, both draws: against OpenBLAS's wasm port this module ties
dgemm (1.00–1.05×) and dsyrk (0.99–1.05×), trails sgemm 1.11–1.17×,
dtrmm and dtrsm 1.15–1.23× and dsymm 1.24–1.27×, and leads zgemm
1.05–1.06× and cgemm 1.5–1.7×. Against BLIS's generic build it leads
every row, 1.1× on zgemm to 3.3× on cgemm. Before K2 and K1 (runs
40220022112 / 40220778110, head `6af68df`) the real-type rows read
1.9–2.3× behind OpenBLAS and the dsyrk, dtrmm and dtrsm rows 1.1–1.4×
behind BLIS. The claim "faster than OpenBLAS and BLIS in the browser"
is true today against BLIS, true on the complex types against
OpenBLAS, and false by 1.1–1.27× on sgemm, dsymm, dtrmm and dtrsm
against OpenBLAS; what it means for the plan is decision D11 in
`STATUS.md`. The script also prints dsyrk, dtrmm and dtrsm through the
race's direct path (`blas.wasm-direct-lda512` and `-lda2048` rows:
the kernel without the wrapper, A at the race's leading dimension) as
a standing cross-check of the harness: in run 40233496997 the two
paths agree within 1% (dsyrk 10.35 / 10.39 / 10.32 ms, dtrmm 12.01 /
12.05 / 12.01, dtrsm 12.28 / 12.42 / 12.37), so the rows above are the
kernel's time. Absolute times are not comparable across runs: the
runner pool's speed varies by up to 1.5× (the two roofline draws
measured peaks of 14.9 and 22.4 GFLOP/s), which is why every verdict
here is a same-run ratio. XNNPACK is not built. The earlier
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
