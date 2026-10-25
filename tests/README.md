# tests/ — correctness suites and the bit-identical results

One test file per BLAS routine, mirroring `../src/` (naming:
`../src/README.md`); `main.rs` per level folder is the Cargo test
target, `common.rs` holds the shared generator and the
higher-precision reference summers. Run with
`cargo test --release` — **228 tests, all green** (L1 68, L2 54, L3
101, the vendored-math cross-check 4, one doctest; CI-enforced by the
gate workflow on every push). The three level suites share one helper
file, `shared.rs` (the generator, the summers, the bit-equality
checks), through `#[path]`; each level's `common.rs` adds its size
lists and level-specific replays.
Benchmarks live in `../bench/README.md`; this page is the
correctness half of the contract.

## The vendored-math cross-check (`tests/math.rs`)

The four in-tree math functions (`src/math.rs`: sqrt/sqrtf via the
hardware instruction, hypot/hypotf ported verbatim from libm) are
compared against libm — a dev-dependency only — **bit-for-bit** on
every run: 200k-value LCG sweeps per function over random bit
patterns (every exponent, subnormals included) plus the edge-case
grids (±0, ±inf, NaN, MAX, smallest subnormals, huge/tiny scaling
thresholds). NaN-producing inputs compare NaN-ness, not payload bits
(IEEE leaves payloads unspecified and hardware sqrt may differ from
software there); everything else is exact.

## What each routine is tested to

The standard is the strongest the routine's math allows (the full
contract rationale is in the crate README). All four types are tested
identically; the f32 and c32 bounds use f64-accumulated references
(products formed in the working precision, as the implementation
forms them).

| routine(s) | standard |
|---|---|
| copy, swap, scal, axpy, rot | **bit-for-bit** vs the scalar definition |
| dot, asum | error-bounded vs compensated/f64 reference |
| nrm2 | error-bounded + over/underflow guard cases at each type's range (1e±300 / 1e±30) |
| iamax | **exact index**, incl. BLAS first-occurrence tie-breaking |
| rotg | defining identities + reference edge cases + subnormal smoke |
| gemv, ger, syr, syr2, trmv, trsv | **bit-for-bit same-order scalar replay** + independent bounds (trsv also residual-verified); the transposed trmv/trsv **replay the dot kernel's register fold bit for bit** (the replays themselves checked against the public dot routines at every tail length) and agree with the no-transpose call on an explicit transpose within an n-scaled bound |
| symv | error-bounded (the fused pass legitimately re-folds its reduction) |
| gemm, syrk, syr2k | **bit-for-bit replay** + bounds; gemm's five shapes (colaxpy / tile / col4 / packed / packed_bcast) cross-checked bit-identical on shapes crossing the KC/MC/MR block boundaries, so routing is invisible; every `trans` form **bit-identical to the no-transpose call on an explicit transpose**, all flag pairs, both gemm candidates, workspace NaN-filled; `*syrk_packed` **bit-identical to the exported sweep** for n 0..23 and across the KC edge, both triangles, the unstored triangle NaN-poisoned and checked untouched |
| symm | left: bounds (rides symv); right: **bit-for-bit replay** |
| trmm, trsm | **bit-for-bit replay** both sides + bounds/residuals — the two elimination-order reorders (trmm_right-upper, trsm_right-lower) are documented in their files and the replays mirror them; the other triangles replay the PLAIN order; the four packed shapes **bit-identical to the exported sweeps** (m or n 1..13 and 16, 17, 31, 64, 65, 130, 257, 260, 300; both triangles; unit and non-unit; two α; A's unstored triangle NaN-poisoned; two NaNs count as equal, the promise's stated limit); the `transa` forms: both candidates and the routed wrapper **bit-identical to the no-transpose call on an explicit transpose**, every side/triangle/diagonal, shapes off the block sizes |
| zaxpy, zscal, zdscal, zcopy, zswap, zdrot | **bit-for-bit** vs the scalar `C64` definition (the SIMD product form is a bit-exact rewrite of it) |
| zdotu, zdotc | error-bounded vs a component-wise compensated reference; plus `zdotc(x,y) == zdotu(conj x, y)` **bitwise** (the conjugation folds into lane signs exactly) |
| dznrm2, dzasum | error-bounded (delegations to the d-streams; nrm2 guard cases at 1e±300) |
| izamax | **exact index** on \|re\|+\|im\|, incl. first-occurrence ties |
| zrotg | defining identities (c²+\|s\|²=1, c·a+s·b=r, −conj(s)a+cb=0) + a=0 reference case + 1e±150/±300 extremes |
| zgemv, zgeru, zgerc, ztrmv, ztrsv | **bit-for-bit same-order replay** (+ bounds; ztrsv residual-verified; `zgerc == zgeru(conj y)` and `zgemv_c == zgemv_t(conj A)` bitwise) |
| zhemv | error-bounded (fused pass re-folds its reduction); stored diagonal imag poisoned to prove it is ignored |
| zher, zher2 | **bit-for-bit replay** + diagonal ends exactly real (+0.0 imag, the Hermitian invariant) |
| zgemm | **bit-for-bit replay** + bounds; col4 AND packed (the complex register tile) vs colaxpy cross-checked bit-identical, plus the dispatch-zone replay |
| zhemm | left: bounds (rides zhemv); right: **bit-for-bit replay** through the conjugating triangle lookup |
| zherk, zher2k | **bit-for-bit replay** incl. the real-β component scale and the exactly-real diagonals; the `ConjTrans` forms **bit-identical to the no-transpose call on an explicit conjugate transpose** |
| ztrmm, ztrsm | **bit-for-bit replay** both sides — same reorder disclosures as the real twins, replays mirror them; residuals for ztrsm |
| zsymm | left: bounds (rides the unconjugated fused pass); right: **bit-for-bit replay** of the plain sweep through the unconjugated triangle lookup |
| zsyrk, zsyr2k | **bit-for-bit replay** of the dsyrk/dsyr2k sequence on complex lanes, unconjugated; `Trans` **bit-identical to the no-transpose call on an explicit transpose**; `ConjTrans` is an argument error, as in the reference |
| zrot | **bit-for-bit** vs the scalar definition x ← c·x + s·y, y ← c·y − conj(s)·x, and equal to `zdrot` bitwise when s is real |
| the c-routines (c32) | the z-suite cloned one-for-one at the two-complexes-per-register lane geometry: same replays bit-for-bit, bounds vs f64-accumulated references, crotg identities in f32 at 1e±15/±30 extremes, the same conjugation cross-checks and Hermitian-diagonal invariants |
| `semantics.rs` (every level) | the reference's rules about what is not read: `alpha == 0` (and `k == 0`) leaves the operands unread and scales or zeroes the output, `beta == 0` never reads the output, herk/her2k return untouched for `alpha == 0` with `beta == 1`, iamax's NaN rule — the operands that must not be read are NaN-poisoned, so a read would show |
| `errors.rs` (Levels 2 and 3) | every checkable precondition returns `Error` with the routine and the argument position, nothing written; the `*_work` queries asserted against the routed candidate; the dot-fold replays checked against the public dot routines |

## Cross-target bit-identity — the results

The standing guarantee: our code produces **identical bits on native
x86-64 and wasm**, by construction (`../src/lanes.rs` emulates the
SIMD lane structure elementwise off-wasm, and every reduction folds
its lanes in a fixed order). It is verified by 126 determinism probes
(30 f64 + 30 f32 + 33 c64 + 33 c32) — fixed-LCG inputs at odd sizes
so every tail path runs, folded to one f64 — checked on every
roofline run and on every reference-runner draw to date, **all green
on every check**. The guarantee is for finite results: a NaN's sign
and payload bits are outside it (WebAssembly leaves NaN propagation
to the engine, and the review of the transposed trmm/trsm forms saw an
overflowed f32 solve differ from its definition only in the NaN sign
bit).

The expected patterns (a probe value changes ONLY when an
accumulation order is changed deliberately, like the symv fold — any
other change is a bug, not noise):

| probe | f64 | f32 |
|---|---|---|
| L1 dot | `bff59b5a2e61f7e1` | `bff59b59e0000000` |
| L1 asum | `407fb452fdf31648` | `407fb452c0000000` |
| L1 nrm2 | `4032713910a7520b` | `4032713900000000` |
| L1 iamax | `404e800000000000` | `404e800000000000` |
| L2 gemv | `40e795d727449339` | `40e795d700000000` |
| L2 gemv_t | `40e7935bc10f043d` | `40e7935bc0000000` |
| L2 ger | `40887f34f83468a4` | `40887f3500000000` |
| L2 symv | `40e794809ffc051f` | `40e79480a0000000` |
| L2 trmv | `40f0c94074d9a36b` | `40f0c94080000000` |
| L2 trsv | `40841856de1ef82e` | `4084185700000000` |
| L2 syr | `40880d020a903f10` | `40880d0220000000` |
| L2 syr2 | `408811beb2690464` | `408811bec0000000` |
| L3 gemm | `40f891007480cf34` | `40f89100a0000000` |
| L3 symm_left | `4107979134f4a1b1` | `4107979100000000` |
| L3 syrk | `411de8614adbe916` | `411de86140000000` |
| L3 syr2k | `40f54985b47f97c9` | `40f54985a0000000` |
| L3 trmm_left | `410795b7692150e4` | `410795b7a0000000` |
| L3 trsm_left | `40268695800105e7` | `4026869540000000` |
| L3 trmm_right | `41079785fcc3c6a7` | `4107978600000000` |
| L3 trsm_right | `402684638bea76a3` | `4026846380000000` |
| L3 symm_right | `4107947163c33400` | `4107947140000000` |
| L2 trmv_t | `40f0c3cb40dc742a` | `40f0c3cb40000000` |
| L2 trsv_t | `408418578b718f20` | `40841857a0000000` |
| L3 gemm_tn | `40f895a06cbee903` | `40f895a060000000` |
| L3 gemm_tn_packed | `40d132e4926e7486` | `40e3689be8000000` |
| L3 syrk_t | `411dfb833884c35d` | `411dfb82a0000000` |
| L3 trmm_left_t | `410795926f9ff548` | `4107959220000000` |
| L3 trsm_left_t | `402685d273fe9dd2` | `402685d260000000` |
| L3 trmm_right_t | `410795a27293ea82` | `410795a240000000` |
| L3 trsm_right_t | `4026865e6f582fa6` | `4026865e40000000` |

The c64 probes (own recipes — complex LCG fills, re then im per
element; complex results folded re+im):

| probe | c64 |
|---|---|
| L1 dotu | `402f2dabc8e0c875` |
| L1 dotc | `4042707060b9a365` |
| L1 nrm2 | `4039ce75f2de1fe4` |
| L1 asum | `408f4c94a86377f2` |
| L1 iamax | `4089000000000000` |
| L2 gemv | `40f7e63e951acc6e` |
| L2 gemv_t | `40f7de90e5b397ac` |
| L2 gemv_c | `40f7e72beb9aec3f` |
| L2 geru | `408fdda2c5491a74` |
| L2 gerc | `409018a0216915e3` |
| L2 hemv | `40f7e48bc69067de` |
| L2 trmv | `41008f303154958a` |
| L2 trsv | `408829ab1fd32186` |
| L2 her | `408fc76ecf538996` |
| L2 her2 | `408fc865db34fae8` |
| L3 gemm | `4109aa6e73e67844` |
| L3 hemm_left | `411857ff7cf3aaf2` |
| L3 herk | `412187c69051502d` |
| L3 her2k | `41059e3d2efffd35` |
| L3 trmm_left | `4118462c24e53161` |
| L3 trsm_left | `40372c057aec4588` |
| L3 trmm_right | `411847a9ff1fcb03` |
| L3 trsm_right | `40372a3fb5d5004e` |
| L3 hemm_right | `41185b38c1688ef6` |
| L2 trmv_c | `41008b988d4cb311` |
| L2 trsv_c | `408829abed241b68` |
| L3 gemm_cn | `4109a2648ff55d34` |
| L3 gemm_cn_packed | `40d67c49dc8b338b` |
| L3 herk_c | `412186fe5cdd2d34` |
| L3 trmm_left_c | `41183fd4b48a4615` |
| L3 trsm_left_c | `403731f3ecc1bbda` |
| L3 trmm_right_c | `41184470caedab9f` |
| L3 trsm_right_c | `40372cef16c9ea53` |

And the c32 probes (f32 recipes, folds widened to f64 once at the
end):

| probe | c32 |
|---|---|
| L1 dotu | `c005e2f700000000` |
| L1 dotc | `c036ce4280000000` |
| L1 nrm2 | `4039c70360000000` |
| L1 asum | `408f416520000000` |
| L1 iamax | `4088d00000000000` |
| L2 gemv | `40f6bcff40000000` |
| L2 gemv_t | `40f6beeac0000000` |
| L2 gemv_c | `40f6bd73a0000000` |
| L2 geru | `4090e9f5e0000000` |
| L2 gerc | `40909f92c0000000` |
| L2 hemv | `40f6b3dc80000000` |
| L2 trmv | `40ffae4600000000` |
| L2 trsv | `40883fcb60000000` |
| L2 her | `4090051e00000000` |
| L2 her2 | `4090084d00000000` |
| L3 gemm | `41094ac240000000` |
| L3 hemm_left | `41185ae040000000` |
| L3 herk | `41218a1c80000000` |
| L3 her2k | `41054b69c0000000` |
| L3 trmm_left | `41184e0ba0000000` |
| L3 trsm_left | `403729f220000000` |
| L3 trmm_right | `4118482180000000` |
| L3 trsm_right | `40372f9420000000` |
| L3 hemm_right | `4118613a80000000` |
| L2 trmv_c | `40ffb0d6c0000000` |
| L2 trsv_c | `40883fcb40000000` |
| L3 gemm_cn | `4109482b60000000` |
| L3 gemm_cn_packed | `4101b228d4000000` |
| L3 herk_c | `4121929160000000` |
| L3 trmm_left_c | `411848ae00000000` |
| L3 trsm_left_c | `40372e5da0000000` |
| L3 trmm_right_c | `4118467f40000000` |
| L3 trsm_right_c | `40373195e0000000` |

Regenerate/verify: `cd ../bench && cargo run --release --bin native
l{1,2,3}-bits[-f32|-z|-c]` for the native side; the roofline scripts
compare the wasm build against a bits file and abort on any mismatch.
These values are checked on every gate run and on every roofline
run; they change only when an accumulation order is changed
deliberately.

Known limits of the probes: the L2/L3 probe folds
are dominated by their solve-safe diagonals (~5e4), so they are weak
detectors of reduction-order changes below one ULP of the total
(campaign record step 8; `../STATUS.md`, "Not in this
tree") — the per-routine replay tests above are the strong
order locks; the probes' job is cross-target identity, which they
gate hard.
