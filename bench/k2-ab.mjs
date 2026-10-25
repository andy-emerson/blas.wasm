// K2 race: the shipping packed gemm (`*gemm_packed`) against the K2
// candidate (`*gemm_packed_bcast`: α folded into the B panel at pack
// time, B broadcast by lane shuffles in the k loop), f64 and f32, both
// called directly with the workspace from the bench's setup state, so
// no allocation lands inside a timed call; then, at the sizes below the
// packed threshold, the unpacked register tile (`*gemm_tiled`, the
// shipping shape there) against the candidate, to re-race the
// threshold. Interleaved timing, min-of-iters; each (type, shape) also
// asserts the folds are EXACTLY equal, since every shape keeps the
// per-element rounding sequence.
//
//   cargo build --release --target wasm32-unknown-unknown --lib
//   node k2-ab.mjs target/wasm32-unknown-unknown/release/blas_bench.wasm
import { readFileSync } from 'node:fs';

const wasmPath = process.argv[2];
if (!wasmPath) {
	console.error('usage: node k2-ab.mjs <blas-bench-wasm>');
	process.exit(2);
}
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const e = instance.exports;
const N_SETUP = 2048;
e.setup(N_SETUP);

const TYPES = [['f64', 0], ['f32', 1]];
const SHAPES = [[256, 256, 256], [512, 512, 512], [1024, 1024, 1024], [128, 2048, 2048]];

function timeOnce(f) { const t0 = performance.now(); f(); return performance.now() - t0; }

let failed = 0;
for (const [tname, ty] of TYPES) {
	for (const [m, k, n] of SHAPES) {
		const inc = () => e.run_gemm_rect_packed(ty, m, k, n);
		const cand = () => e.run_gemm_rect_packed_bcast(ty, m, k, n);
		const f1 = inc(), f2 = cand();
		if (!Number.isFinite(f1 + f2)) { console.error(`[${tname} ${m}x${k}x${n}] NON-FINITE fold`); failed++; continue; }
		const flop = 2 * m * k * n;
		const probe = timeOnce(inc) + timeOnce(cand);
		const iters = probe > 2000 ? 3 : probe > 500 ? 5 : probe > 100 ? 9 : 15;
		let ti = Infinity, tc = Infinity;
		for (let r = 0; r < iters; r++) { ti = Math.min(ti, timeOnce(inc)); tc = Math.min(tc, timeOnce(cand)); }
		console.log(`[${tname} ${m}x${k}x${n}] packed=${ti.toFixed(2)}ms bcast=${tc.toFixed(2)}ms bcast_speedup=${(ti / tc).toFixed(3)}x (${(flop / tc / 1e6).toFixed(2)} vs ${(flop / ti / 1e6).toFixed(2)} GFLOP/s) iters=${iters}`);
	}
}
// the dispatch threshold: tiled (shipping below it) vs the candidate
const SMALL = [[128, 128, 128], [192, 192, 192], [256, 256, 256], [384, 384, 384], [512, 512, 512], [640, 640, 640]];
console.log('--- below the packed threshold: tiled vs bcast ---');
for (const [tname, ty] of TYPES) {
	for (const [m, k, n] of SMALL) {
		const inc = () => e.run_gemm_rect_tiled(ty, m, k, n);
		const cand = () => e.run_gemm_rect_packed_bcast(ty, m, k, n);
		const f1 = inc(), f2 = cand();
		if (!Number.isFinite(f1 + f2)) { console.error(`[${tname} ${m}x${k}x${n}] NON-FINITE fold`); failed++; continue; }
		const flop = 2 * m * k * n;
		const probe = timeOnce(inc) + timeOnce(cand);
		const iters = probe > 2000 ? 3 : probe > 500 ? 5 : probe > 100 ? 9 : 15;
		let ti = Infinity, tc = Infinity;
		for (let r = 0; r < iters; r++) { ti = Math.min(ti, timeOnce(inc)); tc = Math.min(tc, timeOnce(cand)); }
		console.log(`[${tname} ${m}x${k}x${n}] tiled=${ti.toFixed(3)}ms bcast=${tc.toFixed(3)}ms bcast_speedup=${(ti / tc).toFixed(3)}x (${(flop / tc / 1e6).toFixed(2)} vs ${(flop / ti / 1e6).toFixed(2)} GFLOP/s) iters=${iters}`);
		e.setup(N_SETUP); const fi = e.run_gemm_rect_tiled(ty, m, k, n);
		e.setup(N_SETUP); const fc = e.run_gemm_rect_packed_bcast(ty, m, k, n);
		if (!Object.is(fi, fc)) { failed++; console.log(`[${tname} ${m}x${k}x${n}] tiled vs bcast fold MISMATCH ${fi} != ${fc}`); }
	}
}
console.log('--- bit-identity (fresh state per call) ---');
for (const [tname, ty] of TYPES) {
	for (const [m, k, n] of SHAPES) {
		e.setup(N_SETUP);
		const fi = e.run_gemm_rect_packed(ty, m, k, n);
		e.setup(N_SETUP);
		const fc = e.run_gemm_rect_packed_bcast(ty, m, k, n);
		const ok = Object.is(fi, fc);
		if (!ok) failed++;
		console.log(`[${tname} ${m}x${k}x${n}] fold ${ok ? 'IDENTICAL' : `MISMATCH ${fi} != ${fc}`}`);
	}
}
process.exit(failed ? 1 : 0);
