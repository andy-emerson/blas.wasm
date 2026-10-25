// K1 race: the shipping syrk sweep (`*syrk`, the truncated column-axpy
// fan-out) against the packed candidate (`*syrk_packed`: the gemm tile
// with the B panel read from A's rows, diagonal tiles through scratch),
// f64 and f32, both triangles, both called directly with the workspace
// from the bench's setup state. Interleaved timing, min-of-iters; each
// (type, shape) also asserts the folds are EXACTLY equal.
//
//   cargo build --release --target wasm32-unknown-unknown --lib
//   node k1-ab.mjs target/wasm32-unknown-unknown/release/blas_bench.wasm
import { readFileSync } from 'node:fs';

const wasmPath = process.argv[2];
if (!wasmPath) { console.error('usage: node k1-ab.mjs <blas-bench-wasm>'); process.exit(2); }
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const e = instance.exports;
const N_SETUP = 2048;
e.setup(N_SETUP);

const TYPES = [['f64', 0], ['f32', 1]];
const SHAPES = [[256, 256], [512, 512], [1024, 1024], [512, 2048], [2048, 128]];
function timeOnce(f) { const t0 = performance.now(); f(); return performance.now() - t0; }

let failed = 0;
for (const [tname, ty] of TYPES) {
	for (const [n, k] of SHAPES) {
		for (const [uname, up] of [['upper', 1], ['lower', 0]]) {
			const inc = () => e.run_syrk_shape(ty, n, k, up, 0);
			const cand = () => e.run_syrk_shape(ty, n, k, up, 1);
			const f1 = inc(), f2 = cand();
			if (!Number.isFinite(f1 + f2)) { console.error(`[${tname} syrk ${uname} ${n}x${k}] NON-FINITE fold`); failed++; continue; }
			const flop = n * n * k;
			const probe = timeOnce(inc) + timeOnce(cand);
			const iters = probe > 2000 ? 3 : probe > 500 ? 5 : probe > 100 ? 9 : 15;
			let ti = Infinity, tc = Infinity;
			for (let r = 0; r < iters; r++) { ti = Math.min(ti, timeOnce(inc)); tc = Math.min(tc, timeOnce(cand)); }
			console.log(`[${tname} syrk ${uname} ${n}x${k}] sweep=${ti.toFixed(2)}ms packed=${tc.toFixed(2)}ms packed_speedup=${(ti / tc).toFixed(3)}x (${(flop / tc / 1e6).toFixed(2)} vs ${(flop / ti / 1e6).toFixed(2)} GFLOP/s) iters=${iters}`);
			e.setup(N_SETUP); const fi = e.run_syrk_shape(ty, n, k, up, 0);
			e.setup(N_SETUP); const fc = e.run_syrk_shape(ty, n, k, up, 1);
			if (!Object.is(fi, fc)) { failed++; console.log(`[${tname} syrk ${uname} ${n}x${k}] fold MISMATCH ${fi} != ${fc}`); }
		}
	}
}
console.log(failed ? `${failed} FAILED` : 'folds identical on every shape');
process.exit(failed ? 1 : 0);
