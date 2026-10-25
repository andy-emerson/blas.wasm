// Transposed gemm race (decision D3 applied to gemm): `copy`
// (`*gemm_copy_op`: the transposed operand transposed into the
// workspace, then the size-dispatched shape) vs `pack`
// (`*gemm_packed_op`: the packed shape packing straight from a
// transposed read), both called directly so the race outlives the
// wrapper's routing, all four number types, the flag pairs with a
// transposed operand, square and skinny shapes above the packed
// threshold (the only zone where the two differ: below it the wrapper
// takes the register tile after the copy either way). Interleaved
// timing (copy, pack, copy, ...), min-of-iters reported. Before timing,
// each case asserts the two folds are EXACTLY equal — the candidates are
// bit-identical by construction, so any drift is a bug, not noise.
//
// Build + run:
//   cargo build --release --target wasm32-unknown-unknown --lib
//   node gemm-t-ab.mjs target/wasm32-unknown-unknown/release/blas_bench.wasm
import { readFileSync } from 'node:fs';

const wasmPath = process.argv[2];
if (!wasmPath) {
	console.error('usage: node gemm-t-ab.mjs <blas-bench-wasm>');
	process.exit(2);
}
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const e = instance.exports;
const N_SETUP = 2048;
e.setup(N_SETUP);

const TYPES = [['f64', 0], ['f32', 1], ['c64', 2], ['c32', 3]];
// [transa, transb] with the complex conjugate form standing in for T on c64/c32
const FLAGS = [[1, 0], [0, 1], [1, 1]];
// [m, k, n]: square at and above the packed thresholds, then skinny n
// (A large relative to the arithmetic: where a copy of A costs most).
// Every shape routes to the packed shape in the wrapper, so the race is
// between the transposition strategies, not between shapes.
const SHAPES = {
	0: [[512, 512, 512], [1024, 1024, 1024], [1024, 1024, 8], [1024, 1024, 64], [512, 2048, 16]],
	1: [[1024, 1024, 1024], [2048, 1024, 8], [1024, 1024, 64], [1024, 2048, 16]],
	2: [[256, 256, 256], [512, 512, 512], [512, 512, 8], [512, 512, 64], [256, 1024, 16]],
	3: [[1024, 1024, 256], [1024, 1024, 8], [1024, 1024, 64], [1024, 2048, 16]],
};

function timeOnce(fn) {
	const t0 = performance.now();
	fn();
	return performance.now() - t0;
}

let failed = 0;
for (const [tname, ty] of TYPES) {
	for (const [ta0, tb0] of FLAGS) {
		const ta = ty >= 2 && ta0 ? 2 : ta0;
		const tb = ty >= 2 && tb0 ? 2 : tb0;
		for (const [m, k, n] of SHAPES[ty]) {
			const cp = () => e.run_gemm_t_copy(ty, ta, tb, m, k, n);
			const pk = () => e.run_gemm_t_pack(ty, ta, tb, m, k, n);
			const tag = `[${tname} ${'NTC'[ta]}${'NTC'[tb]} ${m}x${k}x${n}]`;
			const f1 = cp();
			const f2 = pk();
			if (!Number.isFinite(f1 + f2)) {
				console.error(`${tag} NON-FINITE fold — investigate`);
				failed++;
				continue;
			}
			if (!Object.is(f1, f2)) {
				console.error(`${tag} fold MISMATCH ${f1} != ${f2}`);
				failed++;
				continue;
			}
			const flop = 2 * m * k * n * (ty >= 2 ? 4 : 1);
			const probe = timeOnce(cp) + timeOnce(pk);
			const iters = probe > 2000 ? 3 : probe > 500 ? 5 : probe > 100 ? 9 : 15;
			let tc = Infinity, tp = Infinity;
			for (let r = 0; r < iters; r++) {
				tc = Math.min(tc, timeOnce(cp));
				tp = Math.min(tp, timeOnce(pk));
			}
			console.log(
				`${tag} copy=${tc.toFixed(2)}ms pack=${tp.toFixed(2)}ms ` +
				`pack_speedup=${(tc / tp).toFixed(3)}x (${(flop / tp / 1e6).toFixed(2)} vs ${(flop / tc / 1e6).toFixed(2)} GFLOP/s) iters=${iters}`,
			);
		}
	}
}
process.exit(failed ? 1 : 0);
