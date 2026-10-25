// Transposed trmm/trsm race (decision D3): `copy` (op(A) transposed into
// the workspace, then the existing kernel) vs `stream` (no copy: a
// transposed accessor on the right, column-reduction kernels on the
// left), all four number types, both routines, both sides, both
// triangles, square and skinny shapes. Interleaved timing (copy, stream,
// copy, ...), min-of-iters reported. Before timing, each case asserts
// the two folds are EXACTLY equal — the candidates are bit-identical by
// construction, so any drift is a bug, not noise.
//
// Build + run:
//   cargo build --release --target wasm32-unknown-unknown --lib
//   node tri-t-ab.mjs target/wasm32-unknown-unknown/release/blas_bench.wasm
import { readFileSync } from 'node:fs';

const wasmPath = process.argv[2];
if (!wasmPath) {
	console.error('usage: node tri-t-ab.mjs <blas-bench-wasm>');
	process.exit(2);
}
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const e = instance.exports;
const N_SETUP = 1024;
e.setup(N_SETUP);

const TYPES = [['f64', 0], ['f32', 1], ['c64', 2], ['c32', 3]];
const ROUTINES = [['trmm', 0], ['trsm', 1]];
// [side, m, n]: square sizes, then the skinny shapes where op(A) is large
// relative to B (left: m=1024,n=16; right: m=16,n=1024) — that is where a
// copy of op(A) costs most relative to the arithmetic
const SHAPES = [
	[0, 64, 64], [1, 64, 64],
	[0, 256, 256], [1, 256, 256],
	[0, 1024, 1024], [1, 1024, 1024],
	[0, 1024, 16], [1, 16, 1024],
	[0, 1024, 128], [1, 128, 1024],
];

function timeOnce(fn) {
	const t0 = performance.now();
	fn();
	return performance.now() - t0;
}

let failed = 0;
for (const [tname, ty] of TYPES) {
	for (const [rname, r] of ROUTINES) {
		for (const upper of [1, 0]) {
			for (const [side, m, n] of SHAPES) {
				// complex 1024³ is minutes-long per candidate; keep the race within the runner budget
				if (ty >= 2 && m === 1024 && n === 1024) continue;
				const cp = () => e.run_tri_t_copy(ty, r, side, upper, m, n);
				const st = () => e.run_tri_t_stream(ty, r, side, upper, m, n);
				const f1 = cp();
				const f2 = st();
				const tag = `[${tname} ${rname} ${side ? 'R' : 'L'} ${upper ? 'U' : 'L'} ${m}x${n}]`;
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
				const na = side ? n : m;
				const flop = m * n * na * (ty >= 2 ? 4 : 1); // multiply-adds over the triangle, ×2 for the pair, /2 for the triangle
				const probe = timeOnce(cp) + timeOnce(st);
				const iters = probe > 2000 ? 3 : probe > 500 ? 5 : probe > 100 ? 9 : 15;
				let tc = Infinity, ts = Infinity;
				for (let k = 0; k < iters; k++) {
					tc = Math.min(tc, timeOnce(cp));
					ts = Math.min(ts, timeOnce(st));
				}
				console.log(
					`${tag} copy=${tc.toFixed(3)}ms stream=${ts.toFixed(3)}ms ` +
					`stream_speedup=${(tc / ts).toFixed(3)}x (${(flop / ts / 1e6).toFixed(2)} vs ${(flop / tc / 1e6).toFixed(2)} GFLOP/s) iters=${iters}`,
				);
			}
		}
	}
}
process.exit(failed ? 1 : 0);
