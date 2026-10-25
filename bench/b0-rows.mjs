// B0 sanity rows (STATUS.md): our wasm module on the same Level 3
// shapes bench/b0/b0.c times through a native CBLAS library on the same
// runner, printed in the same `library,op,n,ms,gflops` form so the two
// logs read side by side. Flop counts match b0.c (gemm 2n³; symm 2n³;
// syrk, trmm, trsm n³; complex gemm ×4). Min over the iterations.
//
// Build + run:
//   cargo build --release --target wasm32-unknown-unknown --lib
//   node b0-rows.mjs target/wasm32-unknown-unknown/release/blas_bench.wasm
import { readFileSync } from 'node:fs';

const wasmPath = process.argv[2];
if (!wasmPath) {
	console.error('usage: node b0-rows.mjs <blas-bench-wasm>');
	process.exit(2);
}
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const e = instance.exports;

function timeOnce(fn) {
	const t0 = performance.now();
	fn();
	return performance.now() - t0;
}
function row(op, n, fn, flops) {
	const iters = n >= 1024 ? 5 : 9;
	fn(); // warm + compile
	let best = Infinity;
	for (let r = 0; r < iters; r++) best = Math.min(best, timeOnce(fn));
	console.log(`blas.wasm,${op},${n},${best.toFixed(3)},${(flops / (best / 1e3) / 1e9).toFixed(2)}`);
}

e.setup(1024);
for (const n of [512, 1024]) {
	const f3 = 2 * n * n * n;
	row('dgemm', n, () => e.run_gemm_rect_incumbent(0, n, n, n), f3);
	row('sgemm', n, () => e.run_gemm_rect_incumbent(1, n, n, n), f3);
	row('zgemm', n, () => e.run_gemm_rect_incumbent(2, n, n, n), 4 * f3);
	row('cgemm', n, () => e.run_gemm_rect_incumbent(3, n, n, n), 4 * f3);
}
// the n = 512 triangular and symmetric rows ride the roofline's layer ops
// (run_l3_layer: 1 symm_left, 2 syrk, 4 trmm_left, 5 trsm_left at the setup size)
e.setup(512);
const f512 = 2 * 512 * 512 * 512;
row('dsymm_left', 512, () => e.run_l3_layer(1), f512);
row('dsyrk', 512, () => e.run_l3_layer(2), f512 / 2);
row('dtrmm_left', 512, () => e.run_l3_layer(4), f512 / 2);
row('dtrsm_left', 512, () => e.run_l3_layer(5), f512 / 2);
