// Smoke test for the blas.wasm C-ABI module: loads the raw wasm (no
// bindgen, no imports), drives a routine from each level and each
// number type through the flat-pointer ABI, and checks exact values —
// these are tiny integer-valued problems, so results are exact, not
// toleranced.
//   node check.mjs target/wasm32-unknown-unknown/release/blas_wasm_abi.wasm
import { readFileSync } from 'node:fs';

const path = process.argv[2] ?? 'target/wasm32-unknown-unknown/release/blas_wasm_abi.wasm';
const { instance } = await WebAssembly.instantiate(readFileSync(path), {});
const e = instance.exports;

let failed = 0;
function check(what, got, want) {
	const ok = Object.is(got, want);
	if (!ok) failed++;
	console.log(`${ok ? 'ok  ' : 'FAIL'} ${what}: ${got}${ok ? '' : ` (want ${want})`}`);
}

const f64 = (ptr, n) => new Float64Array(e.memory.buffer, ptr, n);
const f32 = (ptr, n) => new Float32Array(e.memory.buffer, ptr, n);
function alloc64(values) {
	const p = e.wasm_alloc(values.length * 8);
	f64(p, values.length).set(values);
	return p;
}
function alloc32(values) {
	const p = e.wasm_alloc(values.length * 4);
	f32(p, values.length).set(values);
	return p;
}

// export surface: all 116 routines + memory + alloc pair
const fns = Object.keys(e).filter((k) => typeof e[k] === 'function');
check('exported functions', fns.length, 118);

// ---- L1 f64: ddot([1,2,3],[4,5,6]) = 32
{
	const x = alloc64([1, 2, 3]);
	const y = alloc64([4, 5, 6]);
	check('ddot', e.ddot(x, y, 3), 32);
}

// ---- L3 f64: C = A*B, A=[[1,2],[3,4]], B=[[5,6],[7,8]] (column-major)
// A col-major [1,3,2,4], B [5,7,6,8]; C = [[19,22],[43,50]] -> [19,43,22,50]
{
	const a = alloc64([1, 3, 2, 4]);
	const b = alloc64([5, 7, 6, 8]);
	const c = alloc64([0, 0, 0, 0]);
	e.dgemm(1, 2, 2, 2, a, 2, b, 2, 0, c, 2);
	const got = Array.from(f64(c, 4));
	check('dgemm', JSON.stringify(got), JSON.stringify([19, 43, 22, 50]));
}

// ---- L2 f32: sgemv y = A*x, same A, x=[1,1] -> [3,7]
{
	const a = alloc32([1, 3, 2, 4]);
	const x = alloc32([1, 1]);
	const y = alloc32([0, 0]);
	e.sgemv(1, 2, 2, a, 2, x, 0, y);
	check('sgemv', JSON.stringify(Array.from(f32(y, 2))), JSON.stringify([3, 7]));
}

// ---- L1 c64: zdotu((1+2i),(3+4i)) = 3+4i+... single pair: (1+2i)*(3+4i) = -5+10i
{
	const x = alloc64([1, 2]);
	const y = alloc64([3, 4]);
	const out = e.wasm_alloc(16);
	e.zdotu(x, y, 1, out);
	check('zdotu', JSON.stringify(Array.from(f64(out, 2))), JSON.stringify([-5, 10]));
	// zdotc conjugates x: conj(1+2i)*(3+4i) = (1-2i)(3+4i) = 11-2i
	e.zdotc(x, y, 1, out);
	check('zdotc', JSON.stringify(Array.from(f64(out, 2))), JSON.stringify([11, -2]));
}

// ---- L3 c32: cgemm 1x1x1: (1+1i)*(2+3i) = -1+5i
{
	const a = alloc32([1, 1]);
	const b = alloc32([2, 3]);
	const c = alloc32([0, 0]);
	e.cgemm(1, 0, 1, 1, 1, a, 1, b, 1, 0, 0, c, 1);
	check('cgemm', JSON.stringify(Array.from(f32(c, 2))), JSON.stringify([-1, 5]));
}

// ---- L1 f64: idamax over [1,-9,3] = index 1
{
	const x = alloc64([1, -9, 3]);
	check('idamax', e.idamax(x, 3), 1);
}

// ---- L2 f64: dtrsv solve Lx=b, L=[[2,0],[1,1]] col-major [2,1,0,1], b=[4,5] -> x=[2,3]
{
	const a = alloc64([2, 1, 0, 1]);
	const x = alloc64([4, 5]);
	e.dtrsv(2, a, 2, 0, 0, x);
	check('dtrsv', JSON.stringify(Array.from(f64(x, 2))), JSON.stringify([2, 3]));
}

console.log(failed ? `\n${failed} FAILED` : '\nall smoke checks passed');
process.exit(failed ? 1 : 0);
