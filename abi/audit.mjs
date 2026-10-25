// Binary audit for the shipped wasm artifacts (DESIGN.md, constraint 3
// and the "Dependency-free" claim). No dependencies: a small reader
// over the module's own sections.
//
//   node audit.mjs <module.wasm> [--imports N] [--exports N]
//   node audit.mjs --self-test
//
// The audit FAILS (exit 1) when
//   - the module does not validate in this engine;
//   - the import count or the export count differs from the expected
//     value given on the command line;
//   - any function body contains a relaxed-SIMD instruction (the 0xFD
//     prefix with sub-opcode >= 0x100: relaxed_madd/nmadd, relaxed
//     swizzle, relaxed truncations, relaxed min/max, relaxed q15mulr,
//     relaxed dot products, relaxed laneselect);
//   - any function body cannot be decoded to its exact end, or ends on
//     anything but `end`. A top-level opcode this reader does not know
//     is a failure, never a skip: the audit fails closed. Unassigned
//     sub-opcodes inside the SIMD (below the relaxed family) and
//     atomics ranges are not enumerated here; the engine's validation,
//     which runs first, rejects them.
//
// --self-test assembles two fixtures in memory, one with
// f32x4.relaxed_madd and one with f32x4.add, and checks that the scan
// rejects the first and accepts the second. The gate runs the self-test
// before auditing the real artifact, so a regression in the reader
// itself is caught by the same step.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';

// ---------- LEB128 and byte reading ----------
class Reader {
	constructor(bytes, pos = 0, end = bytes.length) { this.b = bytes; this.p = pos; this.end = end; }
	get eof() { return this.p >= this.end; }
	u8() { if (this.p >= this.end) throw new Error(`read past end at ${this.p}`); return this.b[this.p++]; }
	u32() {
		let r = 0, s = 0, byte;
		do { byte = this.u8(); r |= (byte & 0x7f) << s; s += 7; if (s > 35) throw new Error(`u32 LEB too long at ${this.p}`); } while (byte & 0x80);
		return r >>> 0;
	}
	s33() { // block type: s33 LEB — only the length matters here
		let s = 0, byte;
		do { byte = this.u8(); s += 7; if (s > 42) throw new Error(`s33 LEB too long at ${this.p}`); } while (byte & 0x80);
	}
	s32() { let s = 0, byte; do { byte = this.u8(); s += 7; if (s > 35) throw new Error(`s32 LEB too long at ${this.p}`); } while (byte & 0x80); }
	s64() { let s = 0, byte; do { byte = this.u8(); s += 7; if (s > 70) throw new Error(`s64 LEB too long at ${this.p}`); } while (byte & 0x80); }
	skip(n) { if (this.p + n > this.end) throw new Error(`skip past end at ${this.p}`); this.p += n; }
	name() { const n = this.u32(); const s = new TextDecoder().decode(this.b.subarray(this.p, this.p + n)); this.skip(n); return s; }
	memarg() { const align = this.u32(); if (align & 0x40) this.u32(); /* memidx */ this.u32(); /* offset */ }
}

// ---------- sections ----------
const SECTION = { 0: 'custom', 1: 'type', 2: 'import', 3: 'function', 4: 'table', 5: 'memory', 6: 'global', 7: 'export', 8: 'start', 9: 'element', 10: 'code', 11: 'data', 12: 'datacount', 13: 'tag' };
const KIND = ['func', 'table', 'memory', 'global', 'tag'];

function sections(bytes) {
	const r = new Reader(bytes);
	if (bytes.length < 8 || bytes[0] !== 0x00 || bytes[1] !== 0x61 || bytes[2] !== 0x73 || bytes[3] !== 0x6d) throw new Error('not a wasm binary (bad magic)');
	r.skip(8);
	const out = [];
	while (!r.eof) {
		const id = r.u8(); const size = r.u32(); const start = r.p;
		out.push({ id, name: SECTION[id] ?? `unknown(${id})`, start, end: start + size });
		r.skip(size);
	}
	return out;
}

function imports(bytes, sec) {
	if (!sec) return [];
	const r = new Reader(bytes, sec.start, sec.end), n = r.u32(), out = [];
	for (let i = 0; i < n; i++) {
		const mod = r.name(), name = r.name(), kind = r.u8();
		switch (kind) {
			case 0: r.u32(); break;                       // typeidx
			case 1: r.u8(); limits(r); break;             // reftype, limits
			case 2: limits(r); break;                     // memory limits
			case 3: r.u8(); r.u8(); break;                // valtype, mut
			case 4: r.u8(); r.u32(); break;               // tag
			default: throw new Error(`unknown import kind ${kind}`);
		}
		out.push({ module: mod, name, kind: KIND[kind] });
	}
	return out;
}
function limits(r) { const flags = r.u8(); r.u32(); if (flags & 1) r.u32(); }

function exports(bytes, sec) {
	if (!sec) return [];
	const r = new Reader(bytes, sec.start, sec.end), n = r.u32(), out = [];
	for (let i = 0; i < n; i++) { const name = r.name(), kind = r.u8(); r.u32(); out.push({ name, kind: KIND[kind] ?? `unknown(${kind})` }); }
	return out;
}

// ---------- instruction decoding ----------
// Returns the list of relaxed-SIMD hits; throws on anything it cannot decode.
function scanCode(bytes, sec, firstFuncIndex, relaxed) {
	if (!sec) return { functions: 0, instructions: 0, relaxed };
	const r = new Reader(bytes, sec.start, sec.end);
	const n = r.u32();
	let instructions = 0;
	for (let f = 0; f < n; f++) {
		const size = r.u32(); const bodyEnd = r.p + size;
		const b = new Reader(bytes, r.p, bodyEnd);
		const localGroups = b.u32();
		for (let i = 0; i < localGroups; i++) { b.u32(); b.u8(); }
		let last = -1;
		while (!b.eof) {
			const at = b.p; const op = b.u8(); last = op; instructions++;
			decodeImmediates(b, op, (sub) => relaxed.push({ func: firstFuncIndex + f, offset: at, subop: sub }));
		}
		if (b.p !== bodyEnd) throw new Error(`function ${firstFuncIndex + f}: body decoded to ${b.p}, declared end ${bodyEnd}`);
		if (last !== 0x0b) throw new Error(`function ${firstFuncIndex + f}: body does not end on \`end\` (last opcode 0x${last.toString(16)})`);
		r.p = bodyEnd;
	}
	if (r.p !== sec.end) throw new Error(`code section decoded to ${r.p}, declared end ${sec.end}`);
	return { functions: n, instructions, relaxed };
}

function decodeImmediates(b, op, onRelaxed) {
	if (op === 0x02 || op === 0x03 || op === 0x04) return b.s33();          // block, loop, if
	if (op === 0x0c || op === 0x0d) return b.u32();                         // br, br_if
	if (op === 0x0e) { const n = b.u32(); for (let i = 0; i <= n; i++) b.u32(); return; } // br_table
	if (op === 0x10 || op === 0x12) return b.u32();                         // call, return_call
	if (op === 0x11 || op === 0x13) { b.u32(); b.u32(); return; }           // call_indirect, return_call_indirect
	if (op === 0x1c) { const n = b.u32(); b.skip(n); return; }              // select t*
	if (op >= 0x20 && op <= 0x26) return b.u32();                           // local/global/table get/set/tee
	if (op >= 0x28 && op <= 0x3e) return b.memarg();                        // loads, stores
	if (op === 0x3f || op === 0x40) return b.u32();                         // memory.size, memory.grow
	if (op === 0x41) return b.s32();
	if (op === 0x42) return b.s64();
	if (op === 0x43) return b.skip(4);
	if (op === 0x44) return b.skip(8);
	if (op === 0xd0) return b.u8();                                         // ref.null
	if (op === 0xd2) return b.u32();                                        // ref.func
	if (op === 0xfc) {                                                      // misc prefix
		const sub = b.u32();
		if (sub <= 7) return;                                               // trunc_sat
		switch (sub) {
			case 8: b.u32(); b.u32(); return;    // memory.init
			case 9: b.u32(); return;             // data.drop
			case 10: b.u32(); b.u32(); return;   // memory.copy
			case 11: b.u32(); return;            // memory.fill
			case 12: b.u32(); b.u32(); return;   // table.init
			case 13: b.u32(); return;            // elem.drop
			case 14: b.u32(); b.u32(); return;   // table.copy
			case 15: case 16: case 17: b.u32(); return; // table.grow/size/fill
			default: throw new Error(`unknown 0xFC sub-opcode ${sub}`);
		}
	}
	if (op === 0xfd) {                                                      // SIMD prefix
		const sub = b.u32();
		if (sub >= 0x100) { onRelaxed(sub); return; } // relaxed family: no immediates
		if (sub <= 11) return b.memarg();                                   // v128.load*, v128.store
		if (sub === 12 || sub === 13) return b.skip(16);                    // v128.const, i8x16.shuffle
		if (sub >= 21 && sub <= 34) return b.u8();                          // extract/replace lane
		if (sub >= 84 && sub <= 91) { b.memarg(); b.u8(); return; }         // load/store lane
		if (sub === 92 || sub === 93) return b.memarg();                    // load32/64_zero
		if (sub <= 0xff) return;                                            // every other SIMD op: no immediates
		throw new Error(`unknown 0xFD sub-opcode ${sub}`);
	}
	if (op === 0xfe) {                                                      // atomics prefix
		const sub = b.u32();
		if (sub === 0x03) return b.u8();                                    // atomic.fence
		if (sub <= 0x4e) return b.memarg();
		throw new Error(`unknown 0xFE sub-opcode ${sub}`);
	}
	if (op <= 0x01 || op === 0x05 || op === 0x0b || op === 0x0f || op === 0x1a || op === 0x1b || (op >= 0x45 && op <= 0xc4) || op === 0xd1) return;
	// Everything else (exception handling 0x06-0x0a, GC 0xd3-0xd6 and
	// the 0xFB prefix, unassigned bytes) is a failure, not a skip.
	throw new Error(`unknown opcode 0x${op.toString(16)}`);
}

// ---------- the audit ----------
export function audit(bytes, expect = {}) {
	const problems = [];
	if (!WebAssembly.validate(bytes)) problems.push('module does not validate');
	const secs = sections(bytes);
	const by = (name) => secs.find((s) => s.name === name);
	const imp = imports(bytes, by('import'));
	const exp = exports(bytes, by('export'));
	const importedFuncs = imp.filter((i) => i.kind === 'func').length;
	const hits = []; // kept even if decoding fails part-way, so a relaxed op before the failure is still reported
	let code;
	try { code = scanCode(bytes, by('code'), importedFuncs, hits); }
	catch (e) { problems.push(`cannot decode code section: ${e.message}`); code = { functions: 0, instructions: 0, relaxed: hits }; }
	if (code.relaxed.length) problems.push(`${code.relaxed.length} relaxed-SIMD instruction(s): ` + code.relaxed.slice(0, 5).map((h) => `func ${h.func} @${h.offset} sub 0x${h.subop.toString(16)}`).join(', ') + (code.relaxed.length > 5 ? ', …' : ''));
	if (expect.imports != null && imp.length !== expect.imports) problems.push(`imports: ${imp.length}, expected ${expect.imports}`);
	if (expect.exports != null && exp.length !== expect.exports) problems.push(`exports: ${exp.length}, expected ${expect.exports}`);
	const kinds = {};
	for (const e of exp) kinds[e.kind] = (kinds[e.kind] ?? 0) + 1;
	return { ok: problems.length === 0, problems, imports: imp, exports: exp, exportKinds: kinds, code, sections: secs.map((s) => `${s.name}:${s.end - s.start}`) };
}

// ---------- fixtures for the self-test ----------
function fixture(bodyOps) {
	// (module (func (param v128 v128 v128) (result v128) <bodyOps>))
	const body = [0x00, ...bodyOps, 0x0b]; // no locals, ops, end
	const codeSec = [0x01, body.length, ...body];
	return new Uint8Array([
		0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
		0x01, 0x08, 0x01, 0x60, 0x03, 0x7b, 0x7b, 0x7b, 0x01, 0x7b, // type: (v128,v128,v128)->v128
		0x03, 0x02, 0x01, 0x00,                                     // function: one, type 0
		0x0a, codeSec.length, ...codeSec,
	]);
}
export function selfTest() {
	const relaxed = fixture([0x20, 0x00, 0x20, 0x01, 0x20, 0x02, 0xfd, 0x85, 0x02]); // f32x4.relaxed_madd (0x105)
	const plain = fixture([0x20, 0x00, 0x20, 0x01, 0xfd, 0xe4, 0x01]);                // f32x4.add (0xe4)
	const a = audit(relaxed), b = audit(plain);
	const results = [];
	results.push(['relaxed fixture validates in this engine (so the scan, not the engine, is what rejects it)', WebAssembly.validate(relaxed) === true]);
	results.push(['relaxed fixture is REJECTED by the scan', !a.ok && a.code.relaxed.length === 1 && a.code.relaxed[0].subop === 0x105]);
	results.push(['plain simd fixture is accepted by the scan', b.ok && b.code.instructions === 4]);
	return results;
}

// ---------- CLI ----------
if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
	const args = process.argv.slice(2);
	if (args.includes('--self-test')) {
		let failed = 0;
		for (const [what, ok] of selfTest()) { console.log(`${ok ? 'ok  ' : 'FAIL'} ${what}`); if (!ok) failed++; }
		console.log(failed ? `\naudit self-test: ${failed} FAILED` : '\naudit self-test passed');
		process.exit(failed ? 1 : 0);
	}
	const usage = () => { console.error('usage: node audit.mjs <module.wasm> [--imports N] [--exports N] | --self-test'); process.exit(2); };
	let file; const expect = {};
	for (let i = 0; i < args.length; i++) {
		const a = args[i];
		if (a === '--imports' || a === '--exports') {
			const v = args[++i];
			if (v === undefined || !/^\d+$/.test(v)) usage();
			expect[a.slice(2)] = Number(v);
		} else if (a.startsWith('--')) usage();
		else if (file === undefined) file = a;
		else usage();
	}
	if (file === undefined) usage();
	const res = audit(readFileSync(file), expect);
	console.log(`sections: ${res.sections.join(' ')}`);
	console.log(`imports: ${res.imports.length}${res.imports.length ? ' (' + res.imports.map((i) => `${i.module}.${i.name}:${i.kind}`).join(', ') + ')' : ''}`);
	console.log(`exports: ${res.exports.length} (${Object.entries(res.exportKinds).map(([k, v]) => `${v} ${k}`).join(', ')})`);
	console.log(`code: ${res.code.functions} functions, ${res.code.instructions} instructions decoded, ${res.code.relaxed.length} relaxed-SIMD`);
	for (const p of res.problems) console.log(`FAIL ${p}`);
	console.log(res.ok ? '\naudit passed' : `\naudit FAILED (${res.problems.length})`);
	process.exit(res.ok ? 0 : 1);
}
