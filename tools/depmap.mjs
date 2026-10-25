// The dependency maps in src/**/README.md are generated from the source
// by this script and checked by the gate, so the maps cannot drift from
// the code.
//
//   node tools/depmap.mjs --write   rewrite every generated block
//   node tools/depmap.mjs --check   fail if any block differs from the code
//
// What counts: a call `name(` or `name::<` in a routine file to a free
// function defined at the top level of another file of the crate. Method
// calls (`x.f(`), associated functions (`F64x2::load(`), macros and the
// lane types are not edges: they are plumbing every routine shares.
// Names are resolved through the file's `use` lines and the modules'
// re-exports, so two functions with one name in two files (the per-type
// `packed_route`s) go to the right file. A name this script cannot
// resolve is an error, not a guess.
//
// Every directory under src/ that holds routine files owns a README.md
// with one generated block between the markers
//   <!-- deps:begin -->  and  <!-- deps:end -->
// A family directory's block is the table of its routines and what they
// call; a level directory's block is the graph of the families that
// level reaches, in Mermaid; src/README.md's block is the whole graph.
import { readFileSync, writeFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { join, relative, dirname, basename } from 'node:path';

const ROOT = join(dirname(new URL(import.meta.url).pathname), '..');
const SRC = join(ROOT, 'src');
const mode = process.argv[2];
if (mode !== '--write' && mode !== '--check') {
	console.error('usage: node tools/depmap.mjs --write | --check');
	process.exit(2);
}

// ---- the files ------------------------------------------------------------
function walk(dir, out = []) {
	for (const e of readdirSync(dir)) {
		const p = join(dir, e);
		if (statSync(p).isDirectory()) walk(p, out);
		else if (e.endsWith('.rs')) out.push(p);
	}
	return out;
}
const files = walk(SRC).map((p) => relative(SRC, p)).sort();
const isLevelPath = (f) => /^L[123]\//.test(f);
const isRoutineFile = (f) => isLevelPath(f) && basename(f) !== 'mod.rs';
const stem = (f) => basename(f, '.rs');
const dirOf = (f) => dirname(f); // 'L3' or 'L3/gemm' or '.'
const levelOf = (f) => (isLevelPath(f) ? f.slice(0, 2) : null);

// The family of a routine: its directory when it sits in one, else the
// netlib stem with the type letters removed (the flat layout).
const FAMILY_ALIAS = {
	hemm: 'symm', herk: 'syrk', her2k: 'syr2k', hemv: 'symv', her: 'syr', her2: 'syr2',
	geru: 'ger', gerc: 'ger', dotu: 'dot', dotc: 'dot',
};
function familyOf(f) {
	const d = dirOf(f);
	if (/^L[123]\/.+/.test(d)) return basename(d);
	let s = stem(f);
	if (/^i[dszc]amax$/.test(s)) return 'iamax';
	// the mixed-precision names carry two type letters
	s = /^(dzasum|dznrm2|scasum|scnrm2|zdscal|csscal|zdrot|csrot)$/.test(s) ? s.slice(2) : s.slice(1);
	return FAMILY_ALIAS[s] || s;
}

// ---- strip comments and strings so names inside them are not calls --------
function strip(code) {
	let out = '', i = 0;
	while (i < code.length) {
		const c = code[i], n = code[i + 1];
		if (c === '/' && n === '/') { while (i < code.length && code[i] !== '\n') i++; continue; }
		if (c === '/' && n === '*') { i += 2; while (i < code.length && !(code[i] === '*' && code[i + 1] === '/')) i++; i += 2; continue; }
		if (c === '"') { i++; while (i < code.length && code[i] !== '"') { if (code[i] === '\\') i++; i++; } i++; out += '""'; continue; }
		if (c === "'" && /^'(\\.|[^\\'])'/.test(code.slice(i))) { i += code.slice(i).match(/^'(\\.|[^\\'])'/)[0].length; out += "' '"; continue; }
		out += c; i++;
	}
	return out;
}
const src = {};
for (const f of files) src[f] = strip(readFileSync(join(SRC, f), 'utf8'));

// ---- definitions: free functions at column 0 (methods are indented) -------
const defs = {}; // name -> [file]
for (const f of files) {
	for (const m of src[f].matchAll(/^(?:pub(?:\([^)]*\))?\s+)?(?:unsafe\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)/gm)) {
		(defs[m[1]] ||= []).push(f);
	}
}
for (const k in defs) defs[k] = [...new Set(defs[k])];

// ---- imports and re-exports -----------------------------------------------
// imports[file] = { localName: { path: [segments], name } }
// reexports[modDir] = { name: { path, name } } from `pub use` / `pub(crate) use` in mod.rs
function parseUse(body) {
	// returns [{ path: [...], name, alias }] for one `use ...;` body
	const out = [];
	const rec = (prefix, s) => {
		s = s.trim();
		const brace = s.indexOf('{');
		if (brace === -1) {
			const segs = s.split('::').map((x) => x.trim()).filter(Boolean);
			if (!segs.length) return;
			const last = segs.pop();
			const m = last.match(/^([A-Za-z_][A-Za-z0-9_]*)(?:\s+as\s+([A-Za-z_][A-Za-z0-9_]*))?$/);
			if (!m) return;
			out.push({ path: [...prefix, ...segs], name: m[1], alias: m[2] || m[1] });
			return;
		}
		const head = s.slice(0, brace).split('::').map((x) => x.trim()).filter(Boolean);
		let depth = 0, start = brace + 1, parts = [];
		for (let i = brace + 1; i < s.length; i++) {
			if (s[i] === '{') depth++;
			else if (s[i] === '}') { if (depth === 0) { parts.push(s.slice(start, i)); break; } depth--; }
			else if (s[i] === ',' && depth === 0) { parts.push(s.slice(start, i)); start = i + 1; }
		}
		for (const p of parts) if (p.trim()) rec([...prefix, ...head], p);
	};
	rec([], body);
	return out;
}
const imports = {}, reexports = {};
for (const f of files) {
	imports[f] = {};
	for (const m of src[f].matchAll(/^\s*(pub(?:\([^)]*\))?\s+)?use\s+([^;]+);/gm)) {
		for (const u of parseUse(m[2])) {
			imports[f][u.alias] = u;
			if (m[1] && basename(f) === 'mod.rs') (reexports[dirOf(f)] ||= {})[u.alias] = u;
		}
	}
}

// Resolve a module path (segments) relative to a file to a directory or a
// file of the crate: returns { dir } or { file } or null.
function resolvePath(fromFile, segs) {
	// A routine file is a leaf module: its `super` is the directory that
	// holds it. A mod.rs (or lib.rs) is the directory itself: its `super`
	// is the directory above.
	const leaf = basename(fromFile) !== 'mod.rs' && fromFile !== 'lib.rs';
	const parent = (d) => (d === '.' ? null : d.includes('/') ? dirname(d) : '.');
	let dir = dirOf(fromFile), i = 0;
	if (segs[0] === 'crate') { dir = '.'; i = 1; }
	else if (segs[0] === 'self') { if (leaf) return { file: fromFile, rest: segs.slice(1) }; i = 1; }
	else if (segs[0] === 'super') { dir = leaf ? dir : parent(dir); i = 1; }
	while (segs[i] === 'super') { dir = parent(dir); i++; if (dir === null) return null; }
	for (; i < segs.length; i++) {
		const s = segs[i];
		const asDir = dir === '.' ? s : `${dir}/${s}`;
		if (files.some((f) => f.startsWith(asDir + '/'))) { dir = asDir; continue; }
		const asFile = `${asDir}.rs`;
		if (files.includes(asFile)) return { file: asFile, rest: segs.slice(i + 1) };
		return null;
	}
	return { dir };
}
// A directory's own code is its mod.rs (or, for '.', lib.rs).
const modFile = (dir) => (dir === '.' ? 'lib.rs' : `${dir}/mod.rs`);

// Find where `name` is defined, looking from a module directory: its own
// mod.rs, then its re-exports (chased), then its subtree (unique only).
function findInDir(dir, name, hops = 0) {
	if (hops > 6) return null;
	const mf = modFile(dir);
	if (defs[name]?.includes(mf)) return mf;
	const re = reexports[dir]?.[name];
	if (re) {
		const r = resolvePath(mf, re.path);
		if (r?.file) return defs[re.name]?.includes(r.file) ? r.file : findInDir(dirOf(r.file), re.name, hops + 1) ;
		if (r?.dir !== undefined) return findInDir(r.dir, re.name, hops + 1);
	}
	// glob re-exports (`pub use family::*`) and the subtree
	const cands = (defs[name] || []).filter((f) => dir === '.' || f.startsWith(dir + '/'));
	if (cands.length === 1) return cands[0];
	return null;
}
function resolveCall(file, prefix, name) {
	if (!defs[name]) return null; // not a crate function
	if (defs[name].includes(file) && prefix.length === 0) return 'local';
	if (prefix.length) {
		if (/^[A-Z]/.test(prefix[prefix.length - 1])) return null; // Type::f — associated fn
		const r = resolvePath(file, prefix);
		if (!r) throw new Error(`${file}: cannot resolve path ${prefix.join('::')}::${name}`);
		if (r.file) return defs[name].includes(r.file) ? r.file : null;
		return findInDir(r.dir, name) || throwAmb(file, name);
	}
	const imp = imports[file][name];
	if (imp) {
		const r = resolvePath(file, imp.path);
		if (!r) throw new Error(`${file}: cannot resolve import ${imp.path.join('::')}::${imp.name}`);
		if (r.file) return defs[imp.name].includes(r.file) ? r.file : (findInDir(dirOf(r.file), imp.name) || throwAmb(file, name));
		return findInDir(r.dir, imp.name) || throwAmb(file, name);
	}
	if (defs[name].length === 1) return defs[name][0];
	return throwAmb(file, name);
}
function throwAmb(file, name) {
	throw new Error(`${file}: ambiguous call ${name} — defined in ${defs[name].join(', ')} and not imported`);
}

// ---- the edges -------------------------------------------------------------
const KEYWORDS = new Set(['if', 'while', 'match', 'return', 'loop', 'for', 'let', 'unsafe', 'in', 'as', 'fn', 'mut', 'move', 'else', 'break', 'continue', 'where', 'impl', 'use', 'pub', 'const', 'static', 'type', 'struct', 'enum', 'mod', 'ref', 'true', 'false', 'self', 'super', 'crate', 'Some', 'Ok', 'Err', 'None', 'assert', 'debug_assert']);
const edges = {}; // file -> { calleeFile -> Set(names) }
for (const f of files) {
	if (!isRoutineFile(f)) continue;
	edges[f] = {};
	const code = src[f];
	const re = /((?:[A-Za-z_][A-Za-z0-9_]*::)*)([a-z_][a-z0-9_]*)\s*(?:\(|::<)/g;
	for (const m of code.matchAll(re)) {
		const before = code[m.index - 1];
		if (before === '.' || before === '!') continue; // method or macro
		if (/\bfn\s*$/.test(code.slice(Math.max(0, m.index - 12), m.index))) continue; // a definition
		const name = m[2];
		if (KEYWORDS.has(name)) continue;
		const prefix = m[1] ? m[1].split('::').filter(Boolean) : [];
		if (prefix.length && /^[A-Z]/.test(prefix[prefix.length - 1])) continue;
		const to = resolveCall(f, prefix, name);
		if (!to || to === 'local') continue;
		(edges[f][to] ||= new Set()).add(name);
	}
}

// ---- rendering ---------------------------------------------------------------
function calleeLabel(to) {
	if (isRoutineFile(to)) return stem(to);
	if (basename(to) === 'mod.rs') return `${to.slice(0, 2)} helpers`;
	return stem(to) === 'c64' || stem(to) === 'c32' ? `${stem(to)} view` : stem(to);
}
function familyTable(dir) {
	const rows = files.filter((f) => isRoutineFile(f) && dirOf(f) === dir);
	const lines = ['| routine | calls |', '|---|---|'];
	for (const f of rows) {
		const groups = Object.entries(edges[f])
			.sort(([a], [b]) => calleeOrder(a) - calleeOrder(b) || a.localeCompare(b))
			.map(([to, names]) => `${calleeLabel(to)} (${[...names].sort().map((n) => `\`${n}\``).join(', ')})`);
		lines.push(`| \`${stem(f)}\` | ${groups.length ? groups.join(' · ') : 'nothing in the crate (a leaf)'} |`);
	}
	return lines.join('\n');
}
function calleeOrder(to) {
	if (isRoutineFile(to)) return 0;
	if (basename(to) === 'mod.rs') return 1;
	if (to === 'kernels.rs') return 2;
	return 3;
}
// family-level edges across the whole crate
const famEdges = new Map(); // 'L3/gemm' -> Set('L2/gemv', 'kernels', ...)
for (const f of Object.keys(edges)) {
	const from = `${levelOf(f)}/${familyOf(f)}`;
	const set = famEdges.get(from) || new Set();
	for (const to of Object.keys(edges[f])) {
		if (isRoutineFile(to)) set.add(`${levelOf(to)}/${familyOf(to)}`);
		else if (to === 'kernels.rs') set.add('kernels');
	}
	famEdges.set(from, set);
}
function reach(start) {
	const seen = new Set();
	const go = (n) => { if (seen.has(n)) return; seen.add(n); for (const m of famEdges.get(n) || []) go(m); };
	for (const s of start) go(s);
	return seen;
}
function mermaid(levels) {
	const start = [...famEdges.keys()].filter((k) => levels.includes(k.slice(0, 2)));
	const nodes = reach(start);
	const id = (n) => n.replace('/', '_');
	const lines = ['```mermaid', 'graph LR'];
	for (const L of ['L3', 'L2', 'L1']) {
		const ns = [...nodes].filter((n) => n.startsWith(L + '/')).sort();
		if (!ns.length) continue;
		lines.push(`  subgraph ${L}["Level ${L[1]}"]`);
		for (const n of ns) lines.push(`    ${id(n)}["${n.slice(3)}"]`);
		lines.push('  end');
	}
	if (nodes.has('kernels')) lines.push('  kernels["shared SIMD kernels"]');
	const es = [];
	for (const n of nodes) for (const m of famEdges.get(n) || []) if (nodes.has(m) && n !== m) es.push(`  ${id(n)} --> ${id(m)}`);
	lines.push(...es.sort());
	lines.push('```');
	return lines.join('\n');
}
function familyList(level) {
	const fams = {};
	for (const f of files) if (isRoutineFile(f) && levelOf(f) === level) (fams[familyOf(f)] ||= []).push(stem(f));
	return Object.entries(fams).sort().map(([fam, rs]) => `- **${fam}**: ${rs.map((r) => `\`${r}\``).join(', ')}`).join('\n');
}

const BEGIN = '<!-- deps:begin -->', END = '<!-- deps:end -->';
const NOTE = '_Generated by `tools/depmap.mjs` from the source; the gate fails if this block and the code disagree. Do not edit by hand._';
function block(dir) {
	if (dir === '.') return `${BEGIN}\n${NOTE}\n\n${mermaid(['L1', 'L2', 'L3'])}\n${END}`;
	if (/^L[123]$/.test(dir)) return `${BEGIN}\n${NOTE}\n\nFamilies at this level and what each one's routines call, down to the leaves:\n\n${familyList(dir)}\n\n${mermaid([dir])}\n${END}`;
	return `${BEGIN}\n${NOTE}\n\n${familyTable(dir)}\n${END}`;
}

// Every directory with routine files, every level, and the root own a block.
const dirs = new Set(['.']);
for (const f of files) if (isRoutineFile(f)) { dirs.add(dirOf(f)); dirs.add(levelOf(f)); }

let failed = 0;
for (const dir of [...dirs].sort()) {
	const readme = join(SRC, dir, 'README.md');
	const want = block(dir);
	const have = existsSync(readme) ? readFileSync(readme, 'utf8') : null;
	const a = have?.indexOf(BEGIN) ?? -1, b = have?.indexOf(END) ?? -1;
	if (mode === '--check') {
		if (have === null) { console.error(`${relative(ROOT, readme)}: missing (every routine directory owns a README with a deps block)`); failed++; continue; }
		if (a === -1 || b === -1) { console.error(`${relative(ROOT, readme)}: no deps block`); failed++; continue; }
		const cur = have.slice(a, b + END.length);
		if (cur !== want) { console.error(`${relative(ROOT, readme)}: deps block differs from the code (run node tools/depmap.mjs --write)`); failed++; }
		continue;
	}
	let next;
	if (have === null) next = `# ${dir === '.' ? 'src' : dir}\n\n${want}\n`;
	else if (a === -1 || b === -1) next = have.replace(/\s*$/, '') + `\n\n${want}\n`;
	else next = have.slice(0, a) + want + have.slice(b + END.length);
	if (next !== have) { writeFileSync(readme, next); console.log(`wrote ${relative(ROOT, readme)}`); }
}
if (mode === '--check') {
	if (failed) process.exit(1);
	console.log(`deps blocks match the code in ${dirs.size} README files`);
}
