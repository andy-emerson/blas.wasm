// Verifies every `{{#include path[:anchor]}}` in the book's chapters:
// the file must exist, and a named anchor must be present in it as
// both `ANCHOR: name` and `ANCHOR_END: name`. mdBook itself renders a
// missing anchor as an empty include without failing the build, so
// this check is what makes "a renamed or removed anchor fails the
// build" true. Run from anywhere:
//
//   node book/check-includes.mjs
//
// Exit 1 on any problem. No dependencies.
import { readFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), 'src');

function* chapters(dir) {
	for (const name of readdirSync(dir)) {
		const p = join(dir, name);
		if (statSync(p).isDirectory()) yield* chapters(p);
		else if (name.endsWith('.md')) yield p;
	}
}

const INCLUDE = /\{\{#(?:include|rustdoc_include)\s+([^}\s]+)\s*\}\}/g;
const problems = [];
let includes = 0;

for (const chapter of chapters(root)) {
	const text = readFileSync(chapter, 'utf8');
	for (const m of text.matchAll(INCLUDE)) {
		includes++;
		const [pathPart, ...rest] = m[1].split(':');
		const file = resolve(dirname(chapter), pathPart);
		const where = `${chapter.slice(root.length + 1)}: {{#include ${m[1]}}}`;
		if (!existsSync(file)) { problems.push(`${where}: file not found (${file})`); continue; }
		// `file:12:30` is a line range; `file:name` is an anchor.
		if (rest.length === 1 && !/^\d*$/.test(rest[0])) {
			const anchor = rest[0];
			const src = readFileSync(file, 'utf8');
			const start = new RegExp(`ANCHOR:\\s*${anchor}\\b`).test(src);
			const end = new RegExp(`ANCHOR_END:\\s*${anchor}\\b`).test(src);
			if (!start || !end) problems.push(`${where}: anchor "${anchor}" ${!start && !end ? 'not found' : !start ? 'has no ANCHOR line' : 'has no ANCHOR_END line'} in ${pathPart}`);
		}
	}
}

for (const p of problems) console.log(`FAIL ${p}`);
console.log(problems.length ? `\n${problems.length} include problem(s) in ${includes} include(s)` : `all ${includes} includes resolve (files present, anchors paired)`);
process.exit(problems.length ? 1 : 0);
