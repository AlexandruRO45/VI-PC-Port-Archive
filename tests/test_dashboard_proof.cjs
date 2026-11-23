const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

// The dashboard's "How sure are we?" parsers (proof.js) on hand-written documents: no browser, no network.
const dir = process.env.DASHBOARD_DIR || join(__dirname, '..', 'dashboard');
const source = readFileSync(join(dir, 'proof.js'), 'utf8');
const EXPORTS = '\n;globalThis.t = { proofCells, proofTables, classifyContract, parseContract, parseImports };';

function load() {
  const ctx = { console, document: { addEventListener() {} } };
  vm.runInNewContext(source + EXPORTS, ctx);
  return ctx.t;
}

const DESIGN = [
  '# Design',
  '',
  '### Translator status and runtime contract',
  '',
  'Some text before the table.',
  '',
  '| Piece | Contract |',
  '|---|---|',
  '| Wave ops | GPU-checked by `run_gpu_tests.py` on two vendors. |',
  '| Packed math | Measured on gfx1030; one item still open. |',
  '| Image loads | Source: LLVM `SIInstrInfo`. |',
  '| Export order | Holds over all 1,234 shaders. |',
  '| Integer divide | CPU-checked by `tests/semantic/test_div.py`. |',
  '| 16-bit compare | Not yet GPU-verified; CPU-checked only. |',
  '| Sampler heap | Bound at slot 3. |',
  '| `a|b` select | Source: Mesa `nir_lower`. |',
  '',
  '### Another table',
  '',
  '| Piece | Contract |',
  '|---|---|',
  '| ignored | GPU-checked |',
].join('\n');

test('cells: a pipe inside backticks stays in its cell', () => {
  const t = load();
  assert.deepEqual([...t.proofCells('| `a|b` | two | three |')], ['`a|b`', 'two', 'three']);
});

test('contract: the first matching rule decides, an open item downgrades a GPU row, only the first contract table counts', () => {
  const t = load();
  const c = t.parseContract(DESIGN);
  assert.equal(c.total, 8);
  assert.deepEqual({ ...c.counts }, { gpu: 1, 'gpu+open': 1, reference: 2, corpus: 1, cpu: 1, not_gpu: 1, unstated: 1 });
  assert.deepEqual([...c.waiting], ['16-bit compare']);
  assert.deepEqual([...c.unstated], ['Sampler heap']);
});

test('contract: "not yet GPU-verified" wins over a CPU check named in the same row', () => {
  const t = load();
  assert.equal(t.classifyContract('Not yet GPU-verified; CPU-checked by tests/semantic.'), 'not_gpu');
  assert.equal(t.classifyContract('Nothing said.'), 'unstated');
});

test('contract: a document without the table is an error, not zero', () => {
  const t = load();
  assert.throws(() => t.parseContract('# Nothing here\n'), /contract/);
});

const IMPORTS = [
  '## Measured result',
  '',
  '| Image | Shim | Host | Module | Unimplemented |',
  '|---|---:|---:|---:|---:|',
  '| `one.dll` | 4 | 56 | 0 | 0 |',
  '| `two.dll` | 2 | 10 | 0 | 3 |',
  '',
  '| Consumer | Import | Binding |',
  '|---|---|---|',
  '| `one.dll` | `x.dll!A` | shim |',
  '| `two.dll` | `x.dll!B` | unimplemented |',
  '| `two.dll` | `x.dll!C` | host |',
  '',
  '## The retail executable (RT-5)',
  '',
  'Measured at `1cc6e3c1`, Windows 11:',
  '',
  '| Image | Base | Shim | Host | Module | Unimplemented |',
  '|---|---|---:|---:|---:|---:|',
  '| `title.exe` | `0x140000000` | 53 | 432 | 72 | **30** |',
  '| `one.dll` | relocated | 5 | 52 | 0 | 0 |',
].join('\n');

test('imports: package DLLs, bindings and the executable row with its commit', () => {
  const t = load();
  const im = t.parseImports(IMPORTS);
  assert.deepEqual({ ...im.images }, { total: 2, clean: 1 });
  assert.deepEqual({ ...im.bindings }, { total: 3, bound: 2 });
  assert.deepEqual({ ...im.exe }, { unimplemented: 30, shim: 53, host: 432, module: 72, at: '1cc6e3c1' });
});

test('imports: a document without the tables leaves each part empty', () => {
  const t = load();
  const im = t.parseImports('# Empty\n');
  assert.equal(im.images, null);
  assert.equal(im.bindings, null);
  assert.equal(im.exe, null);
});
