const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

// The dashboard's security reviews section (security.js) on hand-written entries: no browser, no network.
const dir = process.env.DASHBOARD_DIR || join(__dirname, '..', 'dashboard');
const source = readFileSync(join(dir, 'security.js'), 'utf8');
const EXPORTS = '\n;globalThis.t = { secEntry, secPack, secUnpack, secInScope, secSort, sec, VERDICTS };';

function load() {
  const ctx = { console, document: { addEventListener() {} }, scrub: s => s };
  vm.runInNewContext(source + EXPORTS, ctx);
  return ctx.t;
}

const review = (extra = {}) => ({
  pr: 7782, verdict: 'notes', reviewer: 'someone/agent', date: '2026-11-23', head: '55aa0061',
  scope: 'static, diff only', findings: ['unpinned actions'], human: false, commented: true, automated: false, ...extra,
});

test('a usable entry keeps its fields', () => {
  const r = load().secEntry(review());
  assert.equal(r.pr, 7782);
  assert.equal(r.verdict, 'notes');
  assert.equal(r.head, '55aa0061');
  assert.deepEqual([...r.findings], ['unpinned actions']);
  assert.equal(r.commented, true);
  assert.equal(r.human, false);
});

test('entries that cannot be shown are dropped, odd fields are emptied', () => {
  const t = load();
  assert.equal(t.secEntry(review({ pr: 'abc' })), null);
  assert.equal(t.secEntry(review({ pr: -3 })), null);
  assert.equal(t.secEntry(review({ verdict: 'fine' })), null);
  assert.equal(t.secEntry(null), null);
  const r = t.secEntry(review({ head: 'javascript:x', date: 'yesterday', findings: ['ok', 42, '', { a: 1 }], human: 'yes', reviewer: '' }));
  assert.equal(r.head, '', 'only a hex commit is linked');
  assert.equal(r.date, '');
  assert.deepEqual([...r.findings], ['ok']);
  assert.equal(r.human, false, 'only true counts as true');
  assert.equal(r.reviewer, 'unknown');
});

test('the stored copy comes back the same, and a damaged one is ignored', () => {
  const t = load();
  const all = [
    t.secEntry(review()),
    t.secEntry(review({ pr: 8475, verdict: 'do-not-merge', automated: true, human: true, scope: 'static, diff only' })),
    t.secEntry(review({ pr: 9511, verdict: 'clean', findings: [], scope: 'another scope', commented: false })),
  ];
  const packed = t.secPack(all, 1234);
  assert.equal(packed.scopes.length, 2, 'each scope text once');
  const back = t.secUnpack(JSON.parse(JSON.stringify(packed)));
  assert.equal(back.at, 1234);
  assert.deepEqual(JSON.parse(JSON.stringify(back.all)), JSON.parse(JSON.stringify(all)));
  assert.equal(t.secUnpack(null), null);
  assert.equal(t.secUnpack({ at: 1, scopes: [], rows: 'x' }), null);
  assert.equal(t.secUnpack({ at: 1, scopes: [], rows: [[0, 'clean']] }), null, 'rows that fail the checks are dropped');
});

test('filters and the most-serious order', () => {
  const t = load();
  const a = t.secEntry(review({ pr: 1, verdict: 'clean', findings: [] }));
  const b = t.secEntry(review({ pr: 2, verdict: 'do-not-merge', findings: ['hidden payload'] }));
  const c = t.secEntry(review({ pr: 3, verdict: 'notes', automated: true }));
  assert.deepEqual(t.secSort([a, c, b]).map(r => r.pr), [2, 3, 1]);
  t.sec.f.remarks = true;
  assert.equal(t.secInScope(a, null), false);
  t.sec.f.remarks = false;
  t.sec.f.hideAuto = true;
  assert.equal(t.secInScope(c, null), false);
  t.sec.f.hideAuto = false;
  t.sec.f.q = '#2';
  assert.equal(t.secInScope(b, null), true);
  t.sec.f.q = 'payload';
  assert.equal(t.secInScope(b, null), true);
  assert.equal(t.secInScope(a, null), false);
  t.sec.f.q = '';
  t.sec.f.openOnly = true;
  assert.equal(t.secInScope(a, new Map([[2, {}]])), false, 'closed PRs leave the list');
  assert.equal(t.secInScope(a, null), true, 'until the open PRs are known nothing is hidden');
});
