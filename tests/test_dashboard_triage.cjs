const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

// Exercise the dashboard's triage code (duplicates, merge round, GraphQL sweep) without a browser, a token or a network.
const dir = process.env.DASHBOARD_DIR || join(__dirname, '..', 'dashboard');
const source = readFileSync(join(dir, 'triage.js'), 'utf8');
const EXPORTS = '\n;globalThis.t = { refIds, bodyRefs, itemsOf, itemPrefixes, coordItems, findDuplicates, conflictMap, loadGraph, graphCovers, graph, get bodies() { return bodies; } };';

const CATS = ['SHADER', 'FUZZ', 'RT', 'D3D', 'ASSET', 'BUILD', 'TEST', 'PROC', 'BOOT'].map(key => ({ key }));

/** A fresh copy of triage.js with the page globals it uses; `fetch` answers its GraphQL calls. */
function load({ token = 'ghp_test', fetch = async () => { throw new Error('no network in tests'); } } = {}) {
  const saved = new Map();
  const ctx = {
    console: { warn() {}, error() {}, log() {} },
    REPO: 'owner/repo', BRANCH: 'main', CATS, PIPE_STAGES: [], STAGE_INDEX: {}, $: () => null,
    rate: {}, blockedUntil: {}, apiCalls: 0, tokenRefused: false,
    activeToken: () => token,
    noteLimit() {},
    RateLimitError: class extends Error {},
    scrubDeep: v => v,
    plain: s => String(s ?? '').replace(/\[([^\]]+)\]\([^)]+\)/g, '$1').trim(),
    store: { get: k => saved.get(k) ?? null, set: (k, v) => saved.set(k, v), del: k => saved.delete(k) },
    applied: [],
    fetch,
  };
  ctx.applyGraph = items => ctx.applied.push(items);
  vm.runInNewContext(source + EXPORTS, ctx);
  return ctx;
}

const pr = (n, extra = {}) => ({ n, title: `PR ${n}`, user: `u${n}`, url: `https://example.invalid/${n}`, created: `2026-01-${String(n).padStart(2, '0')}T00:00:00Z`, draft: false, stage: 'green', conflict: false, files: [`f${n}`], deps: [], ...extra });

test('item IDs: known prefixes only, the title first, then the first one in the description', () => {
  const { t } = load();
  assert.deepEqual([...t.refIds('SHADER-12 and TEST-3, SHADER-12 again')], ['SHADER-12', 'TEST-3']);
  const refs = t.bodyRefs('Claimed: [BUILD-7](link) in #1.\n\nDepends on: #10, #11\nDepends on #12\nUTF-8 everywhere');
  assert.deepEqual([...refs.lids], ['BUILD-7']);
  assert.deepEqual([...refs.deps], [10, 11, 12]);
  const prefixes = t.itemPrefixes(null);
  const none = new Map();
  assert.deepEqual([...t.itemsOf({ n: 1, title: 'Fix UTF-8 in SHA-256 sums' }, prefixes, none)], []);
  assert.deepEqual([...t.itemsOf({ n: 1, title: 'TEST-4 and RT-2', lids: ['BUILD-1'] }, prefixes, none)], ['TEST-4', 'RT-2']);
  assert.deepEqual([...t.itemsOf({ n: 1, title: 'no id', lids: ['BUILD-1', 'TEST-9'] }, prefixes, none)], ['BUILD-1']);
});

test('the coordination issue names the item before the PR text does', () => {
  const { t } = load();
  const coord = {
    categories: [{ rows: [{ ID: 'EGEN-5 ⭐', PR: '[#7](x), #8' }] }],
    reviewPrs: { 9: { item: 'TEST-1' } },
  };
  const prefixes = t.itemPrefixes(coord);
  assert.ok(prefixes.has('EGEN'), 'a prefix the coordination issue uses counts as an item prefix');
  const listed = t.coordItems(coord);
  assert.deepEqual([...t.itemsOf({ n: 8, title: 'SHADER-1: something' }, prefixes, listed)], ['EGEN-5']);
  assert.deepEqual([...t.itemsOf({ n: 9, title: 'no id' }, prefixes, listed)], ['TEST-1']);
});

test('duplicates: several authors on one item; one author with several PRs is split work', () => {
  const { t } = load();
  const coord = { categories: [{ rows: [{ ID: 'SHADER-5', PR: '#3' }] }], reviewPrs: {} };
  const prs = [
    pr(1, { title: 'SHADER-5: first try', user: 'a', stage: 'failing' }),
    pr(2, { title: 'SHADER-5: second try', user: 'b', stage: 'ready' }),
    pr(3, { title: 'unrelated title', user: 'c', stage: 'waiting' }),
    pr(4, { title: 'TEST-8 part 1', user: 'd' }),
    pr(5, { title: 'TEST-8 part 2', user: 'd' }),
    pr(6, { title: 'BUILD-2 alone', user: 'e' }),
  ];
  const d = t.findDuplicates(prs, coord);
  assert.equal(d.groups.length, 1);
  assert.equal(d.groups[0].id, 'SHADER-5');
  assert.deepEqual([...d.groups[0].prs.map(p => p.n)], [3, 2, 1], 'the PR in the table first, then the furthest along');
  assert.deepEqual([...d.groups[0].listed], [3]);
  assert.equal(d.prs, 3);
  assert.equal(d.series, 1);
});

test('merge round: green, mergeable, no open dependency, no shared file; checked ones first', () => {
  const { t } = load();
  const prs = [
    pr(1, { files: ['a', 'b'] }),
    pr(2, { files: ['b'], stage: 'ready' }),               // checked: taken before 1, which then shares b
    pr(3, { files: ['c'], conflict: true, stage: 'rebase' }),
    pr(4, { files: ['d'], deps: [3] }),                    // waits on open #3
    pr(5, { files: ['e'], deps: [999] }),                  // its dependency is merged already
    pr(6, { files: ['f'], filesCut: true }),               // more files than the sweep reads
    pr(7, { files: ['g'], stage: 'failing' }),
    pr(8, { files: ['a'], draft: true, stage: 'draft' }),  // drafts don't block a round or count as overlap
    pr(9, { files: ['h'], conflict: null }),               // merge state unknown
  ];
  const m = t.conflictMap(prs);
  assert.deepEqual([...m.round.map(p => p.n)], [2, 5]);
  assert.equal(m.candidates, 3);
  assert.equal(m.waiting, 1);
  assert.equal(m.conflicting, 1);
  assert.deepEqual(JSON.parse(JSON.stringify(m.hot)), [{ path: 'b', n: 2, green: 2 }]);
  assert.equal(m.isolated, 6, 'every non-draft PR except #1 and #2 shares no file');
  assert.equal(t.conflictMap([pr(1, { files: undefined })]), null, 'no files known: no map');
});

/** A fake GraphQL endpoint over a list of PR nodes; counts calls by kind. */
function fakeGraphql(nodes, calls) {
  return async (url, init) => {
    assert.equal(url, 'https://api.github.com/graphql');
    assert.equal(init.headers.Authorization, 'Bearer ghp_test');
    const { query, variables } = JSON.parse(init.body);
    const headers = { get: h => ({ 'x-ratelimit-resource': 'graphql', 'x-ratelimit-remaining': '4990', 'x-ratelimit-limit': '5000', 'x-ratelimit-reset': '0' }[h] ?? null) };
    if (query.includes('pullRequests(')) {
      calls.pages++;
      const start = variables.cursor ? Number(variables.cursor) : 0;
      const page = nodes.slice(start, start + 100);
      const next = start + 100 < nodes.length;
      return { ok: true, status: 200, headers, json: async () => ({ data: { repository: { pullRequests: { totalCount: nodes.length, pageInfo: { hasNextPage: next, endCursor: String(start + 100) }, nodes: page } } } }) };
    }
    calls.bodies++;
    const asked = [...query.matchAll(/pullRequest\(number:(\d+)\)/g)].map(m => Number(m[1]));
    calls.bodyPrs.push(...asked);
    const repository = Object.fromEntries(asked.map(n => {
      const node = nodes.find(x => x.number === n);
      return [`p${n}`, { number: n, updatedAt: node.updatedAt, body: node.body }];
    }));
    return { ok: true, status: 200, headers, json: async () => ({ data: { repository } }) };
  };
}

const node = (number, extra = {}) => ({
  number, title: `PR ${number}`, url: `https://example.invalid/${number}`, isDraft: false, createdAt: '2026-01-01T00:00:00Z', updatedAt: 'u1',
  mergeable: 'MERGEABLE', changedFiles: 1, author: { login: 'a', avatarUrl: '' }, comments: { totalCount: 0 },
  files: { nodes: [{ path: `f${number}` }] }, commits: { nodes: [{ commit: { statusCheckRollup: { state: 'SUCCESS' } } }] },
  body: `TEST-${number}\n\nDepends on: #${number + 1000}`, ...extra,
});

test('the sweep pages through every open PR and reads a body only when its PR changed', async () => {
  const nodes = Array.from({ length: 230 }, (_, i) => node(i + 1));
  nodes[0] = node(1, { mergeable: 'CONFLICTING', commits: { nodes: [{ commit: { statusCheckRollup: { state: 'FAILURE' } } }] } });
  nodes[1] = node(2, { mergeable: 'UNKNOWN', changedFiles: 140, commits: { nodes: [{ commit: { statusCheckRollup: null } }] } });
  nodes[2] = node(3, { author: null, commits: { nodes: [{ commit: { statusCheckRollup: { state: 'PENDING' } } }] } });
  const calls = { pages: 0, bodies: 0, bodyPrs: [] };
  const ctx = load({ fetch: fakeGraphql(nodes, calls) });
  const { t } = ctx;
  assert.equal(await t.loadGraph(Date.now()), 'done');
  assert.equal(calls.pages, 3);
  assert.equal(calls.bodies, 5, '230 bodies in batches of 50');
  assert.equal(ctx.applied.length, 2, 'the PR list is shown before the descriptions, then again with them');
  assert.equal(ctx.applied[0].length, 230);
  assert.deepEqual([...ctx.applied[0][3].lids], [], 'the first showing has no descriptions yet');
  const items = ctx.applied[1];
  assert.equal(items.length, 230);
  const [one, two, three, four] = items;
  assert.equal(one.conflict, true);
  assert.equal(one.ci, 'bad');
  assert.equal(two.conflict, null);
  assert.equal(two.ci, '');
  assert.equal(two.filesCut, true);
  assert.equal(three.user, 'ghost');
  assert.equal(three.ci, 'pending', 'pending stays out of "CI running": it is mostly a run waiting for approval');
  assert.equal('ciPending' in three, false);
  assert.equal(four.ci, 'ok');
  assert.deepEqual([...four.lids], ['TEST-4']);
  assert.deepEqual([...four.deps], [1004]);
  assert.ok(t.graphCovers());
  assert.equal(ctx.rate.graphql.remaining, 4990);

  // Not due yet: nothing is asked.
  assert.equal(await t.loadGraph(Date.now()), 'idle');
  assert.equal(calls.pages, 3);

  // Due again with one PR changed and one closed: only that body is read, the closed one is forgotten.
  nodes[9] = node(10, { updatedAt: 'u2', body: 'BUILD-1 now' });
  nodes.pop();
  t.graph.at = 0;
  calls.bodyPrs.length = 0;
  assert.equal(await t.loadGraph(Date.now()), 'done');
  assert.deepEqual(calls.bodyPrs, [10]);
  assert.deepEqual([...ctx.applied[ctx.applied.length - 1][9].lids], ['BUILD-1']);
  assert.equal(t.bodies[230], undefined);
});

test('the PR list is shown even when its descriptions fail to load', async () => {
  const nodes = [node(1), node(2)];
  const working = fakeGraphql(nodes, { pages: 0, bodies: 0, bodyPrs: [] });
  const ctx = load({ fetch: async (url, init) => (JSON.parse(init.body).query.includes('pullRequests(')
    ? working(url, init) : { ok: false, status: 502, headers: { get: () => null }, json: async () => ({}) }) });
  assert.equal(await ctx.t.loadGraph(Date.now()), 'done');
  assert.equal(ctx.applied.length, 1);
  assert.equal(ctx.applied[0].length, 2);
  assert.ok(ctx.t.graphCovers());
  assert.equal(ctx.t.graph.busy, null, 'no progress left on screen');
});

test('a failed sweep hands back to the search path and waits before trying again', async () => {
  let asked = 0;
  const ctx = load({ fetch: async () => { asked++; return { ok: false, status: 502, headers: { get: () => null }, json: async () => ({}) }; } });
  const { t } = ctx;
  assert.equal(await t.loadGraph(Date.now()), 'error');
  assert.equal(t.graphCovers(), false);
  assert.equal(await t.loadGraph(Date.now()), 'error', 'not retried right away');
  assert.equal(asked, 1);
  assert.equal(ctx.applied.length, 0);
});

test('without a token nothing is asked', async () => {
  const ctx = load({ token: '' });
  assert.equal(await ctx.t.loadGraph(Date.now()), 'off');
  assert.equal(ctx.t.graphCovers(), false);
});
