const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

// The PR pipeline's layout (pipeline.js) without a browser: every stage keeps room for its dots at any width.
const dir = process.env.DASHBOARD_DIR || join(__dirname, '..', 'dashboard');
const source = readFileSync(join(dir, 'pipeline.js'), 'utf8');
const EXPORTS = '\n;globalThis.t = { pipe, pipeLayout, positions, PIPE_STAGES, COL_MIN_W, COLS_MIN_WIDTH, PITCH, DOT };';

/** pipeline.js with `counts` open PRs per stage, e.g. { waiting: 820, draft: 40 }. */
function load(counts) {
  const ctx = { console };
  vm.runInNewContext(source + EXPORTS, ctx);
  let n = 1;
  for (const [stage, k] of Object.entries(counts)) {
    for (let i = 0; i < k; i++, n++) ctx.t.pipe.prs.set(n, { n, stage, created: new Date(Date.UTC(2026, 0, 1) + n * 60000).toISOString() });
  }
  return ctx.t;
}

// Roughly the open PRs on 2026-11-22: most wait for a first CI approval.
const BUSY = { draft: 60, waiting: 820, running: 40, failing: 90, rebase: 70, green: 13, ready: 4 };

function assertDotsFit(t, layout) {
  const pos = t.positions(layout);
  assert.equal(pos.size, t.pipe.prs.size);
  for (const col of layout.cols) {
    assert.ok(col.perRow > 1, `${col.key}: ${col.perRow} dot per row`);
    const inCol = [...t.pipe.prs.values()].filter(p => p.stage === col.key);
    for (const p of inCol) {
      const at = pos.get(p.n);
      assert.ok(at.x >= col.x && at.x + t.DOT <= col.x + col.w, `${col.key}: dot #${p.n} at x=${at.x} outside ${col.x}..${col.x + col.w}`);
      assert.ok(at.y >= col.y && at.y + t.DOT <= col.y + col.h, `${col.key}: dot #${p.n} at y=${at.y} outside ${col.y}..${col.y + col.h}`);
    }
  }
}

test('a phone gets one row per stage, dots wrapping over the full width', () => {
  const t = load(BUSY);
  const layout = t.pipeLayout(330, 20);
  assert.equal(layout.orient, 'rows');
  assert.ok(layout.width <= 330, 'no sideways scrolling');
  assertDotsFit(t, layout);
  const all = [...layout.cols, layout.main, layout.merged];
  for (let i = 1; i < all.length; i++) assert.ok(all[i].y >= all[i - 1].y + all[i - 1].h, `row ${i} overlaps the one above`);
  assert.ok(layout.height < 1400, `phone layout is ${layout.height}px tall`);
});

test('the width where 820 waiting PRs used to stack one per row now uses rows', () => {
  const t = load(BUSY);
  for (const width of [600, 760, 900, t.COLS_MIN_WIDTH - 1]) {
    const layout = t.pipeLayout(width, 20);
    assert.equal(layout.orient, 'rows', `width ${width}`);
    assertDotsFit(t, layout);
  }
});

test('wide screens keep the columns, none narrower than the minimum', () => {
  const t = load(BUSY);
  for (const width of [t.COLS_MIN_WIDTH, 1200, 1600]) {
    const layout = t.pipeLayout(width, 20);
    assert.equal(layout.orient, 'cols', `width ${width}`);
    for (const c of [...layout.cols, layout.merged]) assert.ok(c.w >= t.COL_MIN_W - 1e-9, `${c.key} is ${c.w}px at ${width}`);
    assert.ok(layout.width <= width + 1, `columns need ${layout.width}px of ${width}`);
    assertDotsFit(t, layout);
  }
});

test('a merged dot flies into the main lane', () => {
  const t = load(BUSY);
  for (const width of [330, 1400]) {
    const layout = t.pipeLayout(width, 0);
    const m = layout.main;
    for (let i = 0; i < 20; i++) {
      const to = layout.target();
      assert.ok(to.x >= m.x && to.x <= m.x + m.w && to.y >= m.y && to.y <= m.y + m.h, `width ${width}: (${to.x}, ${to.y}) outside main`);
    }
  }
});
