/* Triage for maintainers: which open PRs do the same item twice, and which green PRs can go in one merge round
 * without touching each other's files.
 *
 * - Possible duplicates come from the PR list the pipeline loads anyway (titles, the first lines of each body, the
 *   coordination issue's PR column): no extra calls, with or without a token.
 * - With a token, one GraphQL sweep every GRAPH_MS (about 11 calls of 100 PRs, 1 point each of the 5,000 an hour)
 *   brings every open PR with its files, CI result and whether it merges cleanly. It feeds the pipeline and the merge
 *   queue as well: no 1,000-result search cap, no 100-PR CI lists, conflicts read live instead of from the coordination
 *   issue. Bodies are most of the bytes, so a body is fetched only when its PR changed since the last sweep.
 *
 * Same-file overlap is a pessimistic stand-in for a merge conflict: two PRs that touch different parts of one file
 * often merge cleanly, but two PRs with no file in common never conflict with each other.
 * Loaded after pipeline.js; uses rate, blockedUntil, noteLimit, activeToken, scrubDeep, store, esc, ago, fmt, plain,
 * $, CATS, GH, REPO of index.html, pipe/applyGraph of pipeline.js and latestCoord.
 */
'use strict';

const GRAPH_MS = 2 * 60 * 1000;           // a full sweep this often while the tab is in use
const GRAPH_RETRY_MS = 5 * 60 * 1000;     // after a failed sweep the search path takes over this long
const GRAPH_PAGE = 100;
const GRAPH_MAX_PAGES = 30;               // up to 3,000 open PRs
const GRAPH_FILES = 100;                  // files per PR; a PR with more can't be placed in a round
const BODY_BATCH = 50;                    // bodies per GraphQL call
const BODY_MAX_BATCHES = 25;              // per sweep: the first one fetches every body (about 21 calls)
const GRAPH_KEY = 'graph:v1';
const BODIES_KEY = 'graph:bodies:v1';
const BODY_LEAD_LINES = 3;
const MAX_REFS = 8;
const DUP_LISTED = 12;
const ROUND_LISTED = 40;
const HOT_LISTED = 8;
const ITEM_RE = /\b([A-Z][A-Z0-9]{1,7})-(\d{1,5})\b/g;
const DEPENDS_RE = /depends on:?([^\n]*)/gi;
const STAGE_RANK = { ready: 0, green: 1, running: 2, waiting: 3, failing: 4, rebase: 5, draft: 6 };

const graph = { at: 0, failAt: 0, ok: false, pages: 0, bodyCalls: 0 };
let bodies = {};                          // n → { u: updatedAt, lids, deps }
let triageHtml = '';

/* ================= references in a PR ================= */
const refIds = text => [...new Set([...String(text || '').matchAll(ITEM_RE)].map(m => m[0]))].slice(0, MAX_REFS);

/** Item IDs and `Depends on` PRs a body names; the title's IDs are read from the title itself. */
function bodyRefs(body) {
  const text = String(body || '');
  const lead = text.split('\n').map(l => l.trim()).filter(Boolean).slice(0, BODY_LEAD_LINES).join(' ');
  const deps = [...text.matchAll(DEPENDS_RE)].flatMap(m => (m[1].match(/#\d+/g) || []).map(s => Number(s.slice(1))));
  return { lids: refIds(lead), deps: [...new Set(deps)].slice(0, MAX_REFS) };
}

/** ID prefixes the coordination issue uses (SHADER, TEST, …), so UTF-8 or SHA-256 in a title are not items. */
function itemPrefixes(coord) {
  const set = new Set(CATS.map(c => c.key));
  for (const cat of (coord && coord.categories) || []) {
    for (const r of cat.rows || []) {
      const m = plain(r.ID || '').match(/([A-Z][A-Z0-9]+)-\d+/);
      if (m) set.add(m[1]);
    }
  }
  return set;
}

/** PR number → item IDs, from the coordination issue's tables and its "Ready for review" list. */
function coordItems(coord) {
  const map = new Map();
  const add = (n, id) => { if (!map.has(n)) map.set(n, new Set()); map.get(n).add(id); };
  for (const cat of (coord && coord.categories) || []) {
    for (const r of cat.rows || []) {
      const id = (plain(r.ID || '').match(/[A-Z][A-Z0-9]+-\d+/) || [])[0];
      if (id) for (const m of plain(r.PR || '').matchAll(/#(\d+)/g)) add(Number(m[1]), id);
    }
  }
  for (const [n, r] of Object.entries((coord && coord.reviewPrs) || {})) if (r.item) add(Number(n), r.item);
  return map;
}

/** The items a PR works on: what the coordination issue says, else the title's IDs, else the first one in the body. */
function itemsOf(p, prefixes, listed) {
  if (listed.has(p.n)) return [...listed.get(p.n)];
  const known = ids => (ids || []).filter(id => prefixes.has(id.split('-')[0]));
  const fromTitle = known(p.tids || refIds(p.title));
  return fromTitle.length ? fromTitle : known(p.lids).slice(0, 1);
}

/* ================= analysis ================= */
const byStage = (a, b) => (STAGE_RANK[a.stage] ?? 9) - (STAGE_RANK[b.stage] ?? 9) || String(a.created).localeCompare(String(b.created));

/** Items with open PRs from more than one author; one author's several PRs for an item are split work, not counted. */
function findDuplicates(prs, coord) {
  const prefixes = itemPrefixes(coord);
  const listed = coordItems(coord);
  const groups = new Map();
  for (const p of prs) {
    for (const id of itemsOf(p, prefixes, listed)) {
      if (!groups.has(id)) groups.set(id, []);
      groups.get(id).push(p);
    }
  }
  const dups = [];
  let series = 0;
  for (const [id, list] of groups) {
    if (list.length < 2) continue;
    if (new Set(list.map(p => p.user)).size < 2) { series++; continue; }
    const onRow = p => listed.has(p.n) && listed.get(p.n).has(id);
    dups.push({ id, prs: [...list].sort((a, b) => onRow(b) - onRow(a) || byStage(a, b)), listed: list.filter(onRow).map(p => p.n) });
  }
  dups.sort((a, b) => b.prs.length - a.prs.length || a.id.localeCompare(b.id, 'en', { numeric: true }));
  return { groups: dups, prs: dups.reduce((s, g) => s + g.prs.length, 0), series, mapped: prs.filter(p => itemsOf(p, prefixes, listed).length).length };
}

/**
 * The next merge round: green PRs that merge cleanly into main and wait on no open PR, oldest first (checked ones
 * before the rest), each taken only when it shares no file with one taken before it. Merging all of them leaves every
 * other one of them mergeable.
 */
function conflictMap(prs) {
  const withFiles = prs.filter(p => Array.isArray(p.files));
  if (!withFiles.length) return null;
  const active = withFiles.filter(p => !p.draft);
  const owners = new Map();
  for (const p of active) for (const f of p.files) { if (!owners.has(f)) owners.set(f, []); owners.get(f).push(p); }
  const green = p => p.stage === 'green' || p.stage === 'ready';
  const hot = [...owners].filter(([, list]) => list.length > 1)
    .sort((a, b) => b[1].length - a[1].length)
    .slice(0, HOT_LISTED)
    .map(([path, list]) => ({ path, n: list.length, green: list.filter(green).length }));
  const overlaps = p => { const s = new Set(); for (const f of p.files) for (const q of owners.get(f)) if (q !== p) s.add(q.n); return s.size; };
  const isolated = active.filter(p => p.files.length && overlaps(p) === 0).length;
  const open = new Set(prs.map(p => p.n));
  const candidates = active
    .filter(p => green(p) && p.conflict === false && !p.filesCut && !(p.deps || []).some(d => open.has(d)))
    .sort((a, b) => (b.stage === 'ready') - (a.stage === 'ready') || String(a.created).localeCompare(String(b.created)));
  const taken = new Set();
  const round = [];
  for (const p of candidates) {
    if (p.files.some(f => taken.has(f))) continue;
    p.files.forEach(f => taken.add(f));
    round.push(p);
  }
  return {
    round, hot, isolated,
    candidates: candidates.length,
    waiting: candidates.length - round.length,
    conflicting: withFiles.filter(p => p.conflict === true).length,
    maxShare: hot.length ? hot[0].n : 0,
  };
}

/* ================= GraphQL (token only) ================= */
/** One GraphQL call; partial answers (a PR deleted between calls) keep their data. */
async function graphql(query, variables = {}) {
  const token = activeToken();
  if (!token) throw new Error('GraphQL needs a token');
  if (Date.now() < (blockedUntil.graphql || 0)) throw new RateLimitError('graphql', blockedUntil.graphql);
  apiCalls++;
  const res = await fetch('https://api.github.com/graphql', {
    method: 'POST',
    headers: { Authorization: 'Bearer ' + token, 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, variables }),
  });
  if (res.headers.get('x-ratelimit-resource')) {
    rate.graphql = {
      remaining: Number(res.headers.get('x-ratelimit-remaining')),
      limit: Number(res.headers.get('x-ratelimit-limit')),
      reset: Number(res.headers.get('x-ratelimit-reset')) * 1000,
    };
  }
  if (!res.ok) {
    if (res.status === 401) { tokenRefused = true; console.warn('[dashboard] GitHub refused the saved token; continuing without it'); }
    if (res.status === 403 || res.status === 429) noteLimit(res, 'graphql');
    throw new Error('GitHub answered ' + res.status + ' to GraphQL');
  }
  const body = await res.json();
  if (!body.data) throw new Error((body.errors && body.errors[0] && body.errors[0].message) || 'GraphQL answered without data');
  if (body.errors) console.warn('[dashboard] GraphQL answered in part', body.errors.slice(0, 3));
  return scrubDeep(body.data);
}

const [GQL_OWNER, GQL_NAME] = REPO.split('/');
const PRS_QUERY = `query($owner:String!,$name:String!,$cursor:String){ repository(owner:$owner,name:$name){
  pullRequests(states:OPEN, first:${GRAPH_PAGE}, after:$cursor, orderBy:{field:CREATED_AT, direction:ASC}){
    totalCount pageInfo{ hasNextPage endCursor }
    nodes{ number title url isDraft createdAt updatedAt mergeable changedFiles
      author{ login avatarUrl } comments{ totalCount }
      files(first:${GRAPH_FILES}){ nodes{ path } }
      commits(last:1){ nodes{ commit{ statusCheckRollup{ state } } } } } } } }`;

const CI_OF = { SUCCESS: 'ok', FAILURE: 'bad', ERROR: 'bad', PENDING: 'pending', EXPECTED: 'pending' };

function pickGraphPr(node) {
  const files = ((node.files && node.files.nodes) || []).map(f => f.path);
  const commit = node.commits && node.commits.nodes[0] && node.commits.nodes[0].commit;
  const ci = CI_OF[commit && commit.statusCheckRollup ? commit.statusCheckRollup.state : ''] || '';
  const b = bodies[node.number];
  return {
    n: node.number, title: node.title, user: node.author ? node.author.login : 'ghost', avatar: node.author ? node.author.avatarUrl : '',
    url: node.url, created: node.createdAt, updated: node.updatedAt, draft: !!node.isDraft, comments: node.comments ? node.comments.totalCount : 0,
    conflict: node.mergeable === 'CONFLICTING' ? true : node.mergeable === 'MERGEABLE' ? false : null,
    ci, ciPending: ci === 'pending', files, filesCut: node.changedFiles > files.length,
    tids: refIds(node.title), lids: b ? b.lids : [], deps: b ? b.deps : [],
  };
}

/** Bodies of PRs that changed since their body was read; the rest keep what was read before. */
async function loadBodies(nodes) {
  const due = nodes.filter(n => !bodies[n.number] || bodies[n.number].u !== n.updatedAt).map(n => n.number);
  for (let i = 0, calls = 0; i < due.length && calls < BODY_MAX_BATCHES; i += BODY_BATCH, calls++) {
    const batch = due.slice(i, i + BODY_BATCH);
    const fields = batch.map(n => `p${n}: pullRequest(number:${n}){ number updatedAt body }`).join(' ');
    const data = await graphql(`query($owner:String!,$name:String!){ repository(owner:$owner,name:$name){ ${fields} } }`, { owner: GQL_OWNER, name: GQL_NAME });
    graph.bodyCalls++;
    for (const pr of Object.values(data.repository || {})) {
      if (pr && pr.number) bodies[pr.number] = { u: pr.updatedAt, ...bodyRefs(pr.body) };
    }
  }
  const open = new Set(nodes.map(n => String(n.number)));
  bodies = Object.fromEntries(Object.entries(bodies).filter(([n]) => open.has(n)));
  store.set(BODIES_KEY, bodies);
}

const graphDue = now => !!activeToken() && now - graph.at >= GRAPH_MS && now - graph.failAt >= GRAPH_RETRY_MS;
/** The search path runs only when there is no token or the sweep has not worked lately. */
const graphCovers = () => !!activeToken() && graph.ok && Date.now() - graph.failAt >= GRAPH_RETRY_MS;

/** Sweep every open PR when due. Returns 'off' (no token), 'idle' (not due), 'done' or 'error'. */
async function loadGraph(now) {
  if (!activeToken()) return 'off';
  if (!graphDue(now)) return graph.ok ? 'idle' : 'error';
  const nodes = [];
  try {
    let cursor = null;
    for (let page = 0; page < GRAPH_MAX_PAGES; page++) {
      const data = await graphql(PRS_QUERY, { owner: GQL_OWNER, name: GQL_NAME, cursor });
      const prs = data.repository && data.repository.pullRequests;
      if (!prs) throw new Error('GraphQL returned no pull requests');
      nodes.push(...prs.nodes.filter(Boolean));
      graph.pages = page + 1;
      if (!prs.pageInfo.hasNextPage) break;
      cursor = prs.pageInfo.endCursor;
    }
    await loadBodies(nodes);
  } catch (e) {
    console.error('[dashboard] GraphQL sweep failed; the search path takes over for now', e);
    graph.failAt = Date.now();
    graph.ok = false;
    return 'error';
  }
  graph.at = Date.now();
  graph.ok = true;
  store.set(GRAPH_KEY, { at: graph.at });
  applyGraph(nodes.map(pickGraphPr));
  return 'done';
}

/* ================= render ================= */
const stageColor = key => (PIPE_STAGES[STAGE_INDEX[key]] || {}).color || 'var(--dim)';
const stageLabel = key => (PIPE_STAGES[STAGE_INDEX[key]] || {}).label || '';

function triageRow(p, note) {
  return `<a href="${esc(p.url)}" target="_blank" rel="noopener" title="${esc(p.title)}">
    <span class="n">#${esc(p.n)}</span><span class="t"><i class="sdot" style="background:${stageColor(p.stage)}" title="${esc(stageLabel(p.stage))}"></i>${esc(p.title)} <span style="color:var(--muted)">@${esc(p.user)}</span>${note ? ` <span class="tag">${esc(note)}</span>` : ''}</span><span class="when">${esc(ago(p.created))}</span></a>`;
}

function duplicatesCard(d) {
  const body = d.groups.length
    ? d.groups.slice(0, DUP_LISTED).map(g => `<div class="dup">
        <div class="dup-head"><b>${esc(g.id)}</b> <span>${g.prs.length} PRs by ${new Set(g.prs.map(p => p.user)).size} authors</span></div>
        <div class="prs">${g.prs.map(p => triageRow(p, g.listed.includes(p.n) ? 'in the table' : p.draft ? 'draft' : '')).join('')}</div></div>`).join('')
      + (d.groups.length > DUP_LISTED ? `<p class="footnote">And ${d.groups.length - DUP_LISTED} more items.</p>` : '')
    : '<span style="color:var(--muted)">No item has open PRs from more than one author.</span>';
  return `<div class="card"><div class="subhead" style="margin-top:0">Possible duplicates: one item, several authors</div>${body}
    <p class="footnote">An item is what the coordination issue lists for a PR, else an ID in its title or in the first lines of its description. The PR the coordination table lists comes first, then the furthest along. ${d.series ? `${d.series} more item${d.series > 1 ? 's have' : ' has'} several PRs from one author: split work, not listed.` : ''} ${fmt(d.mapped)} open PRs name an item.</p></div>`;
}

function roundCard(m) {
  if (!m) return tokenCard('Next round');
  const list = m.round.length
    ? `<div class="prs">${m.round.slice(0, ROUND_LISTED).map(p => triageRow(p, p.stage === 'ready' ? 'checked' : '')).join('')}</div>
       ${m.round.length > ROUND_LISTED ? `<p class="footnote">And ${m.round.length - ROUND_LISTED} more; “Copy numbers” copies all of them.</p>` : ''}`
    : '<span style="color:var(--muted)">No green PR merges cleanly right now.</span>';
  return `<div class="card"><div class="subhead" style="margin-top:0;display:flex;justify-content:space-between;gap:10px">Next round: ${fmt(m.round.length)} PRs that touch no common file
      ${m.round.length ? '<button class="linkbtn" type="button" id="copyRound">Copy numbers</button>' : ''}</div>${list}
    <p class="footnote">Green, mergeable into ${esc(BRANCH)}, waiting on no open PR; checked ones first, then oldest. Each is taken only when it shares no file with one taken before, so merging them all leaves the rest of this list mergeable. ${m.waiting ? `${fmt(m.waiting)} more green PRs share a file with one of these and fit a later round.` : ''} Sharing a file doesn't always mean a conflict, so this is on the safe side.</p></div>`;
}

function hotCard(m) {
  if (!m) return tokenCard('Hot files');
  const rows = m.hot.length
    ? m.hot.map(h => `<div class="hot"><code title="${esc(h.path)}">${esc(h.path)}</code><span class="bar"><i style="width:${(h.n / (m.maxShare || 1) * 100).toFixed(1)}%"></i></span><span class="num">${fmt(h.n)} <small>${fmt(h.green)} green</small></span></div>`).join('')
    : '<span style="color:var(--muted)">No file is changed by more than one open PR.</span>';
  return `<div class="card"><div class="subhead" style="margin-top:0">Hot files: changed by the most open PRs</div>${rows}
    <p class="footnote">Every merge that touches one of these can make the other PRs on it conflict. ${fmt(m.isolated)} open PRs share no file with any other: they can go in at any time without causing a rebase.</p></div>`;
}

function tokenCard(title) {
  const text = activeToken()
    ? (graph.failAt && Date.now() - graph.failAt < GRAPH_RETRY_MS
      ? 'The last GraphQL sweep failed (see the browser console); it is tried again within a few minutes.'
      : 'Waiting for the first GraphQL sweep with your token: every open PR with its changed files, about a dozen calls.')
    : "Needs every open PR's changed files: about a dozen GraphQL calls, which GitHub only answers with a token. A read-only token for public repositories is enough; it stays in this browser.";
  return `<div class="card"><div class="subhead" style="margin-top:0">${esc(title)}</div>
    <p style="color:var(--muted);margin:6px 0 10px">${esc(text)}</p>
    ${activeToken() ? '' : '<button class="btn pb-token" type="button">Add a token</button>'}</div>`;
}

function renderTriage() {
  const box = $('#triageBody');
  if (!box || !pipe.prs.size) return;
  const prs = [...pipe.prs.values()];
  const d = findDuplicates(prs, latestCoord);
  const m = graphCovers() ? conflictMap(prs) : null;
  const tiles = [
    ['bad', d.groups.length, `items with PRs by several authors (${fmt(d.prs)} PRs)`],
    ['ok', m ? m.round.length : null, 'green PRs for the next round'],
    ['', m ? m.waiting : null, 'green, but share a file with one of those'],
    ['bad', m ? m.conflicting : null, 'conflict with main now'],
    ['', m ? m.isolated : null, 'share no file with another PR'],
    ['wait', m ? m.maxShare : null, m && m.hot.length ? `PRs change ${m.hot[0].path.split('/').pop()}` : 'PRs on the hottest file'],
  ].map(([cls, n, label]) => `<div class="stat ${cls}"><b>${fmt(n)}</b><span>${esc(label)}</span></div>`).join('');
  const html = `<div class="tiles">${tiles}</div>
    <div class="queue-grid">${roundCard(m)}${duplicatesCard(d)}${hotCard(m)}</div>
    <p class="footnote">${m ? `From your token's sweep ${esc(ago(graph.at))}; it runs every ${Math.round(GRAPH_MS / 60000)} minutes while this tab is in use.` : 'Duplicates come from the same load as the pipeline. The merge round and the hot files need a token.'}</p>`;
  if (html === triageHtml) return;   // nothing changed: leave the DOM alone
  triageHtml = html;
  box.innerHTML = html;
  const copy = $('#copyRound');
  if (copy && m) copy.addEventListener('click', () => copyText(m.round.map(p => '#' + p.n).join(' '), `${m.round.length} PR numbers copied`));
}

async function copyText(text, done) {
  try {
    await navigator.clipboard.writeText(text);
    toast(done);
  } catch (e) {
    console.warn('[dashboard] clipboard failed, falling back', e);
    const ta = document.createElement('textarea');
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand('copy');
    ta.remove();
    toast(ok ? done : 'Copy failed; select the numbers manually');
  }
}

function restoreTriage() {
  const saved = store.get(GRAPH_KEY);
  if (saved && saved.at) { graph.at = saved.at; graph.ok = true; }
  const b = store.get(BODIES_KEY);
  if (b && typeof b === 'object') bodies = b;
}
restoreTriage();
