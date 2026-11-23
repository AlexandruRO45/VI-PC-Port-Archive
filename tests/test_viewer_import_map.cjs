'use strict';

// The map viewer loads three.js from a CDN in a page that can read the user's local world data, so every module the
// import map can reach has to carry a hash. No network: this reads viewer/index.html only.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const html = fs.readFileSync(path.join(__dirname, '..', 'viewer', 'index.html'), 'utf8');
const map = JSON.parse(html.match(/<script type="importmap">([\s\S]*?)<\/script>/)[1]);
const moduleSource = html.match(/<script type="module">([\s\S]*?)<\/script>/)[1];

// GLTFLoader.js imports this one itself, so no import statement in the page names it.
const TRANSITIVE = ['utils/BufferGeometryUtils.js'];

function resolve(specifier) {
    if (map.imports[specifier]) return map.imports[specifier];
    const prefix = Object.keys(map.imports).find(key => key.endsWith('/') && specifier.startsWith(key));
    assert.ok(prefix, `${specifier} is not in the import map`);
    return map.imports[prefix] + specifier.slice(prefix.length);
}

test('every remote import target is pinned to an exact version', () => {
    for (const url of Object.values(map.imports)) {
        assert.match(url, /^https:\/\/cdn\.jsdelivr\.net\/npm\/three@\d+\.\d+\.\d+\//);
    }
});

test('every integrity entry is a SHA-384 hash of a URL the import map can resolve to', () => {
    const roots = Object.values(map.imports);
    for (const [url, hash] of Object.entries(map.integrity)) {
        assert.ok(roots.some(root => url === root || (root.endsWith('/') && url.startsWith(root))), url);
        assert.match(hash, /^sha384-[A-Za-z0-9+/]{64}$/);
    }
});

test('every module the page imports, and what those modules import, has an integrity entry', () => {
    const imported = [...moduleSource.matchAll(/^import .* from "([^"]+)";$/gm)].map(match => match[1]);
    assert.ok(imported.includes('three'));
    const specifiers = [...imported, ...TRANSITIVE.map(file => `three/addons/${file}`)];
    for (const specifier of specifiers) {
        const url = resolve(specifier);
        assert.ok(map.integrity[url], `${url} has no integrity entry`);
    }
});
